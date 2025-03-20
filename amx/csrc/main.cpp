#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include "amx_linear.hpp"
#include "simd_jit_tests.hpp"
#include "simple_perf.hpp"

#include "../../include/linux_perf.hpp"

//===============================================================
// XTILE initialize on Linux
#ifndef _GNU_SOURCE
#    define _GNU_SOURCE /* See feature_test_macros(7) */
#endif
#include <sys/syscall.h> /* For SYS_xxx definitions */
#include <unistd.h>

#define XFEATURE_XTILECFG       17
#define XFEATURE_XTILEDATA      18
#define XFEATURE_MASK_XTILECFG  (1 << XFEATURE_XTILECFG)
#define XFEATURE_MASK_XTILEDATA (1 << XFEATURE_XTILEDATA)
#define XFEATURE_MASK_XTILE     (XFEATURE_MASK_XTILECFG | XFEATURE_MASK_XTILEDATA)
#define ARCH_GET_XCOMP_PERM     0x1022
#define ARCH_REQ_XCOMP_PERM     0x1023

inline bool initXTILE() {
    unsigned long bitmask = 0;
    long status = syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask);
    if (0 != status)
        return false;
    if (bitmask & XFEATURE_MASK_XTILEDATA)
        return true;

    status = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
    if (0 != status)
        return false;  // XFEATURE_XTILEDATA setup is failed, TMUL usage is not allowed
    status = syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask);

    // XFEATURE_XTILEDATA setup is failed, can't use TMUL
    if (0 != status || !(bitmask & XFEATURE_MASK_XTILEDATA))
        return false;

    // XFEATURE_XTILEDATA set successfully, TMUL usage is allowed
    printf("\e[33m initXTILE success!\e[0m \n");
    return true;
}

static bool init_once = initXTILE();

#pragma message "Recompiling ..."

// QKV (normal-linear is a special case while both N[1] and N[2] are 0):
//
//               weight_fp16 x 3 + IO_bf16/fp16
//   dyn-quant:  weight_int8 x 3(per-OC scales) + IO_bf16/fp16
//
using ov::intel_cpu::AMXLinearKernel;
using ov::intel_cpu::func_cvt_output;
using ov::intel_cpu::func_quantize;
using ov::intel_cpu::func_dequantize;
using ov::intel_cpu::func_amx;
using ov::intel_cpu::func_amx2;
using ov::intel_cpu::amx_tile_configer;
using ov::intel_cpu::make_cacheable;
using ov::intel_cpu::scratch_buff;
using ov::intel_cpu::func_amx_repack_Btile;
using ov::intel_cpu::blocking_1d;


struct AMXQKVLinear2 {
    blocking_1d blk_bn;
    blocking_1d blk_bk;

    const int w_dtype_size = 2;
    int REG_BLK_N = 32;
    int REG_BLK_K = 16 * (4/w_dtype_size);

    int Ns[3];
    int N;
    int K;
    int N0;
    int N01;
    int N012;
    std::shared_ptr<int8_t> w_buff;

    struct dst_info_t {
        int id;
        int n;
    };
    std::vector<dst_info_t> dst_info;

    func_amx2::Ptr amx_mm;
    amx_tile_configer::Ptr amx_tcfg;
    func_cvt_output::Ptr cvt_output;

    uint8_t* get_ctemp(size_t new_size) {
        thread_local std::shared_ptr<uint8_t> ctemp;
        thread_local size_t cur_size = 0;
        if (cur_size < new_size) {
            ctemp = alloc_cache_aligned<uint8_t>(new_size);
            cur_size = new_size;
        }
        return ctemp.get();
    }

    template <class F>
    void split_N3(int64_t n0, int64_t n1, const F& f) {
        int64_t stop = Ns[0];
        if (n0 < stop) {
            auto ne = std::min(n1, stop);
            f(0, 0, n0, ne);
            n0 = ne;
            if (n0 >= n1)
                return;
        }
        stop += Ns[1];
        if (n0 < stop) {
            auto ne = std::min(n1, stop);
            f(1, Ns[0], n0, ne);
            n0 = ne;
            if (n0 >= n1)
                return;
        }
        f(2, Ns[0] + Ns[1], n0, n1);
    }

    AMXQKVLinear2(PlainTensor* ws, int w_count) {
        ASSERT(w_count == 3);
        Ns[0] = ws[0].shape(0);
        Ns[1] = ws[1].shape(0);
        Ns[2] = ws[2].shape(0);
        
        N0 = Ns[0];
        N01 = Ns[0] + Ns[1];
        N = N012 = Ns[0] + Ns[1] + Ns[2];
        K = ws[0].shape(1);

        amx_mm = make_cacheable<func_amx2>(K, func_amx2::TMUL_BF16);
        cvt_output = make_cacheable<func_cvt_output>(DTYPE_FP32, DTYPE_BF16);
        amx_tcfg = make_cacheable<amx_tile_configer>();

        blocking_1d blk_bk;
        blk_bn.reset_size(N, getenv("BN", 768));
        blk_bk.reset_size(K, K);
        w_buff = pack_weights(ws, w_count, w_dtype_size, false, blk_bn, blk_bk);

        dst_info.resize(N/REG_BLK_N);
        auto* pdi = dst_info.data();
        for(int n = 0; n < Ns[0]; n += REG_BLK_N, pdi++) *pdi = {0, n};
        for(int n = 0; n < Ns[1]; n += REG_BLK_N, pdi++) *pdi = {1, n};
        for(int n = 0; n < Ns[2]; n += REG_BLK_N, pdi++) *pdi = {2, n};
    }

    void forward(const PlainTensor& x, PlainTensor* ys) {
        int64_t M = x.shape(0);
        //auto prof = LinuxPerf::Profile("forward", M);
        auto i0 = 0;
        auto i1 = 1;
        auto i2 = 2;

        uint8_t* ptr_y[3] = {
            reinterpret_cast<uint8_t*>(ys[i0].mutable_data()),
            reinterpret_cast<uint8_t*>(ys[i1].mutable_data()),
            reinterpret_cast<uint8_t*>(ys[i2].mutable_data()),
        };
        int64_t stride_y[3] = {ys[i0].strides(0), ys[i1].strides(0), ys[i2].strides(0)};
        auto p_x = reinterpret_cast<const uint8_t*>(x.data());
        auto stride_x = x.strides(0);
        const int BM = 32;

        auto nblkN = blk_bn.size();
        auto nblkM = (M + BM - 1) / BM;
        parallel_nt_static(0, [&](int ithr, int nthr) {
            //auto prof = LinuxPerf::Profile("amx", ithr);

            // ping-pong buffer for float32 32x32 C buffer
            thread_local std::shared_ptr<uint8_t> cbuff = alloc_cache_aligned<uint8_t>(81920);

            int last_config = -1;

            func_amx2::call_args args;
            args.k_tiles = K / REG_BLK_K;
            args.strideA = stride_x;    // in bytes

            // first post-ops is an fake/dummy one
            args.pC0 = cbuff.get();
            args.pC = cbuff.get() + 40960;
            args.pdst = cbuff.get();
            args.stride_dst = 128;

            int64_t i0, i1;
            splitter(static_cast<int64_t>(nblkN * nblkM), nthr, ithr, i0, i1);
            for (int64_t i = i0; i < i1; i++) {
                int64_t m0, m1;
                // allocate task reusing same sub-weight to same thread
                m0 = (i % nblkM) * BM;
                m1 = std::min(m0 + BM, M);

                auto blk_n = i / nblkM;
                int n0 = blk_bn[blk_n].first;
                int n1 = blk_bn[blk_n].second;
                // DEBUG_LOG(m0, m1, n0, n1)

                auto* pB0 = reinterpret_cast<const uint8_t*>(w_buff.get()) + n0 * K * 2;

                args.valid_m = m1 - m0;
                args.pA = p_x + m0*stride_x;

                (*amx_tcfg)(args.valid_m, &last_config);

                auto* pdi = &dst_info[n0/REG_BLK_N];
                args.pB = pB0;

                for(int n = n0; n < n1; n += REG_BLK_N, pdi++) {
                    //args.pC = pC + (n - n0)*sizeof(float);
                    (*amx_mm)(&args);
                    args.pB += args.k_tiles * 2048;

                    // prepare post-ops task params for in next tmul
                    auto stride_dst = stride_y[pdi->id];
                    // swap ping-pong buffer
                    std::swap(args.pC0, args.pC);
                    args.pdst = ptr_y[pdi->id] + pdi->n*2 + m0 * stride_dst;
                    args.stride_dst = stride_dst;
                    args.dst_valid_m = m1 - m0;
                }
            }
            // for last post-ops
            (*amx_mm)(&args);
        });
    }
};

struct AMXQKVLinear {
    AMXLinearKernel linear;
    int64_t Ns[3];

    func_cvt_output::Ptr cvt_output;
    func_quantize::Ptr quantize_bf16_i8sym;
    func_quantize::Ptr quantize_fp16_i8sym;
    func_dequantize::Ptr dequantize_bf16;
    func_dequantize::Ptr dequantize_fp16;

    template <class F>
    void split_N3(int64_t n0, int64_t n1, const F& f) {
        auto stop = Ns[0];
        if (n0 < stop) {
            auto ne = std::min(n1, stop);
            f(0, 0, n0, ne);
            n0 = ne;
            if (n0 >= n1)
                return;
        }
        stop += Ns[1];
        if (n0 < stop) {
            auto ne = std::min(n1, stop);
            f(1, Ns[0], n0, ne);
            n0 = ne;
            if (n0 >= n1)
                return;
        }
        f(2, Ns[0] + Ns[1], n0, n1);
    }

    int tmul_type;
    int acc_dtype;
    int o_type;
    int o_type_bytes;
    int w_dtype_size;
    int w_count;

    AMXQKVLinear(PlainTensor* ws, int w_count) : w_count(w_count) {
        // https://docs.python.org/3/library/array.html#module-array
        auto K = ws[0].shape(1);

        const uint8_t* pw[3];
        ssize_t stride_w[3];
        for (int i = 0; i < 3; i++) {
            Ns[i] = 0;
            pw[i] = nullptr;
            stride_w[i] = 0;
            if (i < w_count) {
                ASSERT(ws[i].ndim() == 2);
                ASSERT(ws[i].shape(1) == K);
                Ns[i] = ws[i].shape(0);
                pw[i] = reinterpret_cast<const uint8_t*>(ws[i].data());
                stride_w[i] = ws[i].strides(0);
            }
        }
        int w_dtype = 0;
        switch (ws[0].get_precision()) {
        case element::bf16:
            tmul_type = func_amx::TMUL_BF16;
            w_dtype = DTYPE_BF16;
            acc_dtype = DTYPE_FP32;
            o_type = DTYPE_BF16;
            w_dtype_size = 2;
            o_type_bytes = 2;
            break;
        case element::f16:
            tmul_type = func_amx::TMUL_FP16;
            w_dtype = DTYPE_FP16;
            acc_dtype = DTYPE_FP32;
            o_type = DTYPE_FP16;
            w_dtype_size = 2;
            o_type_bytes = 2;
            break;
        case element::i8:
            tmul_type = func_amx::TMUL_SSD;
            w_dtype = DTYPE_INT8;
            acc_dtype = DTYPE_INT32;
            o_type = DTYPE_INT32;
            w_dtype_size = 1;
            o_type_bytes = 4;
            break;
        default:
            ASSERT(0);
        }
        cvt_output = make_cacheable<func_cvt_output>(acc_dtype, o_type);
        quantize_bf16_i8sym = make_cacheable<func_quantize>(DTYPE_BF16);
        quantize_fp16_i8sym = make_cacheable<func_quantize>(DTYPE_FP16);
        dequantize_bf16 = make_cacheable<func_dequantize>(DTYPE_BF16);
        dequantize_fp16 = make_cacheable<func_dequantize>(DTYPE_FP16);

        auto N0 = Ns[0];
        auto N01 = Ns[0] + Ns[1];
        auto N012 = Ns[0] + Ns[1] + Ns[2];

        linear.init(K, N012, tmul_type, w_dtype, [&](int k, int n, size_t& stride) -> const void* {
            int id = 0;
            if (n >= N01) {
                id = 2;
                n -= N01;
            } else if (n >= N0) {
                id = 1;
                n -= N0;
            }
            // DEBUG_LOG(id, N0, N01, k, n);
            stride = stride_w[id];
            return pw[id] + k * w_dtype_size + n * stride;
        });
    }

    // dynamic quantize
    void forward_dyn_quant_i8(const PlainTensor& x, PlainTensor* ys, PlainTensor* w_scales) {
        // weight-dequantization scale is also provided (suppose weight is INT8)
        auto M = x.shape(0);
        auto i0 = 0;
        auto i1 = std::min(1, w_count - 1);
        auto i2 = std::min(2, w_count - 1);

        //DEBUG_LOG(M, x.shape(1), ys[0].shape(0), ys[0].shape(1), w_scales[0].shape(0), w_scales[0].shape(1))
        int x_type = 0;
        switch(x.get_precision()) {
            case element::bf16:
                x_type = DTYPE_BF16;
                break;
            case element::f16:
                x_type = DTYPE_FP16;
                break;
            default:
                ASSERT(0, "Unsupported input dtype");
        }
        uint8_t* ptr_y[3] = {
            reinterpret_cast<uint8_t*>(ys[i0].mutable_data()),
            reinterpret_cast<uint8_t*>(ys[i1].mutable_data()),
            reinterpret_cast<uint8_t*>(ys[i2].mutable_data()),
        };
        int64_t stride_y[3] = {ys[i0].strides(0), ys[i1].strides(0), ys[i2].strides(0)};
        const float* ptr_wscales[3] = {
            reinterpret_cast<const float*>(w_scales[i0].data()),
            reinterpret_cast<const float*>(w_scales[i1].data()),
            reinterpret_cast<const float*>(w_scales[i2].data()),
        };

        auto p_x = reinterpret_cast<const uint8_t*>(x.data());
        auto stride_x = x.strides(0);
        // per-token quantize input into int8

        static scratch_buff<int8_t> buff_x_quant;
        static scratch_buff<float> buff_x_scale;

        buff_x_quant.ensure_size(x.shape(0) * x.shape(1));
        buff_x_scale.ensure_size(x.shape(0));
        auto* x_quant = buff_x_quant.buff.get();
        int x_quant_stride = x.shape(1);
        auto* x_scale = buff_x_scale.buff.get();

        if (x_type == DTYPE_BF16) {
            parallel_for(x.shape(0), [&](int m0, int m1) {
                auto* psrc = reinterpret_cast<const int8_t*>(x.data()) + m0*x.strides(0);
                auto* pdst = x_quant + m0*x_quant_stride;
                (*quantize_bf16_i8sym)(psrc, x.strides(0), pdst, x_quant_stride, x_scale + m0, m1 - m0, x.shape(1));
            });
            //show<int>(x_quant, 16);
        }
        if (x_type == DTYPE_FP16) {
            parallel_for(x.shape(0), [&](int m0, int m1) {
                auto* psrc = reinterpret_cast<const int8_t*>(x.data()) + m0*x.strides(0);
                auto* pdst = x_quant + m0*x_quant_stride;
                (*quantize_fp16_i8sym)(psrc, x.strides(0), pdst, x_quant_stride, x_scale + m0, m1 - m0, x.shape(1));
            });
        }
        //py::array_t<int8_t> ret({x.shape(0), x.shape(1)}, x_quant);

        // post-ops will convert int32 to fp32, dequantize then store to destination buffer
        linear.execute(M,
                       x_quant,
                       x_quant_stride,
                       [&](int64_t m0, int64_t m1, int64_t n0, int64_t n1, void* pvC, int64_t strideC) {
                           auto* pC = reinterpret_cast<int32_t*>(pvC);
                           split_N3(n0, n1, [&](int idx, int base_n, int ns0, int ns1) {
                               // dequantize int32 acc results & convert to BF16 or FP16
                               auto stride_dst = stride_y[idx];
                               auto* pdst = ptr_y[idx] + (ns0 - base_n) * sizeof(int16_t) + m0 * stride_dst;
                               auto* pscalew = ptr_wscales[idx] + ns0 - base_n;
                               if (x_type == DTYPE_FP16) {
                                    (*dequantize_fp16)(pC + (ns0 - n0), strideC, pdst, stride_dst, x_scale + m0, pscalew,
                                                       m1 - m0,
                                                       ns1 - ns0);
                               }
                               if (x_type == DTYPE_BF16) {
                                    (*dequantize_bf16)(pC + (ns0 - n0), strideC, pdst, stride_dst, x_scale + m0, pscalew,
                                                       m1 - m0,
                                                       ns1 - ns0);
                               }
                               //(*cvt_output)(m1 - m0, ns1 - ns0, psrc, strideC, pdst, stride_dst);
                           });
                       });
        return;
    }

    void forward(const PlainTensor& x, PlainTensor* ys) {
        auto M = x.shape(0);
        auto i0 = 0;
        auto i1 = std::min(1, w_count - 1);
        auto i2 = std::min(2, w_count - 1);

        uint8_t* ptr_y[3] = {
            reinterpret_cast<uint8_t*>(ys[i0].mutable_data()),
            reinterpret_cast<uint8_t*>(ys[i1].mutable_data()),
            reinterpret_cast<uint8_t*>(ys[i2].mutable_data()),
        };
        int64_t stride_y[3] = {ys[i0].strides(0), ys[i1].strides(0), ys[i2].strides(0)};
        auto p_x = reinterpret_cast<const uint8_t*>(x.data());
        auto stride_x = x.strides(0);
        linear.execute(M,
                       p_x,
                       stride_x,
                       [&](int64_t m0, int64_t m1, int64_t n0, int64_t n1, void* pvC, int64_t strideC) {
                           auto* pC = reinterpret_cast<uint32_t*>(pvC);
                           split_N3(n0, n1, [&](int idx, int base_n, int ns0, int ns1) {
                               auto stride_dst = stride_y[idx];
                               auto* pdst = ptr_y[idx] + (ns0 - base_n) * o_type_bytes + m0 * stride_dst;
                               auto* psrc = reinterpret_cast<const uint8_t*>(pC + (ns0 - n0));
                               (*cvt_output)(m1 - m0, ns1 - ns0, psrc, strideC, pdst, stride_dst);
                           });
                       });
        return;
    }
};

void set_nthr(int nthr) {
    //omp_set_dynamic(0);         // Explicitly disable dynamic teams
    omp_set_num_threads(nthr);  // Use 4 threads for all consecutive parallel regions
}

int DTYPE_of(const py::array& src) {
    switch(src.dtype().char_()) {
        case 'h': return DTYPE_BF16;
        case 'e': return DTYPE_FP16;
        case 'b': return DTYPE_INT8;
        case 'f': return DTYPE_FP32;
        default:
            ASSERT(0, "unknown dtype char : ", src.dtype().char_())
    }
    return DTYPE_FP32;
}

void test_quant_i8(const py::array& src, py::array_t<int8_t> dst, py::array_t<float> scale) {
    auto func = make_cacheable<func_quantize>(DTYPE_of(src));
    auto M = src.shape(0);
    auto K = src.shape(1);
    //py::array_t<int8_t> dst({M, K});
    //py::array_t<float> scale({M, M*0 + 1});
    parallel_for(M, [&](int m0, int m1) {
        auto* psrc = reinterpret_cast<const int8_t*>(src.data()) + m0*src.strides(0);
        auto* pdst = reinterpret_cast<int8_t*>(dst.mutable_data()) + m0*dst.strides(0);
        (*func)(psrc, src.strides(0), pdst, dst.strides(0), scale.mutable_data() + m0, m1 - m0, K);
    });
    //return {dst, scale};
}

void test() {
    auto jit = SIMDJit::create([&](SIMDJit* jit, SReg count, SReg i) {
        SReg temp, x;
        temp = (1<<(count - i)) - 1;
        jit->return_(temp);
    });

    std::cout << (*jit)(120, 120) << std::endl;
    std::cout << (*jit)(120, 119) << std::endl;
    std::cout << (*jit)(120, 118) << std::endl;
    std::cout << (*jit)(120, 117) << std::endl;
}

py::array_t<float> test_amx_repack_B(const py::array_t<float>& src) {
    auto func = make_cacheable<func_amx_repack_Btile>(false);
    py::array_t<float> dst({16, 16});
    (*func)(src.data(), src.strides(0), dst.mutable_data());
    return dst;
}

PYBIND11_MODULE(PACKAGE_NAME, m) {
    m.doc() = R"pbdoc(
    )pbdoc";

    py::class_<AMXQKVLinear>(m, "AMXQKVLinear")
        .def(py::init([](std::vector<py::array> ws){
            return new AMXQKVLinear(reinterpret_cast<PlainTensor*>(&ws[0]), ws.size());
        }))
        .def("forward", [](AMXQKVLinear& self, const py::array& x, std::vector<py::array> ys) {
            return self.forward(reinterpret_cast<const PlainTensor&>(x), reinterpret_cast<PlainTensor*>(&ys[0]));
        })
        .def("forward_dyn_quant_i8", [](AMXQKVLinear& self, const py::array& x, std::vector<py::array> ys, std::vector<py::array> w_scales) {
            return self.forward_dyn_quant_i8(reinterpret_cast<const PlainTensor&>(x), reinterpret_cast<PlainTensor*>(&ys[0]), reinterpret_cast<PlainTensor*>(&w_scales[0]));
        });

    py::class_<AMXQKVLinear2>(m, "AMXQKVLinear2")
        .def(py::init([](std::vector<py::array> ws){
            return new AMXQKVLinear2(reinterpret_cast<PlainTensor*>(&ws[0]), ws.size());
        }))
        .def("forward", [](AMXQKVLinear2& self, const py::array& x, std::vector<py::array> ys) {
            return self.forward(reinterpret_cast<const PlainTensor&>(x), reinterpret_cast<PlainTensor*>(&ys[0]));
        });

    m.def("set_nthr", set_nthr);
    m.def("test", test);

    m.def("test_amx_repack_B", test_amx_repack_B);
    m.def("test_quant_i8", test_quant_i8);
    m.def("set_nthr", set_nthr);

    m.def("simd_test_basic", simd_test_basic);
    m.def("simd_test_tput", simd_test_tput);
    m.def("simd_test_printreg", simd_test_printreg);
}
