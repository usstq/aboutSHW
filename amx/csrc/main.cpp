#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "amx_linear.hpp"
#include "simd_jit_tests.hpp"
#include "simple_perf.hpp"

#include "../../include/linux_perf.hpp"
namespace py = pybind11;

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
using ov::intel_cpu::make_cacheable;
using ov::intel_cpu::scratch_buff;
using ov::intel_cpu::func_amx_repack_Btile;
using ov::intel_cpu::blocking_1d;

enum class element {
    bf16,
    f16,
    i8
};

struct PlainTensor: public py::array {
    element get_precision() const {
        switch(dtype().char_()) {
            case 'e':
                return element::f16;
            break;
            case 'h':
                return element::bf16;
            break;
            case 'b':
                return element::i8;
            break;
        }
        return element::i8;
    }
};



inline std::shared_ptr<int8_t> pack_weights(PlainTensor* ws, int w_count, int w_dtype_size,
                                     blocking_1d& blk_bn,
                                     blocking_1d& blk_bk) {
    static auto amx_repack_Btile = make_cacheable<func_amx_repack_Btile>();

    auto K = ws[0].shape(1);
    std::vector<int> runsum_N(w_count);
    int sumN = 0;
    for(int i = 0; i < w_count; i++) {
        runsum_N[i] = sumN;
        sumN += ws[i].shape(0);
    }

    auto get_weight = [&](int k, int n, size_t& stride) -> const void* {
        int id;
        for(id = w_count - 1; id > 0; id --) {
            if (n >= runsum_N[id]) {
                n -= runsum_N[id];
                break;
            }
        }
        // DEBUG_LOG(id, N0, N01, k, n);
        stride = ws[id].strides(0);
        return reinterpret_cast<const uint8_t*>(ws[id].data()) + k * w_dtype_size + n * stride;
    };

    auto REG_BLK_N = 32;
    auto REG_BLK_K = 32;

    auto w_buff = alloc_cache_aligned<int8_t>(sumN * K * (w_dtype_size));
    auto* pdst_base = reinterpret_cast<uint8_t*>(w_buff.get());
    blk_bn.for_each_blk(0, [&](int ibn, int n0, int n1) {
        blk_bk.for_each_blk([&](int ibk, int k0, int k1) {
            auto* pdst = pdst_base + (n0 * K + (n1 - n0) * k0) / (REG_BLK_N * REG_BLK_K) * 2048;
            for (int n = n0; n < n1; n += REG_BLK_N) {
                for (int k = k0; k < k1; k += REG_BLK_K) {
                    size_t stride_src;
                    auto* psrc0 = get_weight(k, n, stride_src);
                    (*amx_repack_Btile)(psrc0, stride_src, pdst);

                    auto* psrc1 = get_weight(k, n + 16, stride_src);
                    (*amx_repack_Btile)(psrc1, stride_src, pdst + 1024);
                    pdst += 2048;
                }
            }
        });
    });
    return w_buff;
}

class amx_tile_configer {
    struct config_t {
        uint8_t palette_id = 1;
        uint8_t startRow = 0;
        uint8_t reserved[14] = {0};
        uint16_t cols[16] = {0};
        uint8_t rows[16] = {0};
        void set(int tmm_id, uint8_t r, uint16_t c) {
            rows[tmm_id] = r;
            cols[tmm_id] = c;
        }
    } __attribute__((__packed__));
    config_t tile_cfgs[32];
    std::shared_ptr<SIMDJit> jit_cfg;

public:
    using Ptr = std::shared_ptr<const amx_tile_configer>;

    amx_tile_configer() {
        jit_cfg = SIMDJit::create([&](SIMDJit* jit, SReg p) {
            jit->if_(
                p == 0,
                [&] {
                    jit->tilerelease();
                },
                [&] {
                    jit->ldtilecfg(jit->ptr[p.r64()]);
                });
        });
        for (int m = 0; m < 32; m++) {
            auto& cfg = tile_cfgs[m];
            auto valid_M = (m == 0) ? 32 : m;
            auto rows0 = 16;
            auto rows1 = 16;
            if (valid_M > 16) {
                rows0 = 16;
                rows1 = valid_M - 16;
            } else {
                //  both A0 & A1 load from same memory, to avoid code-regeneration
                rows0 = rows1 = valid_M;
            }
            cfg.set(0, rows0, 64);  // C00:0
            cfg.set(1, rows0, 64);  // C01:1
            cfg.set(2, rows1, 64);  // C10:2
            cfg.set(3, rows1, 64);  // C11:3

            cfg.set(4, rows0, 64);  // A0
            cfg.set(5, rows1, 64);  // A1

            cfg.set(6, 16, 64);  // B0
            cfg.set(7, 16, 64);  // B1
        }
    }
    void operator()(int i, int* last_i = nullptr) const {
        if (i >= 0) i = i % 32;
        if (last_i && (*last_i == i))
            return;
        (*jit_cfg)(i < 0 ? nullptr : &tile_cfgs[i]);
        if (last_i) *last_i = i;
    }
};

// AMX based matmul in unit of 32x32(fp16/bf16) 64x32(int8)
class func_amx2 {
    std::shared_ptr<SIMDJit> jit_amx_mm2x2_small_K;
    using TMUL_TYPE = ov::intel_cpu::TMUL_TYPE;
    TMUL_TYPE tmul_type;
public:
    struct call_args {
        const uint8_t* pA;  // bfloat16/int8
        int64_t strideA;    // in bytes
        const uint8_t* pB;  // bfloat16/int8
        uint8_t* pC;        // float32/int32
        // the post-ops source & dest
        uint8_t* pC0;

        const uint8_t* pdst;
        int64_t stride_dst;

        int64_t k_tiles;  // K / 32
        int64_t valid_m;
    };

    const int REG_BLK_N = 32;
    const int REG_BLK_K;

    using Ptr = std::shared_ptr<const func_amx2>;
    static constexpr int TMUL_FP16 = 1;  // FP16 => FP32
    static constexpr int TMUL_BF16 = 2;  // BF16 => FP32
    static constexpr int TMUL_SSD = 3;   // INT8 INT8 => FP32
    func_amx2(int M_hint, int type) : REG_BLK_K(16 * ((type < TMUL_SSD) ? 2 : 4)) {
        switch (type) {
        case TMUL_FP16:
            tmul_type = TMUL_TYPE::FP16;
            break;
        case TMUL_BF16:
            tmul_type = TMUL_TYPE::BF16;
            break;
        case TMUL_SSD:
            tmul_type = TMUL_TYPE::SSD;
            break;
        }
        // small K, no need to prefetch next subB along K dimension
        // in this case, prefetch output buffer instead of weights
        // because K is small, and we are expecting big M or N,
        // anyway C matrix would be the biggest one.
        //
        jit_amx_mm2x2_small_K = SIMDJit::create([&](SIMDJit* jit, SReg pargs) {
            SReg reg_A_addr, reg_A_stride, reg_B_addr, reg_C_addr, reg_C_stride;
            SReg reg_dst, reg_ktiles, reg_A1_addr, reg_B_stride;
            SReg reg_stride_dst;
            reg_A_addr.load(pargs + offsetof(call_args, pA));
            reg_A_stride.load(pargs + offsetof(call_args, strideA));
            reg_B_addr.load(pargs + offsetof(call_args, pB));
            reg_C_addr.load(pargs + offsetof(call_args, pC0));
            reg_dst.load(pargs + offsetof(call_args, pdst));
            reg_stride_dst.load(pargs + offsetof(call_args, stride_dst));
            //reg_prefetch = reg_prefetch + reg_prefetch_stride*16;

            reg_C_stride = 32*sizeof(float);

            reg_ktiles.load(pargs + offsetof(call_args, k_tiles));
            TReg tmmC00(0), tmmC01(1), tmmC10(2), tmmC11(3);
            TReg tmmA0(4), tmmA1(5), tmmB0(6), tmmB1(7);
            jit->tilezero(tmmC00);
            jit->tilezero(tmmC01);
            jit->tilezero(tmmC10);
            jit->tilezero(tmmC11);

            reg_A1_addr = reg_A_addr + reg_A_stride * 8;
            reg_A1_addr = reg_A1_addr + reg_A_stride * 8;

            reg_B_stride.load(pargs + offsetof(call_args, valid_m));
            jit->cmp(reg_B_stride, 16);
            jit->cmovle(reg_A1_addr, reg_A_addr);

            reg_B_stride = 64;
            auto const_A_steps = 64;
            Zmm d0, d1, d2;

            static int NOPC = getenv("NOPC", 1);
            static int XXX = getenv("XXX", 0);

            auto kernel =  [&] {
                //                B: 1x2 tiles
                // A : 2x1 tiles  C: 2x2 tiles
                if (XXX & 1) tmmA0.load(reg_A_addr + reg_A_stride);
                if (XXX & 1) tmmB0.load(reg_B_addr + reg_B_stride);
#if 0
                // interleave post-ops with AMX TMUL
                jit->prefetchwt1(jit->ptr[reg_dst.r64() + 2*reg_stride_dst.r64()]);
                d0.load(reg_C_addr);
                d1.load(reg_C_addr + 64);
                jit->vcvtne2ps2bf16(d2, d1, d0);
                d2.store(reg_dst);
                //reg_C_addr = reg_C_addr + reg_C_stride;
                reg_dst = reg_dst + reg_stride_dst;
                
#endif
                jit->nop(NOPC);
                reg_B_addr = reg_B_addr + 1024;

                if (XXX & 2) jit->tmul(tmmC00, tmmA0, tmmB0, tmul_type);
                
                if (XXX & 8) jit->tmul(tmmC00, tmmC01, tmmC10, tmul_type);

                if (XXX & 1) tmmA1.load(reg_A1_addr + reg_A_stride);
                
                if (XXX & 2) jit->tmul(tmmC10, tmmA1, tmmB0, tmul_type);

                if (XXX & 8) jit->tmul(tmmC00, tmmC01, tmmC10, tmul_type);

                if (XXX & 1) tmmB1.load(reg_B_addr + reg_B_stride);
#if 0
                jit->prefetchwt1(jit->ptr[reg_dst.r64() + 2*reg_stride_dst.r64()]);
                // interleave post-ops with AMX TMUL
                d0.load(reg_C_addr + reg_C_stride);
                d1.load(reg_C_addr + reg_C_stride + 64);
                jit->vcvtne2ps2bf16(d2, d1, d0);
                d2.store(reg_dst);
                reg_C_addr = reg_C_addr + 2*reg_C_stride;
                reg_dst = reg_dst + reg_stride_dst;
#endif
                jit->nop(NOPC);

                if (XXX & 2) jit->tmul(tmmC01, tmmA0, tmmB1, tmul_type);
                if (XXX & 2) jit->tmul(tmmC11, tmmA1, tmmB1, tmul_type);

                if (XXX & 8) jit->tmul(tmmC00, tmmC01, tmmC10, tmul_type);
                if (XXX & 8) jit->tmul(tmmC00, tmmC01, tmmC10, tmul_type);

                reg_A_addr = reg_A_addr + const_A_steps;
                reg_A1_addr = reg_A1_addr + const_A_steps;
                reg_B_addr = reg_B_addr + 1024;
                reg_ktiles--;
            };

            auto unroll = getenv("UNROLL", 0);
            if(unroll) {
                for(int i = 0; i < unroll; i++) {
                    kernel();
                }
            } else {
                jit->do_while_(reg_ktiles > 0, kernel);
            }

            if (XXX & 4) {
                reg_C_addr.load(pargs + offsetof(call_args, pC));
                //reg_C_stride.load(pargs + offsetof(call_args, strideC));

                tmmC00.store(reg_C_addr + reg_C_stride);
                tmmC01.store(reg_C_addr + reg_C_stride + 64);
                reg_C_addr = reg_C_addr + reg_C_stride * 8;
                reg_C_addr = reg_C_addr + reg_C_stride * 8;
                tmmC10.store(reg_C_addr + reg_C_stride);
                tmmC11.store(reg_C_addr + reg_C_stride + 64);
            }
        });
    }

    // functor can have many functions, but all of them have to be const
    void operator()(call_args* arg) const {
        (*jit_amx_mm2x2_small_K)(arg);
    }
};

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
        amx_mm = make_cacheable<func_amx2>(256, func_amx2::TMUL_BF16);
        cvt_output = make_cacheable<func_cvt_output>(DTYPE_FP32, DTYPE_BF16);
        amx_tcfg = make_cacheable<amx_tile_configer>();

        Ns[0] = ws[0].shape(0);
        Ns[1] = ws[1].shape(0);
        Ns[2] = ws[2].shape(0);
        
        N0 = Ns[0];
        N01 = Ns[0] + Ns[1];
        N = N012 = Ns[0] + Ns[1] + Ns[2];
        K = ws[0].shape(1);

        blocking_1d blk_bk;
        blk_bn.reset_size(N, getenv("BN", 256));
        blk_bk.reset_size(K, K);
        w_buff = pack_weights(ws, w_count, w_dtype_size, blk_bn, blk_bk);

        dst_info.resize(N/REG_BLK_N);
        auto* pdi = dst_info.data();
        for(int n = 0; n < Ns[0]; n += REG_BLK_N, pdi++) *pdi = {0, n};
        for(int n = 0; n < Ns[1]; n += REG_BLK_N, pdi++) *pdi = {1, n};
        for(int n = 0; n < Ns[2]; n += REG_BLK_N, pdi++) *pdi = {2, n};
    }

    void forward(const PlainTensor& x, PlainTensor* ys) {
        auto prof = LinuxPerf::Profile("amx");
        int64_t exec_cnt = 0;
        int64_t M = x.shape(0);
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
            // ping-pong buffer for float32 32x32 C buffer
            thread_local std::shared_ptr<uint8_t> cbuff = alloc_cache_aligned<uint8_t>(8192);

            int last_config = -1;

            func_amx2::call_args args;
            args.k_tiles = K / REG_BLK_K;
            args.strideA = stride_x;    // in bytes

            // first post-ops is an fake/dummy one
            args.pC0 = cbuff.get();
            args.pC = cbuff.get() + 4096;
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

                /*
                args.strideC = (n1 - n0) * sizeof(float);
                if ((args.strideC % 128) == 0)
                    args.strideC += 64;
                auto* pC = get_ctemp(BM * args.strideC);
                */

                auto* pB0 = reinterpret_cast<const uint8_t*>(w_buff.get()) + n0 * K * 2;

                args.valid_m = m1 - m0;
                args.pA = p_x + m0*stride_x;

                (*amx_tcfg)(args.valid_m, &last_config);

                auto* pdi = &dst_info[n0/REG_BLK_N];
                args.pB = pB0;

                for(int n = n0; n < n1; n += REG_BLK_N, pdi++) {
                    //args.pC = pC + (n - n0)*sizeof(float);
                    (*amx_mm)(&args);
                    exec_cnt ++;
                    args.pB += args.k_tiles * 2048;

                    // prepare post-ops task params for in next tmul
                    auto stride_dst = stride_y[pdi->id];
                    // swap ping-pong
                    std::swap(args.pC0, args.pC);
                    //args.pC0 = args.pC;
                    //args.strideC0 = args.strideC;
                    args.pdst = ptr_y[pdi->id] + pdi->n*2 + m0 * stride_dst;
                    args.stride_dst = stride_dst;
                }
                /*
                //post process BM x (n0-n1)
                split_N3(n0, n1, [&](int idx, int base_n, int ns0, int ns1) {
                    auto stride_dst = stride_y[idx];
                    auto* pdst = ptr_y[idx] + (ns0 - base_n) * 2 + m0 * stride_dst;
                    auto* psrc = reinterpret_cast<const uint8_t*>(pC + (ns0 - n0)*sizeof(float));
                    (*cvt_output)(m1 - m0, ns1 - ns0, psrc, args.strideC, pdst, stride_dst);
                });
                */
            }
            // for last post-ops
            (*amx_mm)(&args);
            exec_cnt ++;
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

        linear.init(K, N012, tmul_type, [&](int k, int n, size_t& stride) -> const void* {
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
    omp_set_dynamic(0);         // Explicitly disable dynamic teams
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
    auto func = make_cacheable<func_amx_repack_Btile>();
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
