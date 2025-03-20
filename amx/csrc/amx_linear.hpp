#include <atomic>
#include <mutex>

#ifdef PACKAGE_NAME
#include "parallel.hpp"
#include "simd_jit.hpp"

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

#endif
// this literal lacks of checks on N, make sure id composed of less than 5 chars
constexpr uint32_t operator "" _dt(const char* id, std::size_t N) {
    uint32_t a8 = 0;
    for(int i = 0; i < N; i++) {
        a8 = (a8 << 8) | (static_cast<uint32_t>(id[i]) & 0xFF);
    }
    return a8;
}

static constexpr auto DTYPE_INT32 = "i32"_dt;
static constexpr auto DTYPE_FP32 = "fp32"_dt;
static constexpr auto DTYPE_FP16 = "fp16"_dt;
static constexpr auto DTYPE_BF16 = "bf16"_dt;
static constexpr auto DTYPE_INT8 = "i8"_dt;

namespace ov {
namespace intel_cpu {

template<typename T>
std::shared_ptr<T> alloc_cache_aligned(int count) {
    auto ret = std::shared_ptr<T>(
           reinterpret_cast<T*>(aligned_alloc(64, count * sizeof(T))),
           [](void * p) { ::free(p); });
    return ret;
}

template <class T>
struct scratch_buff {
    std::shared_ptr<T> buff;
    size_t cur_count = 0;
    void ensure_size(size_t count) {
        if (cur_count < count) {
            buff = alloc_cache_aligned<T>(count);
            cur_count = count;
        }
    }
};

class blocking_1d {
    int64_t total_size;
    int64_t block_size;
    std::vector<std::pair<int64_t, int64_t>> blks;
    std::vector<int> blk_sizes;

public:
    // non-uniform size blocking
    void reset_blks(const std::vector<int64_t>& block_sizes) {
        int off = 0;
        blks.clear();
        blk_sizes.clear();
        for (int i = 0; i < block_sizes.size(); i++) {
            auto size = block_sizes[i];
            blks.emplace_back(off, off + size);
            blk_sizes.emplace_back(static_cast<int>(size));
            off += size;
        }
        total_size = off;
        block_size = 0;
    }

    // uniform sized blocking
    void reset_size(int64_t _total_size, int64_t _block_size) {
        total_size = _total_size;
        block_size = _block_size;
        blks.clear();
        blk_sizes.clear();
        for (int64_t i0 = 0; i0 < total_size; i0 += block_size) {
            int64_t i1 = std::min(i0 + block_size, total_size);
            blks.emplace_back(i0, i1);
            blk_sizes.emplace_back(i1 - i0);
        }
    }

    // specify count instead of block size, total_size will be evenly splitted
    void reset_cnt(int64_t _total_size, int64_t atom_size, int64_t _block_cnt) {
        total_size = _total_size;
        block_size = 0;
        blks.clear();
        blk_sizes.clear();
        blks.reserve(_block_cnt);
        for (int64_t i = 0; i < _block_cnt; i++) {
            int64_t n0;
            int64_t n1;
            auto work_groups = total_size / atom_size;
            splitter(work_groups, _block_cnt, i, n0, n1);
            n0 *= atom_size;
            n1 *= atom_size;
            blks.emplace_back(n0, n1);
            blk_sizes.emplace_back(n1 - n0);
        }
    }

    size_t size() {
        return blks.size();
    }

    std::pair<int64_t, int64_t>& operator[](int i) {
        return blks[i];
    }

    const std::vector<int>& sizes() const {
        return blk_sizes;
    }

    // overload support parallel
    template <class F>
    void for_each_blk(int nthreads, const F& f) {
        parallel_nt_static(nthreads, [&](int ithr, int nthr) {
            size_t start, end;
            splitter(blks.size(), nthr, ithr, start, end);
            if (end > start) {
                for (int i = start; i < end; i++) {
                    auto& bi = blks[i];
                    f(i, bi.first, bi.second);
                }
            }
        });
    }

    template <class F>
    void for_each_blk(const F& f) {
        for (int i = 0; i < blks.size(); i++) {
            auto& bi = blks[i];
            f(i, bi.first, bi.second);
        }
    }
    friend std::ostream& operator<<(std::ostream& os, const blocking_1d& rhs);
};

inline std::ostream& operator<<(std::ostream& os, const blocking_1d& rhs) {
    os << "blocking_1d(" << rhs.total_size << "," << rhs.block_size << ", {";
    const char* sep = "";
    for (auto& bi : rhs.blks) {
        os << sep << bi.first << "+" << bi.second - bi.first;
        sep = ",";
    }
    os << "})";
    return os;
}

////////////////////////////////////////////////////////////////////////////
class func_amx_repack_Btile {
    std::shared_ptr<SIMDJit> jit;
    const int stride_dst;

public:
    // const is required for functor
    using Ptr = std::shared_ptr<const func_amx_repack_Btile>;

    func_amx_repack_Btile(bool cvt_f16_to_bf16) : stride_dst(64) {
        // compile-time work
        jit = SIMDJit::create([&](SIMDJit* jit, SReg psrc, SReg stride_src, SReg pdst) {
            Ymm t[8];
            Ymm tt[8];

            for (int i = 0; i < 8; i += 2) {
                tt[i].load(psrc);
                tt[i + 1].load(psrc + stride_src);
                jit->vpunpckldq(t[i], tt[i], tt[i + 1]);
                jit->vpunpckhdq(t[i + 1], tt[i], tt[i + 1]);
                psrc = psrc + stride_src * 2;
            }

            uint8_t imm01 = get_imm8<2>(0, 1, 0, 1);
            uint8_t imm23 = get_imm8<2>(2, 3, 2, 3);

            jit->vshufps(tt[0], t[0], t[2], imm01);
            jit->vshufps(tt[1], t[0], t[2], imm23);
            jit->vshufps(tt[2], t[1], t[3], imm01);
            jit->vshufps(tt[3], t[1], t[3], imm23);
            jit->vshufps(tt[4], t[4], t[6], imm01);
            jit->vshufps(tt[5], t[4], t[6], imm23);
            jit->vshufps(tt[6], t[5], t[7], imm01);
            jit->vshufps(tt[7], t[5], t[7], imm23);

            jit->vperm2i128(t[0].ymm(), tt[0].ymm(), tt[4], 0x20);
            jit->vperm2i128(t[1].ymm(), tt[1].ymm(), tt[5], 0x20);
            jit->vperm2i128(t[2].ymm(), tt[2].ymm(), tt[6], 0x20);
            jit->vperm2i128(t[3].ymm(), tt[3].ymm(), tt[7], 0x20);
            jit->vperm2i128(t[4].ymm(), tt[0].ymm(), tt[4], 0x31);
            jit->vperm2i128(t[5].ymm(), tt[1].ymm(), tt[5], 0x31);
            jit->vperm2i128(t[6].ymm(), tt[2].ymm(), tt[6], 0x31);
            jit->vperm2i128(t[7].ymm(), tt[3].ymm(), tt[7], 0x31);
            for (int i = 0; i < 8; i++) {
                if (cvt_f16_to_bf16) {
                    jit->vcvtph2ps(t[i].zmm(), t[i]);
                    jit->vcvtneps2bf16(t[i], t[i].zmm());
                }
                t[i].store(pdst + stride_dst * i);
            }
            return jit;
        });
    }
    void operator()(const void* pvsrc, size_t stride_src, void* pvdst) const {
        auto* psrc = reinterpret_cast<const int8_t*>(pvsrc);
        auto* pdst = reinterpret_cast<int8_t*>(pvdst);
        (*jit)(psrc, stride_src, pdst);
        (*jit)(psrc + 8 * sizeof(float), stride_src, pdst + 8 * stride_dst);
        (*jit)(psrc + 8 * stride_src, stride_src, pdst + 8 * sizeof(float));
        (*jit)(psrc + 8 * stride_src + 8 * sizeof(float), stride_src, pdst + 8 * stride_dst + 8 * sizeof(float));
    }
};


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
    void operator()(int rows, int* last_rows = nullptr) const {
        if (rows >= 0) rows = rows % 32;
        if (last_rows && (*last_rows == rows))
            return;
        (*jit_cfg)(rows < 0 ? nullptr : &tile_cfgs[rows]);
        if (last_rows) *last_rows = rows;
    }
};

// AMX based matmul in unit of 32x32(fp16/bf16) 64x32(int8)
class func_amx2 {
    std::shared_ptr<SIMDJit> jit_amx_mm2x2_small_K;
    SIMDJit::TMUL_TYPE tmul_type;
public:
    struct call_args {
        const uint8_t* pA;  // bfloat16/int8
        int64_t strideA;    // in bytes
        const uint8_t* pB;  // bfloat16/int8
        uint8_t* pC;        // float32/int32
        int64_t valid_m;
        // the post-ops source & dest
        uint8_t* pC0;
        const uint8_t* pdst;
        int64_t stride_dst;
        int64_t dst_valid_m;    // how many rows to be copied out

        int64_t k_tiles;  // K / 32
    };

    const int REG_BLK_N = 32;
    const int REG_BLK_K;

    using Ptr = std::shared_ptr<const func_amx2>;
    static constexpr int TMUL_FP16 = 1;  // FP16 => FP32
    static constexpr int TMUL_BF16 = 2;  // BF16 => FP32
    static constexpr int TMUL_SSD = 3;   // INT8 INT8 => FP32
    func_amx2(int K, int type) : REG_BLK_K(16 * ((type < TMUL_SSD) ? 2 : 4)) {
        switch (type) {
        case TMUL_FP16:
            tmul_type = SIMDJit::TMUL_TYPE::FP16;
            break;
        case TMUL_BF16:
            tmul_type = SIMDJit::TMUL_TYPE::BF16;
            break;
        case TMUL_SSD:
            tmul_type = SIMDJit::TMUL_TYPE::SSD;
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
            SReg reg_stride_dst, dst_valid_m;
            reg_A_addr.load(pargs + offsetof(call_args, pA));
            reg_A_stride.load(pargs + offsetof(call_args, strideA));
            reg_B_addr.load(pargs + offsetof(call_args, pB));
            reg_C_addr.load(pargs + offsetof(call_args, pC0));
            reg_dst.load(pargs + offsetof(call_args, pdst));
            reg_stride_dst.load(pargs + offsetof(call_args, stride_dst));
            dst_valid_m.load(pargs + offsetof(call_args, dst_valid_m));

            reg_C_stride = 32*sizeof(float);

            //reg_ktiles.load(pargs + offsetof(call_args, k_tiles));
            reg_ktiles = K / REG_BLK_K;

            ASSERT(K % REG_BLK_K == 0);
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

            auto loop_cnt = (K / REG_BLK_K);
            auto post_op_count = (32 + loop_cnt - 1) / loop_cnt;

            ov::intel_cpu::KReg kmask;
            SReg stemp, zero;
            zero = 0;
            stemp = 0xFFFFFFFF;
            jit->kmovd(kmask, stemp.r32());
            // dst valid_m is smaller than 32
            jit->do_while_(reg_ktiles > 0, [&] {
                //                B: 1x2 tiles
                // A : 2x1 tiles  C: 2x2 tiles
                //
                // this kernel is heavily cache-load-bandwidth bounded
                // so when interleaving AVX512 post-ops, the zmm-register
                // loading-part can not run in-parallel.
                //
                tmmA0.load(reg_A_addr + reg_A_stride);
                tmmB0.load(reg_B_addr + reg_B_stride);

                // interleave post-ops with AMX TMUL
                for(int j = 0; j < post_op_count/2; j++) {
                    jit->prefetchwt1(jit->ptr[reg_dst.r64() + 2*reg_stride_dst.r64()]);
                    d0.load(reg_C_addr);
                    d1.load(reg_C_addr + 64);
                    if (tmul_type == SIMDJit::TMUL_TYPE::BF16) {
                        jit->vcvtne2ps2bf16(d2, d1, d0);
                        d2.store(reg_dst, VReg::LDST_TYPE::packed_i32, kmask);
                    } else if (tmul_type == SIMDJit::TMUL_TYPE::FP16) {
                        d0.store(reg_dst, VReg::LDST_TYPE::packed_fp16_fp32, kmask);
                        d1.store(reg_dst + 32, VReg::LDST_TYPE::packed_fp16_fp32, kmask);
                    }
                    dst_valid_m --;
                    jit->cmovz(stemp, zero);
                    jit->kmovd(kmask, stemp.r32());
                    reg_C_addr = reg_C_addr + reg_C_stride;
                    reg_dst = reg_dst + reg_stride_dst;
                }

                reg_B_addr = reg_B_addr + 1024;

                jit->tmul(tmmC00, tmmA0, tmmB0, tmul_type);
                tmmA1.load(reg_A1_addr + reg_A_stride);
                jit->tmul(tmmC10, tmmA1, tmmB0, tmul_type);
                tmmB1.load(reg_B_addr + reg_B_stride);

                // interleave post-ops with AMX TMUL
                for(int j = post_op_count/2; j < post_op_count; j++) {
                    jit->prefetchwt1(jit->ptr[reg_dst.r64() + 2*reg_stride_dst.r64()]);
                    d0.load(reg_C_addr);
                    d1.load(reg_C_addr + 64);
                    if (tmul_type == SIMDJit::TMUL_TYPE::BF16) {
                        jit->vcvtne2ps2bf16(d2, d1, d0);
                        d2.store(reg_dst, VReg::LDST_TYPE::packed_i32, kmask);
                    } else if (tmul_type == SIMDJit::TMUL_TYPE::FP16) {
                        d0.store(reg_dst, VReg::LDST_TYPE::packed_fp16_fp32, kmask);
                        d1.store(reg_dst + 32, VReg::LDST_TYPE::packed_fp16_fp32, kmask);
                    }
                    dst_valid_m --;
                    jit->cmovz(stemp, zero);
                    jit->kmovd(kmask, stemp.r32());
                    reg_C_addr = reg_C_addr + reg_C_stride;
                    reg_dst = reg_dst + reg_stride_dst;
                }

                jit->tmul(tmmC01, tmmA0, tmmB1, tmul_type);
                jit->tmul(tmmC11, tmmA1, tmmB1, tmul_type);

                reg_A_addr = reg_A_addr + const_A_steps;
                reg_A1_addr = reg_A1_addr + const_A_steps;
                reg_B_addr = reg_B_addr + 1024;
                reg_ktiles--;
            });

            reg_C_stride = 32*sizeof(float);
            reg_C_addr.load(pargs + offsetof(call_args, pC));
            tmmC00.store(reg_C_addr + reg_C_stride);
            tmmC01.store(reg_C_addr + reg_C_stride + 64);
            tmmC10.store(reg_C_addr + reg_C_stride + 32*sizeof(float) * 16);
            tmmC11.store(reg_C_addr + reg_C_stride + 32*sizeof(float) * 16 + 64);
        });
    }

    // functor can have many functions, but all of them have to be const
    void operator()(call_args* arg) const {
        (*jit_amx_mm2x2_small_K)(arg);
    }
};

// AMX based matmul in unit of 32x32(fp16/bf16) 64x32(int8)
class func_amx {
    std::shared_ptr<SIMDJit> jit_amx_mm2x2;
    std::shared_ptr<SIMDJit> jit_amx_mm1x2;
    std::shared_ptr<SIMDJit> jit_amx_mm2x2_small_K;

    int m_prefetch_Blines = 0;

    SIMDJit::TMUL_TYPE tmul_type;
    amx_tile_configer::Ptr tile_configer;

public:
    struct call_args {
        const uint8_t* pA;  // bfloat16/int8
        int64_t strideA;    // in bytes
        const uint8_t* pB;  // bfloat16/int8
        uint8_t* pC;        // float32/int32
        int64_t strideC;    // in bytes
        const uint8_t* prefetch;
        int64_t k_tiles;  // K / 32
        int64_t valid_m;
        int64_t do_accumulation;
        int64_t prefetch_stride;
    };

    const int REG_BLK_N = 32;
    const int REG_BLK_K;

    using Ptr = std::shared_ptr<const func_amx>;
    static constexpr int TMUL_FP16 = 1;  // FP16 => FP32
    static constexpr int TMUL_BF16 = 2;  // BF16 => FP32
    static constexpr int TMUL_SSD = 3;   // INT8 INT8 => FP32
    func_amx(int M_hint, int type) : REG_BLK_K(16 * ((type < TMUL_SSD) ? 2 : 4)) {
        switch (type) {
        case TMUL_FP16:
            tmul_type = SIMDJit::TMUL_TYPE::FP16;
            break;
        case TMUL_BF16:
            tmul_type = SIMDJit::TMUL_TYPE::BF16;
            break;
        case TMUL_SSD:
            tmul_type = SIMDJit::TMUL_TYPE::SSD;
            break;
        }
        tile_configer = make_cacheable<amx_tile_configer>();
        if (M_hint) {
            // next block size: 32 * N * sizeof(ov::bfloat16),
            // call number: N / 32 * M / 32
            // each call needs fetch: 32 * N * sizeof(ov::bfloat16) / (N / 32 * M / 32) = 32 * 1024 *
            // sizeof(ov::bfloat16) / M
            m_prefetch_Blines = 32768 * sizeof(uint16_t) / 64 / M_hint;
        }

        jit_amx_mm1x2 = SIMDJit::create([&](SIMDJit* jit, SReg pargs) {
            // auto dump = jit->get_disasm(true);

            SReg reg_A_addr, reg_A_stride, reg_B_addr, reg_C_addr, reg_C_stride;
            SReg reg_prefetch, reg_ktiles, reg_B_stride, reg_flag;
            reg_A_addr.load(pargs + offsetof(call_args, pA));
            reg_A_stride.load(pargs + offsetof(call_args, strideA));
            reg_B_addr.load(pargs + offsetof(call_args, pB));
            reg_C_addr.load(pargs + offsetof(call_args, pC));
            reg_C_stride.load(pargs + offsetof(call_args, strideC));
            reg_prefetch.load(pargs + offsetof(call_args, prefetch));
            reg_ktiles.load(pargs + offsetof(call_args, k_tiles));
            reg_flag.load(pargs + offsetof(call_args, do_accumulation));

            TReg tmmC00(0), tmmC01(1);
            TReg tmmA0(4), tmmB0(6), tmmB1(7);
            jit->if_(reg_flag & 1, [&] {
                jit->tilezero(tmmC00);
                jit->tilezero(tmmC01);
            });
            reg_B_stride = 64;
            jit->do_while_(reg_ktiles > 0, [&] {
                tmmA0.load(reg_A_addr + reg_A_stride);
                tmmB0.load(reg_B_addr + reg_B_stride);
                reg_B_addr = reg_B_addr + 1024;
                jit->tmul(tmmC00, tmmA0, tmmB0, tmul_type);

                tmmB1.load(reg_B_addr + reg_B_stride);
                jit->tmul(tmmC01, tmmA0, tmmB1, tmul_type);

                reg_A_addr = reg_A_addr + 64;
                reg_B_addr = reg_B_addr + 1024;
                reg_ktiles--;
            });
            jit->if_(reg_flag & 2, [&] {
                tmmC00.store(reg_C_addr + reg_C_stride);
                tmmC01.store(reg_C_addr + reg_C_stride + 64);
            });
        });

        jit_amx_mm2x2 = SIMDJit::create([&](SIMDJit* jit, SReg pargs) {
            SReg reg_A_addr, reg_A_stride, reg_B_addr, reg_C_addr, reg_C_stride;
            SReg reg_prefetch, reg_ktiles, reg_A1_addr, reg_B_stride;
            reg_A_addr.load(pargs + offsetof(call_args, pA));
            reg_A_stride.load(pargs + offsetof(call_args, strideA));
            reg_B_addr.load(pargs + offsetof(call_args, pB));
            reg_C_addr.load(pargs + offsetof(call_args, pC));
            reg_C_stride.load(pargs + offsetof(call_args, strideC));
            reg_prefetch.load(pargs + offsetof(call_args, prefetch));
            reg_ktiles.load(pargs + offsetof(call_args, k_tiles));
            TReg tmmC00(0), tmmC01(1), tmmC10(2), tmmC11(3);
            TReg tmmA0(4), tmmA1(5), tmmB0(6), tmmB1(7);
            jit->tilezero(tmmC00);
            jit->tilezero(tmmC01);
            jit->tilezero(tmmC10);
            jit->tilezero(tmmC11);

            auto num_PFB = m_prefetch_Blines;
            int cur_PFB = 0;

            reg_A1_addr = reg_A_addr + reg_A_stride * 8;
            reg_A1_addr = reg_A1_addr + reg_A_stride * 8;

            // reg_A1_addr = reg_A_addr if M <= 16 (to avoid tileloadd segmentfault)
            auto reg_tmp = reg_B_stride;
            reg_tmp.load(pargs + offsetof(call_args, valid_m));
            jit->cmp(reg_tmp, 16);
            jit->cmovle(reg_A1_addr, reg_A_addr);

            reg_tmp.load(pargs + offsetof(call_args, do_accumulation));
            jit->if_(reg_tmp != 0, [&] {
                auto reg_C1_addr = reg_tmp;
                tmmC00.load(reg_C_addr + reg_C_stride);
                tmmC01.load(reg_C_addr + reg_C_stride + 64);
                reg_C1_addr = reg_C_addr + reg_C_stride * 8;
                reg_C1_addr = reg_C1_addr + reg_C_stride * 8;
                tmmC10.load(reg_C1_addr + reg_C_stride);
                tmmC11.load(reg_C1_addr + reg_C_stride + 64);
            });

            reg_B_stride = 64;
            auto const_A_steps = 64;
            jit->do_while_(reg_ktiles > 0, [&] {
                //                B: 1x2 tiles
                // A : 2x1 tiles  C: 2x2 tiles
                tmmA0.load(reg_A_addr + reg_A_stride);
                tmmB0.load(reg_B_addr + reg_B_stride);
                reg_B_addr = reg_B_addr + 1024;

                jit->tmul(tmmC00, tmmA0, tmmB0, tmul_type);
                if (cur_PFB < num_PFB) {
                    jit->prefetcht2(jit->ptr[reg_prefetch.r64() + cur_PFB * 64]);
                    cur_PFB++;
                }

                tmmA1.load(reg_A1_addr + reg_A_stride);
                jit->tmul(tmmC10, tmmA1, tmmB0, tmul_type);
                if (cur_PFB < num_PFB) {
                    jit->prefetcht2(jit->ptr[reg_prefetch.r64() + cur_PFB * 64]);
                    cur_PFB++;
                }

                tmmB1.load(reg_B_addr + reg_B_stride);
                jit->tmul(tmmC01, tmmA0, tmmB1, tmul_type);
                if (cur_PFB < num_PFB) {
                    jit->prefetcht2(jit->ptr[reg_prefetch.r64() + cur_PFB * 64]);
                    cur_PFB++;
                }

                jit->tmul(tmmC11, tmmA1, tmmB1, tmul_type);
                if (cur_PFB < num_PFB) {
                    for (int pi = cur_PFB; pi < num_PFB; pi++) {
                        jit->prefetcht2(jit->ptr[reg_prefetch.r64() + pi * 64]);
                    }
                }

                reg_prefetch = reg_prefetch + 64 * num_PFB;
                reg_A_addr = reg_A_addr + const_A_steps;
                reg_A1_addr = reg_A1_addr + const_A_steps;
                reg_B_addr = reg_B_addr + 1024;
                reg_ktiles--;
            });

            tmmC00.store(reg_C_addr + reg_C_stride);
            tmmC01.store(reg_C_addr + reg_C_stride + 64);
            reg_C_addr = reg_C_addr + reg_C_stride * 8;
            reg_C_addr = reg_C_addr + reg_C_stride * 8;
            tmmC10.store(reg_C_addr + reg_C_stride);
            tmmC11.store(reg_C_addr + reg_C_stride + 64);
        });

    }

    // functor can have many functions, but all of them have to be const
    void matmul_1x2(call_args* arg) const {
        (*jit_amx_mm1x2)(arg);
    }

    // B matrix has been prepacked
    void matmul(int M,
                int N,
                int K,
                bool do_accumulation,
                const void* pvA,
                size_t strideA,
                const void* pvB,
                void* pvC,
                size_t strideC,
                const void* pfetchB) const {
        ASSERT((N % REG_BLK_N) == 0, N, "%", REG_BLK_N, " != 0");
        ASSERT((K % REG_BLK_K) == 0, K, "%", REG_BLK_K, " != 0");

        int cur_tile_cfg_id = -1;

        call_args args;
        args.strideA = strideA;
        args.strideC = strideC;
        args.k_tiles = K / REG_BLK_K;
        args.do_accumulation = do_accumulation;

        auto prefetch_step = m_prefetch_Blines * 64 * args.k_tiles;
        args.prefetch = reinterpret_cast<const uint8_t*>(pfetchB);

        auto* pA = reinterpret_cast<const uint8_t*>(pvA);
        auto* pB = reinterpret_cast<const uint8_t*>(pvB);
        auto* pC = reinterpret_cast<uint8_t*>(pvC);
        // accumulate [m0~m1, k0~k1] x [k0~k1, n0~n1] => [m0~m1, n0~n1]
        for (int64_t m = 0; m < M; m += 32, pC += 32 * strideC, pA += 32 * strideA) {
            auto valid_m = std::min(M - m, (int64_t)32);
            (*tile_configer)(valid_m, &cur_tile_cfg_id);

            args.pB = pB;
            args.pA = pA;
            args.valid_m = valid_m;
            for (int n = 0; n < N; n += REG_BLK_N) {
                args.pC = pC + n * sizeof(float);
                //args.pC = pC;
                (*jit_amx_mm2x2)(&args);
                args.pB += args.k_tiles * 2048;
                args.prefetch += prefetch_step;
            }
        }
    }

    // a series of weight blocks with possible different K but same N
    void matmulx(int M,
                 int N,
                 const int* first_K,
                 const int* last_K,
                 const uint8_t* pA,
                 size_t strideA,
                 const uint8_t* pB,
                 void* pC,
                 size_t strideC) const {
        int itembytes = (tmul_type == SIMDJit::TMUL_TYPE::BF16 || tmul_type == SIMDJit::TMUL_TYPE::FP16) ? 2 : 1;
        // DEBUG_LOG(itembytes, M, N, Ks.size());
        if (M <= 16) {
            // memory-bound case: special looping order
            (*tile_configer)(M);
            call_args args;
            args.strideA = strideA;
            args.strideC = strideC;
            args.pC = reinterpret_cast<uint8_t*>(pC);
            for (int n = 0; n < N; n += REG_BLK_N) {
                auto* pB0 = pB;
                args.pA = pA;
                for (auto* ik = first_K; ik != last_K; ik++) {
                    auto k_size = *ik;
                    args.pB = pB0 + n * k_size / (REG_BLK_N * REG_BLK_K) * 2048;
                    args.k_tiles = k_size / REG_BLK_K;
                    args.do_accumulation = 0;
                    if (ik == first_K)
                        args.do_accumulation |= 1;
                    if (ik == last_K - 1)
                        args.do_accumulation |= 2;
                    (*jit_amx_mm1x2)(&args);
                    pB0 += N * k_size / (REG_BLK_N * REG_BLK_K) * 2048;
                    args.pA += k_size * itembytes;
                }
                args.pC += REG_BLK_N * sizeof(uint32_t);
            }
        } else {
            // compute-bound case
            for (auto* ik = first_K; ik != last_K; ik++) {
                auto& k_size = *ik;
                auto pfetchB = pB;
                if (ik != last_K - 1) {
                    // prefetch next B block
                    pfetchB += N * k_size / (REG_BLK_N * REG_BLK_K) * 2048;
                }
                matmul(M, N, k_size, ik != first_K, pA, strideA, pB, pC, strideC, pfetchB);
                pA += k_size * itembytes;
                pB += (N * k_size) / (REG_BLK_N * REG_BLK_K) * 2048;
            }
        }
    }
};

class func_cvt_output {
    std::shared_ptr<SIMDJit> jit;
    struct call_args {
        const uint8_t* psrc;
        uint8_t* pdst;
        uint8_t* prefetch;
        int64_t count;  // number of 32bit elements
    };

public:
    using Ptr = std::shared_ptr<const func_cvt_output>;

    func_cvt_output(int i_dtype, int o_dtype) {
        // fp32 => fp16 or bf16
        ASSERT(i_dtype == DTYPE_FP32 || i_dtype == DTYPE_INT32);
        jit = SIMDJit::create([&](SIMDJit* jit, SReg pargs) {
            SReg psrc, pdst, prefetch, count, i;
            VReg d0, d1;
            psrc.load(pargs + offsetof(call_args, psrc));
            pdst.load(pargs + offsetof(call_args, pdst));
            prefetch.load(pargs + offsetof(call_args, prefetch));
            count.load(pargs + offsetof(call_args, count));
            i = 0;
            auto simdw = d0.lanes<uint32_t>();
            jit->do_while_(i < count, [&] {
                d0.load(psrc + i * sizeof(float));
                d1.load(psrc + i * sizeof(float) + simdw * sizeof(float));
                if (o_dtype == DTYPE_FP16) {
                    ASSERT(i_dtype == DTYPE_FP32);
                    d0.store(pdst + i * sizeof(uint16_t), VReg::LDST_TYPE::packed_fp16_fp32);
                    d1.store(pdst + i * sizeof(uint16_t) + simdw * sizeof(uint16_t), VReg::LDST_TYPE::packed_fp16_fp32);
                    // jit->vcvtps2ph(jit->ptr[pdst.r64() + i.r64() * sizeof(uint16_t)], d0, 0x4);
                    // jit->vcvtps2ph(jit->ptr[pdst.r64() + i.r64() * sizeof(uint16_t) + simdw * sizeof(uint16_t)],
                    //                d1,
                    //                0x4);
                    jit->prefetchwt1(jit->ptr[prefetch.r64() + i.r64() * sizeof(uint16_t)]);
                } else if (o_dtype == DTYPE_BF16) {
                    ASSERT(i_dtype == DTYPE_FP32);
                    VReg d2;
                    jit->vcvtne2ps2bf16(d2, d1, d0);
                    jit->prefetchwt1(jit->ptr[prefetch.r64() + i.r64() * sizeof(uint16_t)]);
                    d2.store(pdst + i * sizeof(uint16_t));
                } else if (o_dtype == i_dtype) {
                    // both in & out are 32bit
                    d0.store(pdst + i * sizeof(float));
                    d1.store(pdst + i * sizeof(float) + simdw * sizeof(float));
                    jit->prefetchwt1(jit->ptr[prefetch.r64() + i.r64() * sizeof(float)]);
                } else {
                    ASSERT(false);
                }
                i = i + simdw * 2;
            });
        });
    }
    void operator()(int M, int N, const void* psrc, size_t stride_src, void* pdst, size_t stride_dst) const {
        call_args args;
        args.psrc = reinterpret_cast<const uint8_t*>(psrc);
        args.pdst = reinterpret_cast<uint8_t*>(pdst);
        args.count = N;

        for (int m = 0; m < M; m++) {
            args.prefetch = (m + 2 < M) ? (args.pdst + 2 * stride_dst) : args.pdst;
            (*jit)(&args);
            args.psrc += stride_src;
            args.pdst += stride_dst;
        }
    }
};

class func_reduce2 {
    std::shared_ptr<SIMDJit> jit;

public:
    using Ptr = std::shared_ptr<const func_reduce2>;
    static constexpr int SUM_FP32 = 1;
    static constexpr int SUM_INT32 = 2;

    func_reduce2(int type) {
        jit = SIMDJit::create([&](SIMDJit* jit, SReg psrc0, SReg psrc1, SReg pdst, SReg count) {
            VReg d0, d1;
            SReg i;
            i = 0;
            auto simdw = d0.lanes<uint32_t>();
            jit->do_while_(i < count, [&] {
                d0.load(psrc0 + i * sizeof(uint32_t));
                d1.load(psrc1 + i * sizeof(uint32_t));
                if (type == SUM_INT32)
                    jit->vpaddd(d0, d0, d1);
                if (type == SUM_FP32)
                    jit->vaddps(d0, d0, d1);
                d0.store(pdst + i * sizeof(uint32_t));
                i = i + simdw;
            });
        });
    }

    void operator()(const void* pvsrc0,
                    size_t stride_src0,
                    const void* pvsrc1,
                    size_t stride_src1,
                    void* pvdst,
                    size_t stride_dst,
                    int M,
                    int N) const {
        auto* psrc0 = reinterpret_cast<const uint8_t*>(pvsrc0);
        auto* psrc1 = reinterpret_cast<const uint8_t*>(pvsrc1);
        auto* pdst = reinterpret_cast<uint8_t*>(pvdst);
        for (int m = 0; m < M; m++) {
            (*jit)(psrc0, psrc1, pdst, static_cast<int64_t>(N));
            psrc0 += stride_src0;
            psrc1 += stride_src1;
            pdst += stride_dst;
        }
    }
};

class func_quantize {
    std::shared_ptr<SIMDJit> jit;

public:
    using Ptr = std::shared_ptr<const func_quantize>;

    // quantize bf16/fp16 into int8 sym
    func_quantize(int x_dtype) {
        ASSERT(x_dtype == DTYPE_FP16 || x_dtype == DTYPE_BF16);
        jit = SIMDJit::create([&](SIMDJit* jit, SReg psrc, SReg pdst, SReg pscale, SReg count) {
            VReg d0, d1;
            VReg vmax, sign_bit_mask, vscale, vdscale;
            SReg i;

            auto simdw = d0.lanes<int32_t>();

            sign_bit_mask.load(0x80000000);
            vmax.load(0);

            jit->for_loop(i, 0, count, simdw, [&](KReg kmask){
                if (x_dtype == DTYPE_FP16)
                    d0.load(psrc + i * sizeof(int16_t), VReg::LDST_TYPE::packed_fp16_fp32, kmask);
                else if (x_dtype == DTYPE_BF16)
                    d0.load(psrc + i * sizeof(int16_t), VReg::LDST_TYPE::packed_bf16_fp32, kmask);

                jit->vandnps(d0, sign_bit_mask, d0);  // clear sign bit
                jit->vmaxps(vmax, vmax, d0);
            });

            vmax.reduce_ps([&](const VReg& vdst, const VReg& vsrc0, const VReg& vsrc1) {
                jit->vmaxps(vdst, vsrc0, vsrc1);
            });

            VReg v_q_max;
            v_q_max.load(127.0f);
            //jit->vbroadcastss(vscale, vmax.xmm());
            jit->vdivps(vdscale, vmax, v_q_max);
            jit->vdivps(vscale, v_q_max, vmax);

            vdscale.store(pscale, VReg::LDST_TYPE::scalar_fp32);

            jit->for_loop(i, 0, count, simdw, [&](KReg kmask){
                if (x_dtype == DTYPE_FP16)
                    d0.load(psrc + i * sizeof(int16_t), VReg::LDST_TYPE::packed_fp16_fp32, kmask);
                else if (x_dtype == DTYPE_BF16)
                    d0.load(psrc + i * sizeof(int16_t), VReg::LDST_TYPE::packed_bf16_fp32, kmask);
                jit->vmulps(d0, d0, vscale);
                jit->vcvtps2dq(d0, d0);
                d0.store(pdst + i * sizeof(int8_t), VReg::LDST_TYPE::packed_i8_i32, kmask);
            });

        });
    }

    // src : bf16/fp16, dst : int8, scale : float
    void operator()(const void* pvsrc,
                    size_t stride_src,
                    int8_t* pvdst,
                    size_t stride_dst,
                    float* scale_dst,
                    int M,
                    int N) const {
        auto* psrc = reinterpret_cast<const uint8_t*>(pvsrc);
        auto* pdst = reinterpret_cast<uint8_t*>(pvdst);
        for (int m = 0; m < M; m++) {
            (*jit)(psrc, pdst, scale_dst + m, static_cast<int64_t>(N));
            psrc += stride_src;
            pdst += stride_dst;
        }
    }
};


class func_dequantize {
    std::shared_ptr<SIMDJit> jit;

public:
    using Ptr = std::shared_ptr<const func_dequantize>;

    // dequantize int32 into bf16/fp16
    struct call_args {
        const int32_t* src; // int32
        int64_t stride_src;
        void* dst;          // bf16/fp16
        int64_t stride_dst;
        const float* scale_M;
        const float* scale_N;
        int64_t M;
        int64_t N;
    };

    func_dequantize(int x_dtype) {
        ASSERT(x_dtype == DTYPE_FP16 || x_dtype == DTYPE_BF16);
        jit = SIMDJit::create([&](SIMDJit* jit, SReg pargs) {
            SReg psrc, stride_src, pdst, stride_dst;
            SReg pscale_N, pscale_M, M, N, m, n;
            VReg d0, d1;

            psrc.load(pargs + offsetof(call_args, src));
            stride_src.load(pargs + offsetof(call_args, stride_src));
            pdst.load(pargs + offsetof(call_args, dst));
            stride_dst.load(pargs + offsetof(call_args, stride_dst));
            pscale_N.load(pargs + offsetof(call_args, scale_N));
            pscale_M.load(pargs + offsetof(call_args, scale_M));
            M.load(pargs + offsetof(call_args, M));
            N.load(pargs + offsetof(call_args, N));

            auto simdw = d0.lanes<int32_t>();
            jit->for_loop(m, 0, M, 1, [&] {
                VReg vscale_m, vscale_n;
                vscale_m.load(pscale_M + m*sizeof(float), VReg::LDST_TYPE::broadcast_fp32);

                jit->for_loop(n, 0, N, simdw, [&] {
                    vscale_n.load(pscale_N + n * sizeof(float), VReg::LDST_TYPE::packed_fp32);
                    d0.load(psrc + n * sizeof(int32_t), VReg::LDST_TYPE::packed_i32_fp32);
                    jit->vmulps(d0, d0, vscale_n);
                    jit->vmulps(d0, d0, vscale_m);
                    //jit->jcout(jit->jcout.as_f32, m, ",", n, ",", "d0=", d0, " vscale_n=", vscale_n, " vscale_m=", vscale_m);
                    if (x_dtype == DTYPE_BF16)
                        d0.store(pdst + n * sizeof(int16_t), VReg::LDST_TYPE::packed_bf16_fp32);
                    if (x_dtype == DTYPE_FP16)
                        d0.store(pdst + n * sizeof(int16_t), VReg::LDST_TYPE::packed_fp16_fp32);
                });
                psrc = psrc + stride_src;
                pdst = pdst + stride_dst;
            });
        });
    }

    // src : bf16/fp16, dst : int8, scale : float
    void operator()(const int32_t* psrc,
                    size_t stride_src,
                    void* pvdst,
                    size_t stride_dst,
                    const float* scale_M,
                    const float* scale_N,
                    int M,
                    int N) const {
        call_args args;
        args.src = psrc;
        args.stride_src = stride_src;
        args.dst = pvdst;
        args.stride_dst = stride_dst;
        args.scale_M = scale_M;
        args.scale_N = scale_N;
        args.M = M;
        args.N = N;
        (*jit)(&args);
    }
};


inline std::shared_ptr<int8_t> pack_weights(PlainTensor* ws, int w_count, int w_dtype_size, bool cvt_f16_to_bf16,
                                            blocking_1d& blk_bn,
                                            blocking_1d& blk_bk) {
    static auto amx_repack_Btile = make_cacheable<func_amx_repack_Btile>(cvt_f16_to_bf16);

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

struct AMXLinearKernel {
    std::shared_ptr<int8_t> w_buff;
    bool w_is_int8;  // weight is int8 or fp16/bf16

    int64_t N;
    int64_t K;

    blocking_1d blk_bk;
    blocking_1d blk_bn;

    // AMX B-matrix pair size 32x32(bf16/f16) or 64x32(int8)
    int REG_BLK_N = 32;
    int REG_BLK_K;
    int CACHE_BLK_K = 256;

    func_amx::Ptr amx_mm;
    func_amx_repack_Btile::Ptr amx_repack_Btile;
    func_reduce2::Ptr reduce2_fp32;
    func_reduce2::Ptr reduce2_int32;

    struct kgroup_t {
        std::shared_ptr<std::atomic<int>> step;
        void* pC;
        kgroup_t() {
            step = std::make_shared<std::atomic<int>>();
        }
    };
    std::vector<kgroup_t> ksyncs;
    int k_group_workers = 1;

    // weight may be bf16/fp16/int8
    template <class F>
    void init(size_t _K, size_t _N, int tmul_type, int w_dtype, const F& get_weight) {
        K = _K;
        N = _N;

        auto f16_to_bf16 = (w_dtype == DTYPE_FP16 && tmul_type == func_amx::TMUL_BF16);
        amx_mm = make_cacheable<func_amx>(256, tmul_type);
        amx_repack_Btile = make_cacheable<func_amx_repack_Btile>(f16_to_bf16);
        reduce2_fp32 = make_cacheable<func_reduce2>(func_reduce2::SUM_FP32);
        reduce2_int32 = make_cacheable<func_reduce2>(func_reduce2::SUM_INT32);
        w_is_int8 = (tmul_type == func_amx::TMUL_SSD);

        //std::cout << "tmul_type=" << tmul_type << ", w_dtype=" << w_dtype << ", f16_to_bf16=" << f16_to_bf16
        //          << ", DTYPE_FP16=" << DTYPE_FP16 << ", DTYPE_BF16=" << DTYPE_BF16 << std::endl;

        REG_BLK_N = 16 * 2;
        REG_BLK_K = 16 * (w_is_int8 ? 4 : 2);
        CACHE_BLK_K = 256;

        // DEBUG_LOG(N0, N1, N2, K);
        if (K <= CACHE_BLK_K * 4) {
            // do not blocking on K dimension
            blk_bk.reset_size(K, K);
        } else {
            blk_bk.reset_size(K, CACHE_BLK_K);
        }
        auto nthr = parallel_get_max_threads();

        ksyncs.resize(nthr);
        bool b_small_weights = (N / nthr) < 256;

        // determine cache blocking by shape of weight alone.
        k_group_workers = 1;
        if ((K >= N) && b_small_weights && (nthr > 3)) {
            // N dimension is not enough to produce enough work-threads, split along K dimension into groups also
            // if there are odd number of workers, the unpaired worker will have nothing to do, so this method is
            // only enabled when nthr is big enough.
            k_group_workers = 2;
            auto thr_groups = nthr / k_group_workers;
            auto nblock_size = N / thr_groups;
            if (nblock_size > 256) {
                // number of N-blocks can be larger than thread groups
                // BN should fit L2 cache
                blk_bn.reset_size(N, 256);
            } else {
                blk_bn.reset_cnt(N, REG_BLK_N, thr_groups);
            }
        } else if (b_small_weights) {
            // expect big M at runtime to produce enough of worker threads
            blk_bn.reset_size(N, 256);
        } else {
            // classic column linear blocking
            blk_bn.reset_cnt(N, REG_BLK_N, nthr);
        }

        static int show_cnt = 5;
        if (show_cnt) {
            DEBUG_LOG(blk_bk);
            DEBUG_LOG(k_group_workers, blk_bn);
            show_cnt--;
        }

        w_buff = alloc_cache_aligned<int8_t>(N * K * (w_is_int8 ? 1 : 2));
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
    }

    uint32_t* get_ctemp(size_t count) {
        thread_local scratch_buff<uint32_t> ctemp_per_thread;
        ctemp_per_thread.ensure_size(count);
        return ctemp_per_thread.buff.get();
    }

    // this is a general matmul between src & prepacked weight, post_ops are memory based
    void execute(int64_t M,
                 const void* p_vx,
                 int64_t stride_x,
                 const std::function<void(int64_t, int64_t, int64_t, int64_t, void*, int64_t)>& post_ops) {
        // w_buff
        auto* p_x = reinterpret_cast<const uint8_t*>(p_vx);
        auto* pw_base = reinterpret_cast<uint8_t*>(w_buff.get());
        constexpr int BM = 256;
        const auto& Ks = blk_bk.sizes();

        auto nblkN = blk_bn.size();
        auto nblkM = (M + BM - 1) / BM;
        parallel_nt_static(0, [&](int ithr, int nthr) {
            auto n_group = nthr / k_group_workers;
            auto i_group = ithr / k_group_workers;
            if (i_group >= n_group)
                return;

            // share the atomic of first thread in group
            auto group_ithr0 = i_group * k_group_workers;
            auto& sync_atomic = *ksyncs[group_ithr0].step;

            int64_t i0, i1;
            splitter(static_cast<int64_t>(nblkN * nblkM), n_group, i_group, i0, i1);
            for (int64_t i = i0; i < i1; i++) {
                int64_t m0, m1;
                // allocate task reusing same sub-weight to same thread
                m0 = (i % nblkM) * BM;
                m1 = std::min(m0 + BM, M);

                auto blk_n = i / nblkM;
                int n0 = blk_bn[blk_n].first;
                int n1 = blk_bn[blk_n].second;
                // DEBUG_LOG(m0, m1, n0, n1)

                auto strideC = (n1 - n0) * sizeof(float);
                if ((strideC % 128) == 0)
                    strideC += 64;
                auto* pC = get_ctemp((BM)*strideC / sizeof(float));

                // DEBUG_LOG(ithr, nthr, i, group_id, group_off, pC);
                int64_t ki0 = 0, ki1 = blk_bk.size();
                if (k_group_workers > 1) {
                    ksyncs[ithr].pC = pC;
                    splitter(static_cast<int64_t>(blk_bk.size()), k_group_workers, ithr % k_group_workers, ki0, ki1);
                }
                auto k_off = blk_bk[ki0].first;
                auto* pw = pw_base + (n0 * K + k_off * (n1 - n0)) / (REG_BLK_N * REG_BLK_K) * 2048;
                auto* px = p_x + m0 * stride_x + k_off * (w_is_int8 ? 1 : 2);

                amx_mm->matmulx(m1 - m0,
                                n1 - n0,
                                Ks.data() + ki0,
                                Ks.data() + ki1,
                                px,  // p_x + m0 * stride_x,
                                stride_x,
                                pw,  // pw_base + (n0 * K) / (REG_BLK_N * REG_BLK_K) * 2048,
                                pC,
                                strideC);

                if (k_group_workers > 1) {
                    if (sync_atomic.fetch_add(1) < k_group_workers - 1) {
                        // wait for group leader to finish
                        while (sync_atomic.load() > 0) {}
                        continue;
                    }

                    // all others have done their work, reduce & post-ops
                    for (int j = 0; j < k_group_workers; j++) {
                        auto other_ithr = group_ithr0 + j;
                        if (other_ithr != ithr) {
                            auto* pC2 = ksyncs[other_ithr].pC;
                            if (w_is_int8)
                                (*reduce2_int32)(pC, strideC, pC2, strideC, pC, strideC, m1 - m0, n1 - n0);
                            else
                                (*reduce2_fp32)(pC, strideC, pC2, strideC, pC, strideC, m1 - m0, n1 - n0);
                        }
                    }
                    sync_atomic.store(0); // inform group member to continue
                }

                post_ops(m0, m1, n0, n1, pC, strideC);
            }
        });
    }
};

}
}
