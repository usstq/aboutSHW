#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <mutex>

#include "misc.hpp"
#include "omp.h"
#include "simd_jit.hpp"
#include "simd_jit_tests.hpp"
#include "simple_perf.hpp"

using ov::intel_cpu::SIMDJit;
using ov::intel_cpu::SReg;
using ov::intel_cpu::TMUL_TYPE;
using ov::intel_cpu::TReg;
using ov::intel_cpu::VReg;

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
    printf("\e[33m"
           "initXTILE success!\n"
           "\e[0m");
    return true;
}

static bool init_once = initXTILE();

#pragma message "Recompiling ..."

////////////////////////////////////////////////////////////////////////////
// all jit-based/performance-aware function should be a functor/callable because:
//   - it needs to hold reference to kernel (to save build time & resources)
//   - it needs to do other compile time preparation work and hold the relevant
//     runtime-data-struct (to make runtime faster)
// to optimze compile-time-workload itself, the functor instance itself should be
// cached with compile-time parameter as the key.
//
// because it's a functor, which supposed to have no states, so cache-factory should
// always return shared_ptr to constant object, so it won't behave differently when being
// called by different caller, and this also ensure it's multi-threading safe since it
// won't modify it's content.
//
template <typename... TTypes>
class tuple_hasher {
private:
    typedef std::tuple<TTypes...> Tuple;
    template <int N>
    size_t hash(Tuple& value) const {
        return 0;
    }
    template <int N, typename THead, typename... TTail>
    size_t hash(Tuple& value) const {
        constexpr int Index = N - sizeof...(TTail) - 1;
        return std::hash<THead>()(std::get<Index>(value)) ^ hash<N, TTail...>(value);
    }

public:
    size_t operator()(Tuple value) const {
        auto hv = hash<sizeof...(TTypes), TTypes...>(value);
        return hv;
    }
};

// make_cacheable():
//  create any object with internal cache to optimize functor's compilation-time
//      cargs :compile-time
template <class T, typename... CArgs>
std::shared_ptr<const T> make_cacheable(CArgs... cargs) {
    std::shared_ptr<const T> sptr;
    auto key = std::make_tuple(cargs...);
    static std::unordered_map<decltype(key), std::weak_ptr<const T>, tuple_hasher<CArgs...>> cache;
    static std::mutex mutex;
    std::lock_guard<std::mutex> guard(mutex);
    auto it = cache.find(key);
    if (it != cache.end()) {
        auto& wptr = it->second;
        sptr = wptr.lock();
        if (!sptr) {
            sptr = std::make_shared<T>(cargs...);
            ECOUT("make_cacheable constructed: ", typeid(T).name(), "(", cargs..., ")");
            wptr = sptr;
        }
    } else {
        sptr = std::make_shared<T>(cargs...);
        ECOUT("make_cacheable constructed: ", typeid(T).name(), "(", cargs..., ")");
        cache.emplace(std::make_pair(key, std::weak_ptr<const T>(sptr)));
    }
    return sptr;
}
////////////////////////////////////////////////////////////////////////////
class func_amx_repack_Btile {
    std::shared_ptr<SIMDJit> jit;
    const int stride_dst;

public:
    // const is required for functor
    using Ptr = std::shared_ptr<const func_amx_repack_Btile>;

    func_amx_repack_Btile() : stride_dst(64) {
        // compile-time work
        jit = SIMDJit::create([&](SIMDJit* jit, SReg psrc, SReg stride_src, SReg pdst) {
            VReg t[8];
            VReg tt[8];

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

// AMX based matmul in unit of 32x32(fp16/bf16) 64x32(int8)
class func_amx {
    struct config_t {
        uint8_t palette_id = 1;
        uint8_t startRow = 0;
        uint8_t reserved[14] = {0};
        uint16_t cols[16];
        uint8_t rows[16];
        void set(int tmm_id, uint8_t r, uint16_t c) {
            rows[tmm_id] = r;
            cols[tmm_id] = c;
        }
    } __attribute__((__packed__));
    config_t tile_cfgs[32];

    std::shared_ptr<SIMDJit> jit_tile_configer;
    std::shared_ptr<SIMDJit> jit_amx_mm;

    func_amx_repack_Btile amx_repack_Btile;

    struct call_args {
        const uint8_t* pA;  // bfloat16/int8
        int64_t strideA;    // in bytes
        const uint8_t* pB;  // bfloat16/int8
        uint8_t* pC;        // float32/int32
        int64_t strideC;    // in bytes
        const uint8_t* prefetch;
        int64_t k_tiles;  // K / 32
        int64_t do_accumulation;
    };

    const int REG_BLK_N = 32;
    const int REG_BLK_K;
    int m_prefetch_Blines = 0;

    TMUL_TYPE tmul_type;

public:
    using Ptr = std::shared_ptr<const func_amx>;
    static constexpr int TMUL_FP16 = 1;  // FP16 => FP32
    static constexpr int TMUL_BF16 = 2;  // BF16 => FP32
    static constexpr int TMUL_SSD = 3;   // INT8 INT8 => FP32
    func_amx(int M_hint, int type) : REG_BLK_K(16 * ((type < TMUL_SSD) ? 2 : 4)) {
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
        jit_tile_configer = SIMDJit::create([&](SIMDJit* jit, SReg p) {
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

        if (M_hint) {
            // next block size: 32 * N * sizeof(ov::bfloat16),
            // call number: N / 32 * M / 32
            // each call needs fetch: 32 * N * sizeof(ov::bfloat16) / (N / 32 * M / 32) = 32 * 1024 *
            // sizeof(ov::bfloat16) / M
            m_prefetch_Blines = 32768 * sizeof(uint16_t) / 64 / M_hint;
        }

        jit_amx_mm = SIMDJit::create([&](SIMDJit* jit, SReg pargs) {
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
            // reg_tmp.load(pargs + offsetof(call_args, M));
            // jit->if_(reg_tmp <= 16, [&] {
            //     reg_A1_addr = reg_A_addr;
            // });

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
            jit->while_(reg_ktiles > 0, [&] {
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
    void repack_B(int N, int K, const void* pvsrc, size_t stride_src, void* pvdst) const {
        auto* pdst = reinterpret_cast<uint8_t*>(pvdst);
        ASSERT((N % REG_BLK_N) == 0);
        ASSERT((K % REG_BLK_K) == 0);
        for (int n = 0; n < N; n += REG_BLK_N) {
            auto* psrc0 = reinterpret_cast<const uint8_t*>(pvsrc) + n * stride_src;
            auto* psrc1 = psrc0 + 16 * stride_src;
            for (int k = 0; k < K; k += REG_BLK_K, psrc0 += 64, psrc1 += 64) {
                // DEBUG_LOG(k, n, pdst-pdst_base);
                amx_repack_Btile(psrc0, stride_src, pdst);
                amx_repack_Btile(psrc1, stride_src, pdst + 1024);
                pdst += 2048;
            }
        }
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
        ASSERT((N % REG_BLK_N) == 0);
        ASSERT((K % REG_BLK_K) == 0);

        int cur_tile_cfg_id = -1;

        auto config_tile_M = [&](int valid_m) {
            // valid_m : [1 ~ 32]
            auto cfg_id = valid_m % 32;
            if (cur_tile_cfg_id != cfg_id) {
                (*jit_tile_configer)(cfg_id < 0 ? nullptr : &tile_cfgs[cfg_id]);
                cur_tile_cfg_id = cfg_id;
            }
        };

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
        for (int64_t m = 0; m + 32 <= M; m += 32, pC += 32 * strideC, pA += 32 * strideA) {
            auto valid_m = std::min(M - m, (int64_t)32);
            config_tile_M(valid_m);
            args.pB = pB;
            args.pA = pA;
            for (int n = 0; n < N; n += REG_BLK_N) {
                args.pC = pC + n * sizeof(float);
                (*jit_amx_mm)(&args);
                args.pB += args.k_tiles * 2048;
                args.prefetch += prefetch_step;
            }
        }
    }
};

template <typename T, typename Q>
inline void splitter(const T& n, const Q& team, const Q& tid, T& n_start, T& n_end) {
    if (team <= 1 || n == 0) {
        n_start = 0;
        n_end = n;
    } else {
        T n1 = (n + (T)team - 1) / (T)team;
        T n2 = n1 - 1;
        T T1 = n - n2 * (T)team;
        n_end = (T)tid < T1 ? n1 : n2;
        n_start = (T)tid <= T1 ? tid * n1 : T1 * n1 + ((T)tid - T1) * n2;
    }

    n_end += n_start;
}

struct blocking_1d {
    int64_t total_size;
    int64_t block_size;
    struct blk_info {
        int64_t off0;
        int64_t off1;
    };
    std::vector<blk_info> blks;

    // non-uniform size blocking
    void reset_blks(const std::vector<int64_t>& blk_sizes) {
        int off = 0;
        blks.clear();
        for (int i = 0; i < blk_sizes.size(); i++) {
            auto size = blk_sizes[i];
            blks.emplace_back(blk_info{off, off + size});
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
        for (int64_t i0 = 0; i0 < total_size; i0 += block_size) {
            int64_t i1 = std::min(i0 + block_size, total_size);
            blks.emplace_back(blk_info{i0, i1});
        }
    }

    // specify count instead of block size, total_size will be evenly splitted
    void reset_cnt(int64_t _total_size, int64_t atom_size, int64_t _block_cnt) {
        total_size = _total_size;
        block_size = 0;
        blks.clear();
        blks.reserve(_block_cnt);
        for (int64_t i = 0; i < _block_cnt; i++) {
            int64_t n0;
            int64_t n1;
            auto work_groups = total_size / atom_size;
            splitter(work_groups, _block_cnt, i, n0, n1);
            n0 *= atom_size;
            n1 *= atom_size;
            blks.emplace_back(blk_info{n0, n1});
        }
    }

    size_t size() {
        return blks.size();
    }

    // overload support parallel
    template <class F>
    void for_each_blk(int nthr, const F& f) {
        if (nthr == 0)
            nthr = omp_get_max_threads();
#pragma omp parallel
        {
            int ithr = omp_get_thread_num();
            size_t start, end;
            if (ithr < nthr) {
                splitter(blks.size(), nthr, ithr, start, end);
                if (end > start) {
                    for (int i = start; i < end; i++) {
                        auto& bi = blks[i];
                        f(i, bi.off0, bi.off1);
                    }
                }
            }
        }
    }

    template <class F>
    void for_each_blk(const F& f) {
        for (int i = 0; i < blks.size(); i++) {
            auto& bi = blks[i];
            f(i, bi.off0, bi.off1);
        }
    }

    // return block id & updated offset
    int blk_offet(int& offset) {
        if (block_size > 0) {
            auto i = offset / block_size;
            offset -= i * block_size;
            return i;
        }
        int i;
        for (i = 0; i < blks.size(); i++) {
            auto& bi = blks[i];
            if (offset >= bi.off0 && offset < bi.off1) {
                offset -= bi.off0;
                return i;
            }
        }
        offset -= blks.back().off0;
        return i;
    }
};

std::ostream& operator<<(std::ostream& os, const blocking_1d& rhs) {
    os << "blocking_1d(" << rhs.total_size << "," << rhs.block_size << ", {";
    const char* sep = "";
    for (auto& bi : rhs.blks) {
        os << sep << bi.off0 << "+" << bi.off1 - bi.off0;
        sep = ",";
    }
    os << "})";
    return os;
}

// blocked QKV weight memory
struct AMXQKVLinearKernel {
    std::shared_ptr<int8_t> w_buff;
    bool w_is_int8;  // weight is int8 or fp16/bf16

    int64_t N[3];
    int64_t K;

    blocking_1d blk_bk;
    blocking_1d blk_bn;
    blocking_1d blk_rn;

    // AMX B-matrix pair size 32x32(bf16/f16) or 64x32(int8)
    int REG_BLK_N = 32;
    int REG_BLK_K;
    int CACHE_BLK_K = 256;

    func_amx_repack_Btile::Ptr amx_repack_B = make_cacheable<func_amx_repack_Btile>();
    func_amx::Ptr amx_mm;

    template <class F>
    void split_N3(int64_t n0, int64_t n1, const F& f) {
        auto stop = N[0];
        if (n0 < stop) {
            auto ne = std::min(n1, stop);
            f(0, 0, n0, ne);
            n0 = ne;
            if (n0 >= n1)
                return;
        }
        stop += N[1];
        if (n0 < stop) {
            auto ne = std::min(n1, stop);
            f(1, N[0], n0, ne);
            n0 = ne;
            if (n0 >= n1)
                return;
        }
        f(2, N[0] + N[1], n0, n1);
    }

    void init(const void* pv_w0,
              size_t _N0,
              size_t stride_w0,
              const void* pv_w1,
              size_t _N1,
              size_t stride_w1,
              const void* pv_w2,
              size_t _N2,
              size_t stride_w2,
              size_t _K,
              char w_format,  // 'b' int8 or 'e' fp16
              int ttype) {
        N[0] = _N0;
        N[1] = _N1;
        N[2] = _N2;
        K = _K;
        // jit_amx_mm = get_amx_32x32_kernel(tmul_type);
        // amx_mm = build_amx_mm_kernel(256, ttype);
        amx_mm = make_cacheable<func_amx>(256, ttype);

        w_is_int8 = (w_format == 'b');

        const int8_t* pw[3];
        size_t stride_w[3];
        pw[0] = reinterpret_cast<const int8_t*>(pv_w0);
        pw[1] = reinterpret_cast<const int8_t*>(pv_w1);
        pw[2] = reinterpret_cast<const int8_t*>(pv_w2);

        stride_w[0] = stride_w0;
        stride_w[1] = stride_w1;
        stride_w[2] = stride_w2;

        // DEBUG_LOG(N0, N1, N2, K);

        REG_BLK_N = 16 * 2;
        REG_BLK_K = 16 * (w_is_int8 ? 4 : 2);
        CACHE_BLK_K = 256;

        blk_bk.reset_size(K, CACHE_BLK_K);

        auto nthr = omp_get_max_threads();
        blk_bn.reset_cnt(N[0] + N[1] + N[2], REG_BLK_N, nthr);

        // DEBUG_LOG(blk_bk);
        // DEBUG_LOG(blk_bn);

        w_buff = alloc_cache_aligned<int8_t>((N[0] + N[1] + N[2]) * K * (w_is_int8 ? 1 : 2));

        // following loop determines blocking layout
        auto* pdst_base = reinterpret_cast<uint8_t*>(w_buff.get());

        blk_bn.for_each_blk(0, [&](int ibn, int n0, int n1) {
            blk_bk.for_each_blk([&](int ibk, int k0, int k1) {
#if 1
                auto* pdst0 = pdst_base + (n0 * K + (n1 - n0) * k0) / (REG_BLK_N * REG_BLK_K) * 2048;
                split_N3(n0, n1, [&](int idx, int base_n, int ns0, int ns1) {
                    auto stride = stride_w[idx];
                    auto* psrc = pw[idx] + (ns0 - base_n) * stride + (k0 / REG_BLK_K * 64);
                    auto* pdst1 = pdst0 + (ns0 - n0) * K / (REG_BLK_N * REG_BLK_K) * 2048;
                    amx_mm->repack_B(ns1 - ns0, k1 - k0, psrc, stride, pdst1);
                });
#else
                auto* pdst = pdst_base + (n0 * K + (n1 - n0) * k0) / (REG_BLK_N * REG_BLK_K) * 2048;
                for (int n = n0; n < n1; n += REG_BLK_N) {
                    auto& loc = rb2loc[n / REG_BLK_N];
                    auto stride_src = stride_w[loc.id];
                    auto* psrc0 = pw[loc.id] + loc.n_off * stride_src + (k0 / REG_BLK_K * 64);
                    auto* psrc1 = psrc0 + 16 * stride_src;
                    for (int k = k0; k < k1; k += REG_BLK_K, psrc0 += 64, psrc1 += 64) {
                        // DEBUG_LOG(k, n, pdst-pdst_base);
                        (*amx_repack_B)(psrc0, stride_src, pdst);
                        (*amx_repack_B)(psrc1, stride_src, pdst + 1024);
                        pdst += 2048;
                    }
                }
#endif
            });
        });
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

    void convert_to_dst(const void* pv_src, size_t stride_src, void* pv_dst, size_t stride_dst, int rows, int cols) {
        static auto cvt_kernel = SIMDJit::create([&](SIMDJit* jit, SReg psrc, SReg pdst, SReg prefetch_dst, SReg size) {
            // size : how many int32/float
            SReg i;
            VReg d0, d1;
            i = 0;
            auto simdw = jit->vmm_width<uint32_t>();
            jit->do_while_(i < size, [&] {
                d0.load(psrc + i * sizeof(float));
                d1.load(psrc + i * sizeof(float) + simdw * sizeof(float));
                jit->vcvtps2ph(jit->ptr[pdst.r64() + i.r64() * sizeof(uint16_t)], d0, 0x4);
                jit->vcvtps2ph(jit->ptr[pdst.r64() + i.r64() * sizeof(uint16_t) + simdw * sizeof(uint16_t)], d1, 0x4);
                jit->prefetchwt1(jit->ptr[prefetch_dst.r64() + i.r64() * sizeof(uint16_t)]);
                i = i + simdw * 2;
            });
        });

        auto* psrc = reinterpret_cast<const uint8_t*>(pv_src);
        auto* pdst = reinterpret_cast<uint8_t*>(pv_dst);
        for (int m = 0; m < rows; m++) {
            auto* pfetch = pdst + stride_dst;
            (*cvt_kernel)(psrc, stride_src, pdst, stride_dst, pfetch, cols);
            psrc += stride_src;
            pdst += stride_dst;
        }
    }

    // input bf16 / int8
    void execute(int64_t M, const uint8_t* p_x, int64_t stride_x, uint8_t* (&ptr_y)[3], int64_t (&stride_y)[3]) {
        // w_buff
        auto* pw_base = reinterpret_cast<uint8_t*>(w_buff.get());
        auto nthr = omp_get_max_threads();
        ASSERT(blk_bn.size() == nthr);

        constexpr int BM = 256;
        int m_prefetch_Blines = 0;
        if (BM) {
            // next block size: 32 * N * sizeof(ov::bfloat16),
            // call number: N / 32 * M / 32
            // each call needs fetch: 32 * N * sizeof(ov::bfloat16) / (N / 32 * M / 32) = 32 * 1024 *
            // sizeof(ov::bfloat16) / M
            m_prefetch_Blines = 32768 * sizeof(uint16_t) / 64 / BM;
        }
        for (int64_t m0 = 0; m0 < M; m0 += BM) {
            auto m1 = std::min(m0 + BM, M);
            blk_bn.for_each_blk(0, [&](int ibn, int n0, int n1) {
                thread_local scratch_buff<uint32_t> ctemp;
                ctemp.ensure_size(BM * (n1 - n0));

                // which output we belong to ?
                int idx = 0;
                int n_off = 0;
                if (n0 >= N[0] + N[1]) {
                    idx = 2;
                    n_off = n0 - (N[0] + N[1]);
                } else if (n0 >= N[0]) {
                    idx = 1;
                    n_off = n0 - (N[0]);
                } else {
                    idx = 0;
                    n_off = n0;
                }
                auto strideC = stride_y[idx];
                auto* pC = ptr_y[idx] + n_off * sizeof(int32_t) + m0 * strideC;
                blk_bk.for_each_blk([&](int ibk, int k0, int k1) {
                    auto pA = p_x + m0 * stride_x + k0 * (w_is_int8 ? 1 : 2);
                    auto pB = pw_base + (n0 * K + (n1 - n0) * k0) / (REG_BLK_N * REG_BLK_K) * 2048;

                    amx_mm->matmul(m1 - m0, n1 - n0, k1 - k0, k0 > 0, pA, stride_x, pB, pC, strideC, pB);
                });
#if 0
                tcfg.set(0);
                args.strideA = stride_x;
                args.prefetch = pw_base;

                int64_t m1 = std::min(m0 + BM, M);
                blk_bk.for_each_blk([&](int ibk, int k0, int k1) {
                    auto* pw0 = pw_base + (n0 * K + (n1 - n0) * k0) / (REG_BLK_N * REG_BLK_K) * 2048;
                    args.k_tiles = (k1 - k0) / REG_BLK_K;
                    auto prefetch_step = m_prefetch_Blines * 64 * args.k_tiles;
                    args.prefetch = pw0;
                    if (k1 < K) {
                        args.prefetch = pw0 + (n1 - n0) * (k1 - k0) / (REG_BLK_N * REG_BLK_K) * 2048;
                    }
                    args.do_accumulation = k0 > 0;
                    // accumulate [m0~m1, k0~k1] x [k0~k1, n0~n1] => [m0~m1, n0~n1]
                    for (int64_t m = m0; m + 32 <= m1; m += 32) {
                        auto valid_m = std::min(m1 - m, (int64_t)32);
                        tcfg.set(valid_m);
                        args.pB = pw0;
                        args.pA = p_x + m * args.strideA + k0 * (w_is_int8 ? 1 : 2);

                        for (int n = n0; n < n1; n += REG_BLK_N) {
                            // 32x32
                            // auto& loc = rb2loc[n / REG_BLK_N];
                            // args.strideC = stride_y[loc.id];
                            // args.pC = reinterpret_cast<uint32_t*>(ptr_y[loc.id] + loc.n_off * sizeof(int32_t) + m *
                            // args.strideC);
                            args.strideC = (n1 - n0) * sizeof(float);
                            args.pC = ctemp.buff.get() + (m * (n1 - n0) + n - n0);
                            (*jit_amx_mm)(&args);
                            args.pB += args.k_tiles * 2048;
                            args.prefetch += prefetch_step;
                        }
                    }
                });
                // now convert Ctemp to destination buffer
                for (auto& task : obtasks[ibn].tasks) {
                    convert_to_dst(ctemp.buff.get() + task.src_off - n0,
                                   (n1 - n0) * sizeof(float),
                                   ptr_y[task.dst_id] + task.dst_off * sizeof(uint16_t),
                                   stride_y[task.dst_id],
                                   m1 - m0,
                                   task.n);
                }
#endif
            });
        }
    }
};

py::array_t<float> test_amx_repack_B(const py::array_t<float>& src) {
    auto func = make_cacheable<func_amx_repack_Btile>();
    py::array_t<float> dst({16, 16});
    (*func)(src.data(), src.strides(0), dst.mutable_data());
    return dst;
}

struct AMXQKVLinear {
    AMXQKVLinearKernel kernel;

    AMXQKVLinear(const py::array& w0, const py::array& w1, const py::array& w2) {
        ASSERT(w0.dtype().char_() == 'e' || w0.dtype().char_() == 'b');  // fp16 or int8
        ASSERT(w0.ndim() == 2);
        ASSERT(w1.ndim() == 2);
        ASSERT(w2.ndim() == 2);
        ASSERT(w1.shape(1) == w0.shape(1));
        ASSERT(w2.shape(1) == w0.shape(1));

        auto w_format = w0.dtype().char_();
        kernel.init(w0.data(),
                    w0.shape(0),
                    w0.strides(0),
                    w1.data(),
                    w1.shape(0),
                    w1.strides(0),
                    w2.data(),
                    w2.shape(0),
                    w2.strides(0),
                    w0.shape(1),
                    w_format,
                    w_format == 'e' ? func_amx::TMUL_BF16 : func_amx::TMUL_SSD);
    }
    void forward(const py::array& x, py::array& y0, py::array& y1, py::array& y2) {
        // w_buff
        auto M = x.shape(0);
        /*
        DEBUG_LOG(x.shape(0), x.shape(1), x.strides(0), x.strides(1));
        py::array_t<int32_t> y0({M, kernel.N0});
        py::array_t<int32_t> y1({M, kernel.N1});
        py::array_t<int32_t> y2({M, kernel.N2});
        */
        uint8_t* ptr_y[3] = {
            reinterpret_cast<uint8_t*>(y0.mutable_data()),
            reinterpret_cast<uint8_t*>(y1.mutable_data()),
            reinterpret_cast<uint8_t*>(y2.mutable_data()),
        };
        int64_t stride_y[3] = {y0.strides(0), y1.strides(0), y2.strides(0)};

        kernel.execute(M, reinterpret_cast<const uint8_t*>(x.data()), x.strides(0), ptr_y, stride_y);
        return;
    }
};

PYBIND11_MODULE(PACKAGE_NAME, m) {
    m.doc() = R"pbdoc(
    )pbdoc";

    py::class_<AMXQKVLinear>(m, "AMXQKVLinear")
        .def(py::init<const py::array&, const py::array&, const py::array&>())
        .def("forward", &AMXQKVLinear::forward);

    m.def("test_amx_repack_B", test_amx_repack_B);

    m.def("simd_test_basic", simd_test_basic);
    m.def("simd_test_tput", simd_test_tput);
    m.def("simd_test_printreg", simd_test_printreg);
}
