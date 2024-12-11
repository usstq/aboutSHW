#include <functional>
#include <vector>

#include "../include/jit.h"

enum backend_type : unsigned {
    emu = 0,
    avx2 = 1,
    avx512 = 2,
    sve = 3,
};
//==========================================================
template <backend_type BT>
struct var {};

template <>
struct var<emu> {
    int reg;
};

template <>
struct var<avx2> {
    std::shared_ptr<Xbyak::Reg64> reg;
};
//==========================================================
template <backend_type BT>
struct vmm {};

struct emu_vreg {
    union {
        float f32[8];
        int32_t i32[8];
        uint32_t u32[8];
        uint16_t u16[16];
        uint8_t u8[32];
    };
};

template <>
struct vmm<emu> {
    constexpr static size_t SIMDW = 8;
    constexpr static size_t NUMREGS = 16;
    std::shared_ptr<emu_vreg> reg;
};

template <>
struct vmm<avx2> {
    constexpr static size_t SIMDW = 8;
    constexpr static size_t NUMREGS = 16;
    std::shared_ptr<Xbyak::Ymm> reg;
};

template <>
struct vmm<avx512> {
    constexpr static size_t SIMDW = 16;
    constexpr static size_t NUMREGS = 32;
    std::shared_ptr<Xbyak::Zmm> reg;
};
//==========================================================

template <backend_type BT>
class cjit : public jit_generator {};

template <>
class cjit<avx2> : public jit_generator {
public:
    // each back-end specilization determines it's own meaning of var()
    // this class implements the semantics of cjit's core concept
    struct sregister {
        std::shared_ptr<Xbyak::Reg64> reg;
        sregister(std::shared_ptr<Xbyak::Reg64> reg = nullptr) : reg(reg) {}
        operator Xbyak::Reg64() const {
            return *reg;
        }
        Xbyak::Reg32 r32() const {
            return Xbyak::Reg32(reg->getIdx());
        }
    };
    struct vregister {
        std::shared_ptr<Xbyak::Ymm> ymm;
        vregister(std::shared_ptr<Xbyak::Ymm> ymm = nullptr) : ymm(ymm) {}
        operator Xbyak::Ymm() const {
            return *ymm;
        }
        Xbyak::Xmm xmm() const {
            return Xbyak::Xmm(ymm->getIdx());
        }
    };
    using pregister = vregister;

    constexpr static size_t NUM_SREGS = 16;
    constexpr static size_t SIMDW = 8;
    constexpr static size_t NUM_VREGS = 16;

    std::vector<int> m_free_srf_regs;
    std::vector<int> m_free_vrf_regs;
    std::vector<int> m_free_mask_regs;  // mask/predicate regs

    cjit() {
        for (size_t i = 0; i < NUM_SREGS; i++)
            m_free_srf_regs.push_back(i);
        for (size_t i = 0; i < NUM_VREGS; i++)
            m_free_vrf_regs.push_back(i);
    }

    sregister var(int idx = -1) {
        if (m_free_srf_regs.empty())
            throw std::runtime_error("No free registers");
        if (idx >= 0) {
            // user-specified idx
            auto it = std::find(m_free_srf_regs.begin(), m_free_srf_regs.end(), idx);
            if (it == m_free_srf_regs.end())
                throw std::runtime_error(std::string("No required register : ") + std::to_string(idx));
            m_free_srf_regs.erase(it);
        } else {
            idx = m_free_srf_regs.back();
            m_free_srf_regs.pop_back();
        }
        return std::shared_ptr<Xbyak::Reg64>(new Xbyak::Reg64(idx), [this](Xbyak::Reg64* preg) {
            this->m_free_srf_regs.push_back(preg->getIdx());
            delete preg;
        });
    }
    sregister arg(int i) {
        return var(abi_param_regs[i]);
    }

    vregister vec() {
        if (m_free_vrf_regs.empty())
            throw std::runtime_error("No free vector registers");
        auto idx = m_free_vrf_regs.back();
        m_free_vrf_regs.pop_back();
        return std::shared_ptr<Xbyak::Ymm>(new Xbyak::Ymm(idx), [this](Xbyak::Ymm* preg) {
            this->m_free_vrf_regs.push_back(preg->getIdx());
            delete preg;
        });
    }

    template <typename B, typename E, typename S = size_t>
    void j_foreach(sregister& idx, const B& start, const E& stop, const S& step, std::function<void()>&& fn) {
        Xbyak::Label loop, exit;

        mov(idx, start);

        L(loop);
        cmp(idx, stop);
        jge(exit, T_NEAR);

        fn();

        add(idx, step);
        jmp(loop, T_NEAR);
        L(exit);
    }

    // vector version for: (idx + step) will not pass the stop
    template <typename B, typename E, typename S = size_t>
    void j_loop_vec(sregister& idx, const B& start, const E& stop, const S& step, std::function<void()>&& fn) {
        Xbyak::Label loop, exit;

        mov(idx, start);

        L(loop);
        add(idx, step);
        cmp(idx, stop);
        jg(exit, T_NEAR);
        sub(idx, step);

        fn();
        add(idx, step);

        jmp(loop, T_NEAR);
        L(exit);

        // at exit, idx is still within range
        sub(idx, step);
    }

    //==========================================================================================================================
    struct address {
        const sregister base;
        const sregister idx;
        const pregister mask;
        const int scales;
        const int displacement;
        address(const sregister& base, const sregister& idx = {}, const int scales = 32, const int displacement = 0)
            : base(base),
              idx(idx),
              scales(scales),
              displacement(displacement) {}
        address(const pregister& mask, const sregister& base, const sregister& idx = {}, const int scales = 32, const int displacement = 0)
            : mask(mask),
              base(base),
              idx(idx),
              scales(scales),
              displacement(displacement) {}
        operator Xbyak::Address() const {
            if (idx.reg)
                return Xbyak::Address(0, false, *(base.reg) + (*(idx.reg)) * scales + displacement);
            else
                return Xbyak::Address(0, false, *(base.reg) + displacement);
        }
    };

    void j_load(const vregister& dst, const address& addr) {
        if (addr.mask.ymm)
            vmaskmovps(dst, addr.mask, addr);
        else
            vmovdqu(dst, addr);
    }
    void j_store(const vregister& dst, const address& addr) {
        if (addr.mask.ymm)
            vmaskmovps(addr, addr.mask, dst);
        else
            vmovdqu(addr, dst);
    }

    // generate a tail mask based on index & count
    // on AVX2 : https://www.felixcloutier.com/x86/vmaskmov is used to load/store
    // the mask is just a normal SIMD vector register
    pregister j_tail_mask(const int element_bits, const sregister& count) {
        // element_bits : 32(float)/16(half)/8(int8)
        if (element_bits != 32)
            throw std::runtime_error("on 32-bits unit size supported for mask on AVX2");
        static int32_t mem_mask[] = {-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0};
        auto mask = vec();
        // generate the mask
        auto tmp = var();
        auto idx = var();
        mov(idx, count);
        and_(idx, 7);
        neg(idx);
        add(idx, 8);
        //printf("mem_mask=%llx\n", reinterpret_cast<uintptr_t>(mem_mask));
        mov(tmp, reinterpret_cast<uintptr_t>(mem_mask));
        vmovdqu(mask, ptr[(*tmp.reg) + *(idx.reg)*4]);
        return mask;
    }
    // void j_load(const address& addr) {
    //     vmaskmovps(dst, mask, ptr[base]);
    // }
    void j_add(const vregister& dst, const vregister& a, const vregister& b) {
        vaddps(dst, a, b);
    }
    void j_load(const sregister& dst, const address& addr) {
        mov(dst, addr);
    }
    void j_store(const sregister& dst, const address& addr) {
        mov(addr, dst);
    }
    void j_add(const sregister& dst, const sregister& rhs) {
        add(dst, rhs);
    }
    void j_add(const sregister& dst, const int& rhs) {
        add(dst, rhs);
    }
    void j_set1(const vregister& dst, float v) {
        if (v == 0) {
            vpxor(dst, dst, dst);
        } else {
            auto tmp = var();
            mov(tmp, *reinterpret_cast<uint32_t*>(&v));
            vmovd(dst, tmp);
            vbroadcastss(dst, dst);
        }
    }
    void j_set1(const sregister& dst, int64_t v) {
        mov(dst, v);
    }
    void j_set1(const vregister& dst, int32_t v) {
        if (v == 0) {
            vpxor(dst, dst, dst);
        } else {
            auto tmp = var();
            mov(tmp.r32(), *reinterpret_cast<uint32_t*>(&v));
            //printf("============ %d %d \n", dst.xmm().isXMM(), tmp.r32().isREG(32));
            vmovd(dst.xmm(), tmp.r32());
            vbroadcastss(dst, dst.xmm());
        }
    }

    void finish() {
        ret();
        finalize();
    }
};

/*
    the code that generate code: G
    the generated code         : B

    G calls B: OK
    B calls G:
    only happens once at compilation time
*/

template <backend_type BT>
std::shared_ptr<cjit<BT>> func_body() {
    auto h = std::make_shared<cjit<BT>>();
    auto src1 = h->arg(0);
    auto src2 = h->arg(1);
    auto dst = h->arg(2);
    auto cnt = h->arg(3);

    auto idx = h->var();
    auto value1 = h->vec();
    auto value2 = h->vec();

    auto simdw = h->SIMDW;
    h->j_loop_vec(idx, 0, cnt, simdw, [&]() {
        h->j_load(value1, {src1, idx, 4});
        h->j_load(value2, {src2, idx, 4});
        h->j_add(value1, value1, value2);
        h->j_store(value1, {dst, idx, 4});
    });

    auto mask = h->j_tail_mask(32, cnt);
    h->j_load(value1, {mask, src1, idx, 4});
    h->j_load(value2, {mask, src2, idx, 4});
    h->j_add(value1, value1, value2);
    h->j_store(value1, {mask, dst, idx, 4});

    h->finish();
    return h;
}
