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
#if 0
template <>
class cjit<emu> : public jit_generator {
    union sregister{
        int64_t i64;
        int32_t i32;
        operator int64_t() const {
            return i64;
        }
    };
    union vregister {
        float f32[8];
        int32_t i32[8];
        bool mask[8];
    };
    using pregister = vregister;

    sregister var(int idx = -1) {
        return sregister();
    }
    sregister arg(int i) {
        return var(i);
    }
    vregister vec() {
        return vregister();
    }

    cjit() {}

    // vector version for: (idx + step) will not pass the stop
    template <typename B, typename E, typename S = size_t>
    void j_loop_vec(sregister& idx, const B& start, const E& stop, const S& step, std::function<void()>&& fn) {
        Xbyak::Label loop, exit;
        for (idx.i64 = static_cast<int64_t>(start); idx.i64 + step <= static_cast<int64_t>(stop); idx.i64 += step) {
            fn();
        }
    }

    struct address {
        const sregister base;
        const sregister idx;
        const pregister mask;
        const int scales;
        const int displacement;
        bool with_idx;
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
        uintptr_t addr() const {
            if (idx.reg)
                return Xbyak::Address(0, false, *(base.reg) + (*(idx.reg)) * scales + displacement);
            else
                return Xbyak::Address(0, false, *(base.reg) + displacement);
        }
    };



};
#endif

#if 0
template <typename... Ts>
void debug_log(const char * file_path, int line, Ts... args) {
    std::stringstream ss;
    int dummy[sizeof...(Ts)] = {(ss << args, 0)...};
    (void)(dummy);
    std::cout << file_path << ":" << line << "   " << ss.str() << "" << std::endl;
}
#define DEBUG_LOG(...) debug_log(__FILE__, __LINE__, ## __VA_ARGS__)
#else
#define DEBUG_LOG(...) 
#endif
template <>
class cjit<avx2> : public jit_generator {
public:
    // each back-end specilization determines it's own meaning of var()
    // this class implements the semantics of cjit's core concept
    struct sregister {
        std::shared_ptr<Xbyak::Reg64> reg;
        sregister(std::shared_ptr<Xbyak::Reg64> reg = nullptr) : reg(reg) {
            if (reg) DEBUG_LOG("\tallocated sreg #", reg->getIdx());
        }
        operator Xbyak::Reg64() const {
            return *reg;
        }
        Xbyak::Reg32 r32() const {
            return Xbyak::Reg32(reg->getIdx());
        }
    };
    struct vregister {
        std::shared_ptr<Xbyak::Ymm> ymm;
        vregister(std::shared_ptr<Xbyak::Ymm> ymm = nullptr) : ymm(ymm) {
            if (ymm) DEBUG_LOG("\tallocated vreg #", ymm->getIdx());
        }
        operator Xbyak::Ymm() const {
            return *ymm;
        }
        Xbyak::Xmm xmm() const {
            return Xbyak::Xmm(ymm->getIdx());
        }
    };
    using pregister = vregister;

    using SREG = sregister;
    using VREG = vregister;

    cjit() {
        init_frame();
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
        auto mask = cur_frame().local_vreg();
        // generate the mask
        auto tmp = cur_frame().local_sreg();
        auto idx = cur_frame().local_sreg();
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
    void j_add(const sregister& dst, const sregister& rhs, const int scales = 1) {
        if (scales == 1)
            add(dst, rhs);
        else
            lea(dst, ptr[(*dst.reg) + (*rhs.reg) * scales]);
    }
    void j_add(const sregister& dst, const int& rhs) {
        add(dst, rhs);
    }
    void j_set1(const vregister& dst, float v) {
        if (v == 0) {
            vpxor(dst, dst, dst);
        } else {
            auto tmp = cur_frame().local_sreg();
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
            auto tmp = cur_frame().local_sreg();
            mov(tmp.r32(), *reinterpret_cast<uint32_t*>(&v));
            //printf("============ %d %d \n", dst.xmm().isXMM(), tmp.r32().isREG(32));
            vmovd(dst.xmm(), tmp.r32());
            vbroadcastss(dst, dst.xmm());
        }
    }

    int inject_frame_level = 1;
    constexpr static size_t SIMDW = 8;
    constexpr static size_t NUM_SREGS = 16;
    constexpr static size_t NUM_VREGS = 16;

    // owner id:
    //  -2  arch reserved register, always keep untouched (like RSP)
    //  -1  function caller used register, must be preserved to/from stack before using
    //   0  free register, can be used w/o preserv
    //  >0  onwer inject frame index
    int m_srf_owner[NUM_SREGS];
    int m_vrf_owner[NUM_VREGS];

    void init_frame() {
        inject_frame_level = 1;
        for (size_t i = 0; i < NUM_SREGS; i++) {
            m_srf_owner[i] = 0; // free : no need to push/pop
            m_vrf_owner[i] = 0; // free : no need to push/pop
        }
        for (size_t i = 0; i < sizeof(abi_save_gpr_regs)/sizeof(abi_save_gpr_regs[0]); i++) {
            // owner id 1 means caller owns it, need to 
            m_srf_owner[abi_save_gpr_regs[i]] = 1;
        }
        // RSP is always kept untouched.
        m_srf_owner[Xbyak::Operand::RSP] = -2;
    }

    // stack frame for inline/inject only:
    //   manages visiblity of registers
    //   we need to extract total number of free-registers required for
    //   current frame, and preserve enough of registers by adding pre/postamble
    // but how can we know the total number of free-registers for current frame?
    // 1. since there is no IR, we must run the generation process once to only collect
    //    number of registers required. then we run again with this knowledge.
    // 2. we put a pre/postamble code area filled with nop instruction and later-on 
    //    at frame destruction time, we change those nop instruction to push/pops.
    // 3. most easy solution: we just preserve all non-free sreg/vreg onto stack
    //    since call_inline_frame is supposed to represent a relatively big function call.
    //    and normal injector shouldn't use it (they should just allocated from current frame).
    struct call_inline_frame {
        cjit<avx2>& h;
        struct reginfo {
            int id;
            int owner;
            bool pushed;
            reginfo(int id, int owner, bool pushed) : id(id), owner(owner), pushed(pushed) {}
        };
        std::vector<reginfo> arg_sregs;
        std::vector<reginfo> arg_vregs;

        call_inline_frame(cjit<avx2>& h, const std::vector<sregister>& visible_sregs, const std::vector<vregister>& visible_vregs)
            : h(h) {
            // when allocation/free happens on swappable register
            // it always happens in stack style (LIFO), and each reg's life-time is maintained by it's own shared_ptr
            //
            // In C++, local variables within a function or block are constructed in the order they are defined
            // and destructed in the reverse order of their construction. This behavior is deterministic and
            // follows the Last-In-First-Out (LIFO) principle.
            //
            // Given this fact, all regs's lifecycle are maintained by it's own shared_ptr with custom-deleter,
            // even when stack-swapping is required.
            h.inject_frame_level ++;
            DEBUG_LOG("Entering call_inline_frame at level ", h.inject_frame_level);
            // set injector's arg register status to `already allocated & owned by current call_inline_frame`
            for (auto& r : visible_sregs) {
                auto idx = r.reg->getIdx();
                arg_sregs.emplace_back(idx, h.m_srf_owner[idx], false);
                h.m_srf_owner[idx] = h.inject_frame_level;
            }
            for (auto& r : visible_vregs) {
                auto idx = r.ymm->getIdx();
                arg_vregs.emplace_back(idx, h.m_vrf_owner[idx], false);
                h.m_vrf_owner[idx] = h.inject_frame_level;
            }
            // we will save all sreg & vreg with owner to stack since call_inline_frame means a big function
            // this is allows temporary registers to be used w/o any push/pop
            // (if we have a way to know in advance how many sregs & vregs is going to be used, we can
            // protect them selectively rather than all of them, but as a thin wrapper of jit we don't have that)
            for (size_t i = 0; i < NUM_SREGS; i++) {
                auto owner = h.m_srf_owner[i];
                if (owner != 0 && owner != -2 && owner != h.inject_frame_level) {
                    arg_sregs.emplace_back(i, owner, true); // record orginal owner & set it to 0 (since it's free now)
                    h.m_srf_owner[i] = 0;
                    h.push(Xbyak::Reg64(i));
                    DEBUG_LOG("pushed sreg ", i);
                }
            }
            for (size_t i = 0; i < NUM_VREGS; i++) {
                auto owner = h.m_vrf_owner[i];
                if (owner != 0 && owner != -2 && owner != h.inject_frame_level) {
                    arg_vregs.emplace_back(i, owner, true);
                    h.sub(h.rsp, 32);
                    h.vmovups(Xbyak::Ymm(i), h.ptr[h.rsp]);
                    DEBUG_LOG("pushed vreg ", i);
                }
            }
        }
        ~call_inline_frame() {
            // restore orginal owner id for arguments
            for (auto it = arg_vregs.rbegin(); it != arg_vregs.rend(); ++it) {
                h.m_vrf_owner[it->id] = it->owner;
                if (it->pushed) {
                    h.vmovups(h.ptr[h.rsp], Xbyak::Ymm(it->id));
                    h.add(h.rsp, 32);
                    DEBUG_LOG("popped vreg ", it->id);
                }
            }
            for (auto it = arg_sregs.rbegin(); it != arg_sregs.rend(); ++it) {
                h.m_srf_owner[it->id] = it->owner;
                if (it->pushed) {
                    h.pop(Xbyak::Reg64(it->id));
                    DEBUG_LOG("popped sreg ", it->id);
                }
            }
            DEBUG_LOG("Exit call_inline_frame at level ", h.inject_frame_level);
            h.inject_frame_level --;

            if (h.inject_frame_level == 1) {
                // this is last frame, do final works
                DEBUG_LOG("Finalize");
                h.ret();
                h.finalize();
            }
        }
        sregister arg(int arg_idx) {
            return local_sreg(abi_param_regs[arg_idx]);
        }
        sregister local_sreg(int idx = -1) {
            auto* ph = &h;
            if (idx >= 0) {
                if (h.m_srf_owner[idx] != 0) {
                    throw std::runtime_error(std::string("Specified sregister ") + std::to_string(idx) + " is owned by " + std::to_string(h.m_srf_owner[idx]));
                }
                h.m_srf_owner[idx] = h.inject_frame_level;
                return std::shared_ptr<Xbyak::Reg64>(new Xbyak::Reg64(idx), [ph](Xbyak::Reg64* preg) {
                    DEBUG_LOG("\tfreeing arg sreg #", preg->getIdx());
                    ph->m_srf_owner[preg->getIdx()] = 0;
                    delete preg;
                });
            }
            // try to allocate free reg:
            for (size_t i = 0; i < NUM_SREGS; i++) {
                if (h.m_srf_owner[i] == 0) {
                    h.m_srf_owner[i] = h.inject_frame_level;
                    return std::shared_ptr<Xbyak::Reg64>(new Xbyak::Reg64(i), [ph](Xbyak::Reg64* preg) {
                        DEBUG_LOG("\tfreeing local free sreg #", preg->getIdx());
                        ph->m_srf_owner[preg->getIdx()] = 0;
                        delete preg;
                    });
                }
            }
            throw std::runtime_error("No free sregister available!");
        }

        vregister local_vreg() {
            auto* ph = &h;
            // try to allocate free reg:
            for (size_t i = 0; i < NUM_VREGS; i++) {
                if (h.m_vrf_owner[i] == 0) {
                    h.m_vrf_owner[i] = h.inject_frame_level;
                    return std::shared_ptr<Xbyak::Ymm>(new Xbyak::Ymm(i), [ph](Xbyak::Ymm* preg) {
                        DEBUG_LOG("\tfreeing local free vreg #", preg->getIdx());
                        ph->m_vrf_owner[preg->getIdx()] = 0;
                        delete preg;
                    });
                }
            }
            throw std::runtime_error("No free vregister available!");
        }
    };

    std::list<call_inline_frame> m_inject_frames;

    std::shared_ptr<call_inline_frame> new_frame(const std::vector<sregister>& sregs = {}, const std::vector<vregister>& vregs = {}) {
        // these are registers that are preserved in current frame
        // do not swap to memory
        m_inject_frames.emplace_back(*this, sregs, vregs);
        return std::shared_ptr<call_inline_frame>(&m_inject_frames.back(), [this](call_inline_frame* f){
            m_inject_frames.pop_back();
        });
    }
    // injector can also use parent frame w/o allocate it's own frame (as long as it's light-weighted)
    call_inline_frame& cur_frame() {
        return m_inject_frames.back();
    }
};

/*
universal injector : call-syntax w/o call instruction
*/

template <class CJIT, class VREG>
void kernel_compute(CJIT& h, VREG c, VREG a, VREG b) {
    h.j_add(c, a, b);
}

template <class CJIT>
void func_compute(CJIT& h, typename CJIT::SREG src1, typename CJIT::SREG src2, typename CJIT::SREG dst, typename CJIT::SREG cnt) {
    auto frame = h.new_frame({src1, src2, dst, cnt}, {});
    auto idx = frame->local_sreg();
    auto value1 = frame->local_vreg();
    auto value2 = frame->local_vreg();
    auto simdw = h.SIMDW;
    h.j_loop_vec(idx, 0, cnt, simdw, [&]() {
        h.j_load(value1, {src1, idx, 4});
        h.j_load(value2, {src2, idx, 4});
        kernel_compute(h, value1, value1, value2);
        h.j_store(value1, {dst, idx, 4});
    });

    auto mask = h.j_tail_mask(32, cnt);
    h.j_load(value1, {mask, src1, idx, 4});
    h.j_load(value2, {mask, src2, idx, 4});
    kernel_compute(h, value1, value1, value2);
    h.j_store(value1, {mask, dst, idx, 4});
}

template <backend_type BT>
std::shared_ptr<cjit<BT>> func_body() {
    auto ph = std::make_shared<cjit<BT>>();
    auto& h = *ph;
    auto frame = h.new_frame();
    auto src1 = frame->arg(0);
    auto src2 = frame->arg(1);
    auto dst = frame->arg(2);
    auto M = frame->arg(3);
    auto N = frame->arg(4);
    auto idx = frame->local_sreg();
    h.j_foreach(idx, 0, M, 1, [&](){
        func_compute(h, src1, src2, dst, N);
        h.j_add(src1, N, 4);
        h.j_add(src2, N, 4);
        h.j_add(dst, N, 4);
    });
    return ph;
}
