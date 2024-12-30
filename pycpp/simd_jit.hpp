#pragma once
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <vector>

#include "simd_jit_utils.hpp"

//=========================================================================================
#ifdef __x86_64__
#    include "../thirdparty/xbyak/xbyak/xbyak.h"
#    define DECLARE_CPU_JIT_AUX_FUNCTIONS(x)
static const bool use_avx512 = false;

using XbyakSReg64 = Xbyak::Reg64;
using XbyakSReg32 = Xbyak::Reg32;
using XbyakVReg = Xbyak::Xmm;
using XbyakLabel = Xbyak::Label;

constexpr Xbyak::Operand::Code abi_save_gpr_regs[] = {
    Xbyak::Operand::RBP,
    Xbyak::Operand::RBX,
    Xbyak::Operand::R12,
    Xbyak::Operand::R13,
    Xbyak::Operand::R14,
    Xbyak::Operand::R15,
#    ifdef _WIN32
    Xbyak::Operand::RDI,
    Xbyak::Operand::RSI,
#    endif
};

struct jit_generator : public Xbyak::CodeGenerator {
    const char* m_kernel_name;
    jit_generator(const char* name) : m_kernel_name(name), sreg_pool("sreg_pool"), vreg_pool("vreg_pool") {
        vreg_pool.add_range(0, 15);
        if (use_avx512)
            vreg_pool.add_range(16, 31);
#    ifdef _WIN32
        sreg_pool.add_range(std::vector<int>({Xbyak::Operand::RCX,
                                              Xbyak::Operand::RDX,
                                              Xbyak::Operand::R8,
                                              Xbyak::Operand::R9,

                                              // regs for local variables
                                              Xbyak::Operand::RDI,
                                              Xbyak::Operand::RSI,
                                              Xbyak::Operand::RBX,
                                              Xbyak::Operand::RBP,
                                              Xbyak::Operand::R10,
                                              Xbyak::Operand::R11,
                                              Xbyak::Operand::R12,
                                              Xbyak::Operand::R13,
                                              Xbyak::Operand::R14,
                                              Xbyak::Operand::R15}));
#    else
        sreg_pool.add_range(std::vector<int>({// args passed in register
                                              Xbyak::Operand::RDI,
                                              Xbyak::Operand::RSI,
                                              Xbyak::Operand::RDX,
                                              Xbyak::Operand::RCX,
                                              Xbyak::Operand::R8,
                                              Xbyak::Operand::R9,

                                              // regs for local variables
                                              Xbyak::Operand::RBX,
                                              Xbyak::Operand::RBP,
                                              Xbyak::Operand::R10,
                                              Xbyak::Operand::R11,
                                              Xbyak::Operand::R12,
                                              Xbyak::Operand::R13,
                                              Xbyak::Operand::R14,
                                              Xbyak::Operand::R15}));
#    endif
    }

    static std::string& jit_debug() {
        static std::string _jdbg = []() {
            auto* p_jit_debug = std::getenv("JIT_DEBUG");
            if (p_jit_debug == nullptr)
                return std::string{};
            return std::string(p_jit_debug);
        }();
        return _jdbg;
    }

    static const size_t num_abi_save_gpr_regs = sizeof(abi_save_gpr_regs) / sizeof(abi_save_gpr_regs[0]);
    void preamble() {
        for (size_t i = 0; i < num_abi_save_gpr_regs; ++i) {
            push(XbyakSReg64(abi_save_gpr_regs[i]));
            // Stack magic: save rsp into rbp state to be able to unwind stack.
            if (i == 0)
                mov(rbp, rsp);
        }
    }
    void uni_vzeroupper() {
        // if (mayiuse(avx))
        vzeroupper();
    }
    void postamble() {
        for (size_t i = 0; i < num_abi_save_gpr_regs; ++i)
            pop(XbyakSReg64(abi_save_gpr_regs[num_abi_save_gpr_regs - 1 - i]));
        uni_vzeroupper();
        ret();
    }
    const void* jit_kernel_code = nullptr;
    const void* jit_ker() const {
        return jit_kernel_code;
    }

    void return_(int imm32 = 0) {
        mov(rax, imm32);
        postamble();
    }

    void return_(XbyakSReg64 return_value = XbyakSReg64(Xbyak::Operand::RAX)) {
        if (return_value.getIdx() != rax.getIdx())
            mov(rax, return_value);
        postamble();
    }

    int finalize() {
        int err_code = Xbyak::GetError();
        if (err_code != Xbyak::ERR_NONE)
            return err_code;
        if (!jit_debug().empty()) {
            std::cout << "jit_generator generate() is done: " << m_kernel_name << std::endl;
            if (jit_debug() == m_kernel_name || jit_debug() == "*") {
                dump();
            }
        }
        jit_kernel_code = getCode();
        return (jit_kernel_code) ? 0 : -1;
    }

protected:
    ov::intel_cpu::reg_pool sreg_pool;
    ov::intel_cpu::reg_pool vreg_pool;

    std::shared_ptr<XbyakSReg64> alloc_reg64(int index) {
        auto reg_index = sreg_pool.allocate(index);
        return std::shared_ptr<XbyakSReg64>(new XbyakSReg64(reg_index), [this, reg_index](XbyakSReg64* preg) {
            if (preg) {
                sreg_pool.free(reg_index);
                delete preg;
            }
        });
    }

    std::shared_ptr<XbyakVReg> alloc_vreg(int index) {
        auto reg_index = vreg_pool.allocate(index);
        if (use_avx512) {
            return std::shared_ptr<XbyakVReg>(new Xbyak::Zmm(reg_index), [this, reg_index](Xbyak::Zmm* preg) {
                vreg_pool.free(reg_index);
                delete preg;
            });
        } else {
            return std::shared_ptr<XbyakVReg>(new Xbyak::Ymm(reg_index), [this, reg_index](Xbyak::Ymm* preg) {
                vreg_pool.free(reg_index);
                delete preg;
            });
        }
    }
};
#endif
//=========================================================================================
#ifdef __aarch64__
#    include "../thirdparty/xbyak_aarch64/xbyak_aarch64/xbyak_aarch64.h"
#    define DECLARE_CPU_JIT_AUX_FUNCTIONS(x)

using XbyakSReg64 = Xbyak_aarch64::XReg;
using XbyakSReg32 = Xbyak_aarch64::WReg;
using XbyakVReg = Xbyak_aarch64::VReg;
using XbyakLabel = Xbyak_aarch64::Label;

struct jit_generator : public Xbyak_aarch64::CodeGenerator {
    const char* m_kernel_name;
    jit_generator(const char* name, bool preserve_extra_xregs = false, bool preserve_extra_vregs = false)
        : m_kernel_name(name),
          m_preserve_extra_xregs(preserve_extra_xregs),
          m_preserve_extra_vregs(preserve_extra_vregs),
          sreg_pool("sreg_pool"),
          vreg_pool("vreg_pool") {
        sreg_pool.add_range(0, 15);
        if (preserve_extra_xregs)
            sreg_pool.add_range(19, 28);
        vreg_pool.add_range(0, 7);
        vreg_pool.add_range(16, 31);
        if (preserve_extra_vregs)
            vreg_pool.add_range(8, 15);
    }

    void preamble() {
        // X19 ~ X28 : Callee-saved.
        //  but do we need these extra general purpose registers?
        if (m_preserve_extra_xregs) {
            stp(x19, x20, pre_ptr(sp, -16));
            stp(x21, x22, pre_ptr(sp, -16));
            stp(x23, x24, pre_ptr(sp, -16));
            stp(x25, x26, pre_ptr(sp, -16));
            stp(x27, x28, pre_ptr(sp, -16));
        }
        if (m_preserve_extra_vregs) {
            stp(d8, d9, pre_ptr(sp, -16));
            stp(d10, d11, pre_ptr(sp, -16));
            stp(d12, d13, pre_ptr(sp, -16));
            stp(d14, d15, pre_ptr(sp, -16));
        }
    }
    void postamble() {
        if (m_preserve_extra_vregs) {
            stp(d14, d15, post_ptr(sp, 16));
            stp(d12, d13, post_ptr(sp, 16));
            stp(d10, d11, post_ptr(sp, 16));
            stp(d8, d9, post_ptr(sp, 16));
        }
        if (m_preserve_extra_xregs) {
            ldp(x27, x28, post_ptr(sp, 16));
            ldp(x25, x26, post_ptr(sp, 16));
            ldp(x23, x24, post_ptr(sp, 16));
            ldp(x21, x22, post_ptr(sp, 16));
            ldp(x19, x20, post_ptr(sp, 16));
        }
    }

    void return_(int64_t imm) {
        mov(x0, reinterpret_cast<uint64_t&>(imm));
        postamble();
        ret();
    }

    void return_(Xbyak_aarch64::XReg xret = Xbyak_aarch64::XReg(0)) {
        if (xret.getIdx() != 0) {
            mov(x0, xret);
        }
        postamble();
        ret();
    }

    const void* jit_kernel_code = nullptr;
    const void* jit_ker() const {
        return jit_kernel_code;
    }

    int finalize() {
        /*
        if (!jit_debug().empty()) {
            std::cout << "jit_generator generate() is done: " << m_kernel_name << std::endl;
            if (jit_debug() == m_kernel_name || jit_debug() == "*") {
                dump();
            }
        }*/
        ready();
        jit_kernel_code = getCode();
        return (jit_kernel_code) ? 0 : -1;
    }

protected:
    // https://developer.arm.com/documentation/102374/0102/Procedure-Call-Standard
    // https://learn.microsoft.com/en-us/cpp/build/arm64-windows-abi-conventions?view=msvc-170#integer-registers
    // https://github.com/ARM-software/abi-aa/blob/main/aapcs64/aapcs64.rst#61machine-registers
    // https://github.com/ARM-software/abi-aa/blob/main/aapcs64/aapcs64.rst#612simd-and-floating-point-registers
    //
    // Volatile xregs: x0-x8 x9-x15 (x19-x28 if extra xregs preserved)
    // Volatile vregs: v0-v7 v16-v31 (v8-v15 if extra vregs preserved)
    const bool m_preserve_extra_xregs;  // x19-x28	Non-volatile	Scratch registers
    const bool m_preserve_extra_vregs;  // v8-v15	Both	Low 64 bits are Non-Volatile. High 64 bits are Volatile.

    ov::intel_cpu::reg_pool sreg_pool;
    ov::intel_cpu::reg_pool vreg_pool;

    std::shared_ptr<XbyakSReg64> alloc_reg64(int index) {
        auto reg_index = sreg_pool.allocate(index);
        return std::shared_ptr<XbyakSReg64>(new XbyakSReg64(reg_index), [this, reg_index](XbyakSReg64* preg) {
            if (preg) {
                sreg_pool.free(reg_index);
                delete preg;
            }
        });
    }

    std::shared_ptr<XbyakVReg> alloc_vreg(int index) {
        auto reg_index = vreg_pool.allocate(index);
        return std::shared_ptr<XbyakVReg>(new XbyakVReg(reg_index), [this, reg_index](XbyakVReg* preg) {
            vreg_pool.free(reg_index);
            delete preg;
        });
    }
};

#endif

//=========================================================================================
namespace ov {
namespace intel_cpu {

#ifdef _WIN32
#    define abi_param_regs_num 4
#else
#    define abi_param_regs_num 6
#endif

class SIMDJit;
class SRegExpr;

class SReg {
private:
    SIMDJit* jit = nullptr;
    std::shared_ptr<XbyakSReg64> reg;

public:
    SReg(SIMDJit* jit, std::shared_ptr<XbyakSReg64> reg) : jit(jit), reg(reg) {}
    SReg() = default;
    bool empty() const {
        return !static_cast<bool>(reg);
    }
    operator XbyakSReg64&() {
        return *reg;
    }
    operator const XbyakSReg64&() const {
        return *reg;
    }
    XbyakSReg64& r64() {
        return *reg;
    }
    const XbyakSReg64& r64() const {
        return *reg;
    }

    inline const SReg& operator=(const SReg& reg) const;
    inline const SReg& operator=(SRegExpr&& expr) const;
    inline const SReg& operator+=(SRegExpr&& expr) const;
    inline const SReg& operator-=(SRegExpr&& expr) const;
    inline const SReg& operator*=(SRegExpr&& expr) const;
    inline void operator++() const;
    inline void operator--() const;
    inline void operator++(int) const;
    inline void operator--(int) const;
    friend class SIMDJit;
    friend class SRegExpr;
};

class VReg {
private:
    SIMDJit* jit = nullptr;
    std::shared_ptr<XbyakVReg> reg;

public:
    VReg(SIMDJit* jit, std::shared_ptr<XbyakVReg> reg) : jit(jit), reg(reg) {}
    VReg() = default;
    bool empty() const {
        return !static_cast<bool>(reg);
    }
    operator XbyakVReg&() {
        return *reg;
    }
    operator const XbyakVReg&() const {
        return *reg;
    }
};

class SRegExpr {
public:
    std::unique_ptr<RegExprImpl> pimpl;
    // Addressing is a special expression in following pattern
    //  - base [+ disp]
    //  - index * scale [+ disp]
    //  - base + index * scale + [+ disp]
    // which can be fast evaluated using LEA or PTR
    // this pattern is grew from construction time w/o requiring parsing of the AST
    // `paddr` only exists when current expression AST is a valid addressing pattern
    struct Addressing {
        int base_reg = -1;
        int index_reg = -1;
        int scale = 0;
        int64_t disp = 0;
        Addressing(int base_reg, int index_reg, int scale, int64_t disp)
            : base_reg(base_reg),
              index_reg(index_reg),
              scale(scale),
              disp(disp) {}
    };
    std::unique_ptr<Addressing> paddr;

    SRegExpr(int data) : pimpl(new RegExprImpl("i", data)) {}

    SRegExpr(SReg r) : pimpl(new RegExprImpl("r", r.r64().getIdx())) {}

    SRegExpr(const char* type, int data) : pimpl(new RegExprImpl(type, data)) {}
    SRegExpr(const char* op, SRegExpr&& lhs) : pimpl(new RegExprImpl(op, lhs.pimpl)) {}
    SRegExpr(const char* op, SRegExpr&& lhs, SRegExpr&& rhs) : pimpl(new RegExprImpl(op, lhs.pimpl, rhs.pimpl)) {
        // regularize operand order to allow best reuse temp register
        if (pimpl->is_op("+") || pimpl->is_op("*")) {
            if (!pimpl->rhs->is_leaf())
                std::swap(pimpl->lhs, pimpl->rhs);
            else if (pimpl->lhs->is_imm())
                std::swap(pimpl->lhs, pimpl->rhs);
        }

        // create Addressing from the first leaf-op when expr pattern is valid:
        if (pimpl->lhs->is_reg() && pimpl->rhs->is_leaf()) {
            if (pimpl->is_op("+")) {
                if (pimpl->rhs->is_reg())
                    // (base + index)
                    paddr.reset(new Addressing(pimpl->lhs->data, pimpl->rhs->data, 1, 0));
                else
                    // (base + disp)
                    paddr.reset(new Addressing(pimpl->lhs->data, -1, 0, pimpl->rhs->data));
            }
            if (pimpl->is_op("*") && pimpl->rhs->is_imm()) {
                // (index * scale)
                auto scale = pimpl->rhs->as_imm32();
                if (scale == 1 || scale == 2 || scale == 4 || scale == 8)
                    paddr.reset(new Addressing(-1, pimpl->lhs->data, pimpl->rhs->data, 0));
            }
        } else if (pimpl->is_op("+") && pimpl->rhs->is_leaf()) {
            // merge addressing mode: only (+base) or (+disp) is allowed
            if (rhs.paddr)
                paddr = std::move(rhs.paddr);
            if (lhs.paddr)
                paddr = std::move(lhs.paddr);
            if (paddr) {
                // update pattern
                if (pimpl->rhs->is_imm()) {
                    paddr->disp += pimpl->rhs->data;
                } else if (pimpl->rhs->is_reg()) {
                    if (paddr->base_reg < 0) {
                        paddr->base_reg = pimpl->rhs->data;
                    } else if (paddr->index_reg < 0) {
                        paddr->index_reg = pimpl->rhs->data;
                        paddr->scale = 1;
                    } else {
                        // invalid pattern
                        paddr.reset();
                    }
                } else {
                    paddr.reset();
                }
            }
        }
    }

    void show(std::string title) const {
        std::cout << "\033[32m::::" << title << "::::\033[0m" << std::endl;
        if (paddr) {
            std::cout << "Addressing:";
            if (paddr->base_reg >= 0)
                std::cout << " {r" << paddr->base_reg << "}";
            else
                std::cout << " {}";

            if (paddr->index_reg >= 0) {
                std::cout << " + {r" << paddr->index_reg << "} x " << paddr->scale;
            }
            std::cout << " + " << paddr->disp << std::endl;
        }
        pimpl->for_each_op([&](RegExprImpl* p) {
            std::cout << p->name() << " = " << p->lhs->name() << " " << p->op << " "
                      << (p->rhs ? p->rhs->name() : std::string("( )")) << std::endl;
            return true;
        });
    }
};

inline SRegExpr operator+(SRegExpr&& lhs, SRegExpr&& rhs) {
    return SRegExpr("+", std::move(lhs), std::move(rhs));
}
inline SRegExpr operator*(SRegExpr&& lhs, SRegExpr&& rhs) {
    return SRegExpr("*", std::move(lhs), std::move(rhs));
}
inline SRegExpr operator-(SRegExpr&& lhs, SRegExpr&& rhs) {
    return SRegExpr("-", std::move(lhs), std::move(rhs));
}
inline SRegExpr operator-(SRegExpr&& rhs) {
    SRegExpr lhs(0);
    return SRegExpr("-", std::move(lhs), std::move(rhs));
}
/*
inline SRegExpr operator/(SRegExpr&& lhs, SRegExpr&& rhs) {
    return SRegExpr("/", std::move(lhs), std::move(rhs));
}
inline SRegExpr operator%(SRegExpr&& lhs, SRegExpr&& rhs) {
    return SRegExpr("%", std::move(lhs), std::move(rhs));
}
*/
inline SRegExpr operator>>(SRegExpr&& lhs, SRegExpr&& rhs) {
    return SRegExpr(">>", std::move(lhs), std::move(rhs));
}
inline SRegExpr operator<<(SRegExpr&& lhs, SRegExpr&& rhs) {
    return SRegExpr("<<", std::move(lhs), std::move(rhs));
}
inline SRegExpr operator&(SRegExpr&& lhs, SRegExpr&& rhs) {
    return SRegExpr("&", std::move(lhs), std::move(rhs));
}
inline SRegExpr operator|(SRegExpr&& lhs, SRegExpr&& rhs) {
    return SRegExpr("|", std::move(lhs), std::move(rhs));
}
inline SRegExpr operator^(SRegExpr&& lhs, SRegExpr&& rhs) {
    return SRegExpr("^", std::move(lhs), std::move(rhs));
}
inline SRegExpr operator>(SRegExpr&& lhs, SRegExpr&& rhs) {
    return SRegExpr(">", std::move(lhs), std::move(rhs));
}
inline SRegExpr operator>=(SRegExpr&& lhs, SRegExpr&& rhs) {
    return SRegExpr(">=", std::move(lhs), std::move(rhs));
}
inline SRegExpr operator<(SRegExpr&& lhs, SRegExpr&& rhs) {
    return SRegExpr("<", std::move(lhs), std::move(rhs));
}
inline SRegExpr operator<=(SRegExpr&& lhs, SRegExpr&& rhs) {
    return SRegExpr("<=", std::move(lhs), std::move(rhs));
}
inline SRegExpr operator==(SRegExpr&& lhs, SRegExpr&& rhs) {
    return SRegExpr("==", std::move(lhs), std::move(rhs));
}
inline SRegExpr operator!=(SRegExpr&& lhs, SRegExpr&& rhs) {
    return SRegExpr("!=", std::move(lhs), std::move(rhs));
}
inline SRegExpr operator&&(SRegExpr&& lhs, SRegExpr&& rhs) {
    OPENVINO_ASSERT(lhs.pimpl->is_logical_op());
    OPENVINO_ASSERT(rhs.pimpl->is_logical_op());
    return SRegExpr("&&", std::move(lhs), std::move(rhs));
}
inline SRegExpr operator||(SRegExpr&& lhs, SRegExpr&& rhs) {
    OPENVINO_ASSERT(lhs.pimpl->is_logical_op());
    OPENVINO_ASSERT(rhs.pimpl->is_logical_op());
    return SRegExpr("||", std::move(lhs), std::move(rhs));
}
inline SRegExpr operator!(SRegExpr&& lhs) {
    OPENVINO_ASSERT(lhs.pimpl->is_logical_op());
    return SRegExpr("!", std::move(lhs));
}

//=========================================================================================
class SIMDJit : public jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(SIMDJit);

    class JitDisassembler {
    public:
        size_t start;
        SIMDJit* jit;
        JitDisassembler(SIMDJit* jit) : jit(jit) {
            start = jit->getSize();
        }
        ~JitDisassembler() {
            auto cur_loc = jit->getSize();
            std::ofstream outfile;
            outfile.open("temp.bin", std::ios_base::binary);
            outfile.write(reinterpret_cast<const char*>(jit->getJitCode()) + start, cur_loc - start);
            outfile.close();
#ifdef __x86_64__
            auto ret = std::system("objdump -D -b binary -mi386:x86-64 -M intel temp.bin");
#endif
#ifdef __aarch64__
            auto ret = std::system("objdump -D -b binary -maarch64 -M intel temp.bin");
#endif
            (void)ret;
        }
    };
    friend class JitDisassembler;

    const void* getJitCode() {
        return CodeGenerator::getCode();
    }
    std::unique_ptr<JitDisassembler> get_disasm(int enable) {
        if (enable) {
            auto* dis = new JitDisassembler(this);
            return std::unique_ptr<JitDisassembler>(dis);
        }
        return nullptr;
    }

    SIMDJit(const char* name = "") : jit_generator(name) {
#ifdef __x86_64__
        mov(rax, rsp);
#endif
        preamble();
    }

    // add an int64_t return value
    template <typename... kernel_args_t>
    int64_t operator()(kernel_args_t... args) const {
        using jit_kernel_func_t = int64_t (*)(const kernel_args_t... args);
        auto* fptr = (jit_kernel_func_t)jit_ker();
        return (*fptr)(std::forward<kernel_args_t>(args)...);
    }

    SReg get_arg(int idx) {
        auto ret = SReg(this, alloc_reg64(idx));
#ifdef __x86_64__
        // https://en.wikipedia.org/wiki/X86_calling_conventions#x86-64_calling_conventions
        if (idx >= abi_param_regs_num)
            mov(ret, ptr[rax + (idx - abi_param_regs_num + 1) * 8]);  // load from stack
#endif
#ifdef __aarch64__
        // https://en.wikipedia.org/wiki/Calling_convention#ARM_(A64)
        OPENVINO_ASSERT(idx < 8);
#endif
        return ret;
    }

    // find a free register, note argument registers are also allocatable, make sure
    // allocate argument registers before any local register-var
    SReg get_sreg() {
        return SReg(this, alloc_reg64(-1));
    }

    VReg get_vreg() {
        return VReg(this, alloc_vreg(-1));
    }

    std::vector<VReg> get_vregs(size_t num_vregs) {
        std::vector<VReg> ret(num_vregs);
        for (auto& v : ret)
            v = get_vreg();
        return ret;
    }

#ifdef __x86_64__
    // simd_xxx helpers have meaning similar to x86 intrinsics
    // it's more well-known than raw instruction can it also can be
    // made cross-platform(avx2/avx512/neon/...)

    void simd_set1_epi32(XbyakVReg vmm, int32_t imm32) {
        // this set1 is not performance critical
        mov(dword[rsp - 4], imm32);
        vpbroadcastd(vmm, dword[rsp - 4]);
    }
    void simd_and(XbyakVReg c, XbyakVReg a, XbyakVReg b) {
        if (use_avx512) {
            vpandd(c, a, b);
        } else {
            vpand(c, a, b);
        }
    }
    void simd_srli_epi32(XbyakVReg vdst, XbyakVReg vsrc, int32_t imm8) {
        vpsrld(vdst, vsrc, imm8);
    }
    void simd_srai_epi32(XbyakVReg vdst, XbyakVReg vsrc, int32_t imm8) {
        vpsrad(vdst, vsrc, imm8);
    }
    void simd_slli_epi32(XbyakVReg vdst, XbyakVReg vsrc, int32_t imm8) {
        vpslld(vdst, vsrc, imm8);
    }
    void simd_setzero_ps(XbyakVReg vmm) {
        if (use_avx512) {
            vpxord(vmm, vmm, vmm);
        } else {
            vpxor(vmm, vmm, vmm);
        }
    }
    void simd_loadu_ps(XbyakVReg vmm, const Xbyak::Address& addr) {
        vmovups(vmm, addr);
    }
    // load packed half into packed single
    void simd_loadu_phps(XbyakVReg vmm, const Xbyak::Address& addr) {
        vcvtph2ps(vmm, addr);
    }
    void simd_load_epu8_epi32(XbyakVReg vmm, const Xbyak::Address& addr) {
        vpmovzxbd(vmm, addr);
    }
    void simd_load_epi8_epi32(XbyakVReg vmm, const Xbyak::Address& addr) {
        vpmovsxbd(vmm, addr);
    }
    void simd_storeu_ps(const Xbyak::Address& addr, XbyakVReg vmm) {
        vmovups(addr, vmm);
    }
    void simd_fmadd_ps(XbyakVReg c, XbyakVReg a, const Xbyak::Operand& b) {
        vfmadd231ps(c, a, b);
    }
    void simd_sub_ps(XbyakVReg c, XbyakVReg a, XbyakVReg b) {
        vsubps(c, a, b);
    }
    void simd_add_ps(XbyakVReg c, XbyakVReg a, XbyakVReg b) {
        vaddps(c, a, b);
    }
    void simd_mul_ps(XbyakVReg c, XbyakVReg a, XbyakVReg b) {
        vmulps(c, a, b);
    }
    void simd_broadcast_ss(XbyakVReg vmm, const Xbyak::Address& addr) {
        vbroadcastss(vmm, addr);
    }
    void simd_cvtepi32_ps(XbyakVReg vmm_dst, XbyakVReg vmm_src) {
        vcvtdq2ps(vmm_dst, vmm_src);
    }
#endif

    //***********************************************
    // for_loop(idx, start, stop, step, loop_body) performs following:
    //    for(int idx=start; idx + step <= stop; idx+=step) {
    //       loop_body();
    //    }
    template <typename Fn, typename START, typename STEP>
    void for_loop(XbyakSReg64 idx, START start, XbyakSReg64 stop, STEP step, const Fn& loop_body);

    //***********************************************
    // while_(rax > 0, loop_body) performs following:
    //    while(rax > 0) {
    //       loop_body();
    //    }
    template <typename Fn>
    void while_(SRegExpr regcmp, const Fn& loop_body);

    template <typename Fn>
    void do_while_(SRegExpr regcmp, const Fn& loop_body);

    inline void if_(SRegExpr regcmp,
                    const std::function<void()>& then_body,
                    const std::function<void()>& else_body = {});

    // being specialization in platform-dependent header
    inline void evaluate(SRegExpr& expr,
                         const SReg* pdst = nullptr,
                         const char assign_op = '=',
                         const XbyakLabel& label = {});

    template <typename DT>
    static int vmm_width() {
#ifdef __x86_64__
        return (use_avx512 ? 512 : 256) / (sizeof(DT) * 8);
#endif
#ifdef __aarch64__
        return (128) / (sizeof(DT) * 8);
#endif
    }
};

//========================================================================================================
#ifdef __x86_64__
template <typename Fn, typename START, typename STEP>
void SIMDJit::for_loop(XbyakSReg64 idx, START start, XbyakSReg64 stop, STEP step, const Fn& loop_body) {
    Xbyak::Label loop, exit;
    mov(idx, start);

    align(64, false);
    L(loop);
    add(idx, step);
    cmp(idx, stop);
    jg(exit, T_NEAR);
    sub(idx, step);

    loop_body();
    add(idx, step);

    jmp(loop, T_NEAR);
    L(exit);
    // at exit, idx is pointing to tail
    sub(idx, step);
}

template <typename Fn>
void SIMDJit::while_(SRegExpr regcmp, const Fn& loop_body) {
    Xbyak::Label loop, exit;

    align(64, false);
    L(loop);

    evaluate(regcmp, nullptr, 'F', exit);

    loop_body();

    jmp(loop, T_NEAR);
    L(exit);
}

template <typename Fn>
void SIMDJit::do_while_(SRegExpr regcmp, const Fn& loop_body) {
    Xbyak::Label loop;

    align(64, false);
    L(loop);

    loop_body();

    evaluate(regcmp, nullptr, 'T', loop);
}

inline void SIMDJit::if_(SRegExpr regcmp,
                         const std::function<void()>& then_body,
                         const std::function<void()>& else_body) {
    Xbyak::Label if_else, if_exit;

    evaluate(regcmp, nullptr, 'F', if_else);

    then_body();

    if (else_body)
        jmp(if_exit, T_NEAR);

    L(if_else);

    if (else_body)
        else_body();

    L(if_exit);
}
inline const SReg& SReg::operator=(const SReg& rhs) const {
    jit->mov(*reg, rhs);
    return *this;
}
inline const SReg& SReg::operator=(SRegExpr&& expr) const {
    jit->evaluate(expr, this, '=');
    return *this;
}
inline const SReg& SReg::operator+=(SRegExpr&& expr) const {
    jit->evaluate(expr, this, '+');
    return *this;
}
inline const SReg& SReg::operator-=(SRegExpr&& expr) const {
    jit->evaluate(expr, this, '-');
    return *this;
}
inline const SReg& SReg::operator*=(SRegExpr&& expr) const {
    jit->evaluate(expr, this, '*');
    return *this;
}
inline void SReg::operator++() const {
    jit->inc(*reg);
}
inline void SReg::operator--() const {
    jit->dec(*reg);
}
inline void SReg::operator++(int) const {
    jit->inc(*reg);
}
inline void SReg::operator--(int) const {
    jit->dec(*reg);
}

inline void SIMDJit::evaluate(SRegExpr& expr, const SReg* pdst, const char assign_op, const Xbyak::Label& label) {
    int debug_log = SIMDJIT_DEBUG & 1;
    if (debug_log) {
        std::cout << "\033[32m==========================================\033[0m" << std::endl;
    }
    auto jit_dis = get_disasm(debug_log);

    auto paddr = expr.paddr.get();
    auto pimpl = expr.pimpl.get();
    // do_jump: the expression as condition of control-flow, will not be assigned to any register
    //          instead it will emmit `jump` instruction:
    // assign_op == 'T' jump to label if expression is true
    // assign_op == 'F' jump to label if expression is false
    const bool do_jump = (assign_op == 'T') || (assign_op == 'F');
    const bool do_assign = (pdst != nullptr) && (!do_jump);

    if (debug_log) {
        pimpl->show_rpn();
        if (pdst)
            std::cout << assign_op << " assign-to : r" << pdst->r64().getIdx() << std::endl;
    }

    // short expression optimization
    if (pdst) {
        auto& dst = *pdst;
        auto* lhs = pimpl;
        if (lhs->is_reg()) {
            switch (assign_op) {
            case '=':
                mov(dst, lhs->as_r64<XbyakSReg64>());
                break;
            case '+':
                add(dst, lhs->as_r64<XbyakSReg64>());
                break;
            case '-':
                sub(dst, lhs->as_r64<XbyakSReg64>());
                break;
            case '*':
                imul(dst, lhs->as_r64<XbyakSReg64>());
                break;
            default:
                OPENVINO_ASSERT(false);
                break;
            }
            return;
        }
        if (lhs->is_imm()) {
            switch (assign_op) {
            case '=':
                mov(dst, lhs->as_imm32());
                break;
            case '+':
                add(dst, lhs->as_imm32());
                break;
            case '-':
                sub(dst, lhs->as_imm32());
                break;
            case '*':
                imul(dst, dst, lhs->as_imm32());
                break;
            default:
                OPENVINO_ASSERT(false);
                break;
            }
            return;
        }
        // addressing expression
        if (paddr) {
            auto to_RegExp = [&] {
                OPENVINO_ASSERT(paddr);

                if (paddr->base_reg < 0) {
                    OPENVINO_ASSERT(paddr->index_reg >= 0);
                    return XbyakSReg64(paddr->index_reg) * paddr->scale + paddr->disp;
                } else if (paddr->index_reg >= 0)
                    return XbyakSReg64(paddr->base_reg) + XbyakSReg64(paddr->index_reg) * paddr->scale + paddr->disp;
                else
                    return XbyakSReg64(paddr->base_reg) + paddr->disp;
            };
            if (assign_op == '=') {
                lea(dst, ptr[to_RegExp()]);
                return;
            } else {
                auto temp = get_sreg();
                lea(temp, ptr[to_RegExp()]);
                switch (assign_op) {
                case '+':
                    add(dst, temp);
                    break;
                case '-':
                    sub(dst, temp);
                    break;
                case '*':
                    imul(dst, temp);
                    break;
                default:
                    OPENVINO_ASSERT(false);
                    break;
                }
            }
            return;
        }
    }

    // const-folding neighbor op
    pimpl->for_each_op([&](RegExprImpl* p) {
        if (p->is_op("-") && p->rhs->is_imm()) {
            p->op = "+";
            p->rhs->data = -(p->rhs->data);
        }
        return true;
    });

    pimpl->for_each_op([&](RegExprImpl* p) {
        if (p->rhs && !p->rhs->is_imm())
            return true;

        if (p->is_op("+")) {
            if (p->lhs->is_op("+") && p->lhs->rhs->is_imm()) {
                p->rhs->data += p->lhs->rhs->as_imm32();
                p->lhs = std::move(p->lhs->lhs);
            }
        }
        if (p->is_op("*")) {
            if (p->lhs->is_op("*") && p->lhs->rhs->is_imm()) {
                p->rhs->data *= p->lhs->rhs->as_imm32();
                p->lhs = std::move(p->lhs->lhs);
            }
        }
        return true;
    });
    if (debug_log)
        expr.show(" After const folding");

    // complex expression: need multiple passes on IR to work
    // assign scratch register & convert to 2-OP instruction form
    static reg_pool scratch_reg_sn_pool("sreg_expr_scratch_registers", 32);
    scratch_reg_sn_pool.clear();

    auto scratch_reg_base = 1000;
    pimpl->for_each_op([&](RegExprImpl* p) {
        if (!p->lhs->is_leaf()) {
            // `reuse lhs as dst` is the best case:
            //   dst = lhs + rhs  ===> lhs += rhs
            p->data = p->lhs->data;
            if (p->rhs && !p->rhs->is_leaf())
                scratch_reg_sn_pool.free(p->rhs->data - scratch_reg_base);
            return true;
        }

        if (!p->rhs->is_leaf() && p->is_op("-")) {
            // reuse rhs scratch by replacing 'lhs - rhs' with 'neg(lhs)+rhs'
            p->op = "n+";
            std::swap(p->lhs, p->rhs);
            p->data = p->lhs->data;
            return true;
        }

        // as last comparasion OP of a jump condition, no need to allocate scratch
        if (do_jump && p == pimpl && p->is_cmp() && p->lhs->is_reg()) {
            p->data = p->lhs->data;
            return true;
        }
        // otherwise, a comparasion OP needs to assign the comparasion result as boolean
        // to target scratch register, the expected instruction sequence would be:
        //      cmp lhs, rhs
        //      setcc dst
        // so dst register will be required (and it can be rhs)
        if (p->is_cmp() && !p->rhs->is_leaf()) {
            // reuse rhs register, also need to reverse compare
            // beause `cmp` instruction requires lhs to be register
            if (p->is_op(">"))
                p->op = "<";
            else if (p->is_op(">="))
                p->op = "<=";
            else if (p->is_op("<"))
                p->op = ">";
            else if (p->is_op("<="))
                p->op = ">=";
            std::swap(p->lhs, p->rhs);
            p->data = p->lhs->data;
            return true;
        }

        // there are still cases where rhs cannot be used as dst of 2-op instruction
        // for example: dst = r0 >> (rhs), in such case, we need to :
        //   - allocate new scratch for dst
        //   - insert a `dst = lhs` before current op
        auto new_scratch_reg_sn = scratch_reg_sn_pool.allocate() + scratch_reg_base;
        if (!p->rhs->is_leaf())
            scratch_reg_sn_pool.free(p->rhs->data - scratch_reg_base);

        // some instruction support 3-OP w/o need to insert mov
        if (p->is_op("*") && p->lhs->is_reg() && p->rhs->is_imm()) {
            p->data = new_scratch_reg_sn;
            return true;
        }

        // insert 'dst = lhs' in lhs data-path (when there is no 3-OP instruction)
        // space op " " means simply move lhs to dst `dst = lhs`
        std::unique_ptr<RegExprImpl> pmov(new RegExprImpl(" ", p->lhs));
        pmov->data = new_scratch_reg_sn;
        p->lhs = std::move(pmov);
        p->data = new_scratch_reg_sn;
        return true;
    });

    if (debug_log)
        expr.show(" After scratch reg allocation & convert to 2-OP form");

    // substitute scratch register with real physical register:
    bool dst_register_assigned_inplace = false;
    if (pdst && assign_op == '=') {
        // try to replace last scratch register with assign destination register
        auto assign_dst_reg_idx = pdst->r64().getIdx();
        auto assign_dst_reg_scratch_sn = pimpl->data;
        OPENVINO_ASSERT(assign_dst_reg_scratch_sn >= scratch_reg_base);
        // find the appearance of last access
        int last_access_exec_id = -1;
        int op_exec_id = 0;
        pimpl->for_each_op([&](RegExprImpl* p) {
            op_exec_id++;
            if (p->lhs->is_reg() && p->lhs->data == assign_dst_reg_idx) {
                last_access_exec_id = op_exec_id;
            }
            if (p->rhs && p->rhs->is_reg() && p->rhs->data == assign_dst_reg_idx) {
                last_access_exec_id = op_exec_id;
            }
            return true;
        });
        // replace assign dst scratch with real assign dest reg
        op_exec_id = 0;
        bool replaced = false;
        pimpl->for_each_op([&](RegExprImpl* p) {
            op_exec_id++;
            if (op_exec_id >= last_access_exec_id && p->data == assign_dst_reg_scratch_sn) {
                // the scratch reg has longer life-cycle, cannot replace
                if (p->lhs->data == assign_dst_reg_scratch_sn)
                    return false;
                p->data = assign_dst_reg_idx;
                replaced = true;
            }
            return true;
        });
        if (replaced) {
            dst_register_assigned_inplace = true;
            // remove useless mov
            pimpl->for_each_op([&](RegExprImpl* p) {
                if (p->lhs->is_op(" ") && p->lhs->lhs->is_reg() && p->lhs->lhs->data == p->lhs->data) {
                    p->lhs = std::move(p->lhs->lhs);
                }
                return true;
            });
        }
    }

    if (debug_log)
        expr.show(" After replace dst scratch register");

    // allocate physical registers
    std::map<int, SReg> scratch_regs;
    pimpl->for_each_op([&](RegExprImpl* p) {
        if (p->data >= scratch_reg_base) {
            auto it = scratch_regs.find(p->data);
            if (it != scratch_regs.end()) {
                p->data = it->second.r64().getIdx();
            } else {
                // allocate new scratch reg
                auto sreg = get_sreg();
                scratch_regs.emplace(p->data, sreg);
                p->data = sreg.r64().getIdx();
            }
        }
        return true;
    });

    if (debug_log)
        expr.show(" After allocation of all scratch registers");

    // emmit code
    pimpl->for_each_op([&](RegExprImpl* p) {
        auto dst = XbyakSReg64(p->data);
        if (p->is_op(" ")) {
            if (p->lhs->is_imm())
                mov(dst, p->lhs->as_imm32());
            else
                mov(dst, p->lhs->as_r64<XbyakSReg64>());
        } else if (p->is_op("+")) {
            if (p->rhs->is_imm())
                add(dst, p->rhs->as_imm32());
            else
                add(dst, p->rhs->as_r64<XbyakSReg64>());
        } else if (p->is_op("n+")) {
            neg(dst);
            if (p->rhs->is_imm()) {
                add(dst, p->rhs->as_imm32());
            } else
                add(dst, p->rhs->as_r64<XbyakSReg64>());
        } else if (p->is_op("*")) {
            if (p->rhs->is_imm()) {
                // support 3-OP
                imul(dst, p->lhs->as_r64<XbyakSReg64>(), p->rhs->as_imm32());
            } else {
                imul(dst, p->rhs->as_r64<XbyakSReg64>());
            }
        } else if (p->is_op("-")) {
            if (p->rhs->is_imm())
                sub(dst, p->rhs->as_imm32());
            else
                sub(dst, p->rhs->as_r64<XbyakSReg64>());
        } else if (p->is_op(">>")) {
            if (p->rhs->is_imm())
                sar(dst, p->rhs->as_imm32());
            else {
                // only cl register supportted, we need allocate cl
                OPENVINO_ASSERT(false);  // sar(dst, p->rhs->as_r64());
            }
        } else if (p->is_op("<<")) {
            if (p->rhs->is_imm())
                shl(dst, p->rhs->as_imm32());
            else {
                // only cl register supportted, we need allocate cl
                OPENVINO_ASSERT(false);  // shl(dst, p->rhs->as_r64());
            }
        } else if (p->is_op("&")) {
            if (p->rhs->is_imm())
                and_(dst, p->rhs->as_imm32());
            else {
                and_(dst, p->rhs->as_r64<XbyakSReg64>());
            }
        } else if (p->is_op("|")) {
            if (p->rhs->is_imm())
                or_(dst, p->rhs->as_imm32());
            else {
                or_(dst, p->rhs->as_r64<XbyakSReg64>());
            }
        } else if (p->is_op("&&")) {
            if (p->rhs->is_imm())
                and_(dst, p->rhs->as_imm32() ? 1 : 0);
            else {
                and_(dst, p->rhs->as_r64<XbyakSReg64>());
            }
        } else if (p->is_op("||")) {
            if (p->rhs->is_imm())
                or_(dst, p->rhs->as_imm32() ? 1 : 0);
            else {
                or_(dst, p->rhs->as_r64<XbyakSReg64>());
            }
        } else if (p->is_op("!")) {
            xor_(dst, 1);
        } else if (p->is_op("^")) {
            if (p->rhs->is_imm())
                xor_(dst, p->rhs->as_imm32());
            else {
                xor_(dst, p->rhs->as_r64<XbyakSReg64>());
            }
        } else if (p->is_cmp()) {
            if (p->rhs->is_imm())
                cmp(dst, p->rhs->as_imm32());
            else {
                cmp(dst, p->rhs->as_r64<XbyakSReg64>());
            }
            if (!(do_jump && p == pimpl)) {
                // generate boolean value based on cmp results
                if (do_assign)
                    mov(dst, 0);  // note only lowest byte is set, clear high bytes
                if (p->is_op("=="))
                    sete(dst.cvt8());
                if (p->is_op("!="))
                    setne(dst.cvt8());
                if (p->is_op(">"))
                    setg(dst.cvt8());
                if (p->is_op(">="))
                    setge(dst.cvt8());
                if (p->is_op("<"))
                    setl(dst.cvt8());
                if (p->is_op("<="))
                    setle(dst.cvt8());
            }
        } else {
            OPENVINO_ASSERT(0, "Unsupported OP: ", p->op);
        }
        return true;
    });

    if (pdst) {
        if (assign_op == '=' && !dst_register_assigned_inplace) {
            mov(*pdst, pimpl->as_r64<XbyakSReg64>());
        } else {
            switch (assign_op) {
            case '=':
                break;
            case '+':
                add(*pdst, pimpl->as_r64<XbyakSReg64>());
                break;
            case '-':
                sub(*pdst, pimpl->as_r64<XbyakSReg64>());
                break;
            case '*':
                imul(*pdst, pimpl->as_r64<XbyakSReg64>());
                break;
            default:
                OPENVINO_ASSERT(false);
                break;
            }
        }
    }

    // generate jump
    if (assign_op == 'T') {
        if (pimpl->is_cmp()) {
            if (pimpl->is_op("=="))
                je(label, T_NEAR);
            if (pimpl->is_op("!="))
                jne(label, T_NEAR);
            if (pimpl->is_op(">"))
                jg(label, T_NEAR);
            if (pimpl->is_op(">="))
                jge(label, T_NEAR);
            if (pimpl->is_op("<"))
                jl(label, T_NEAR);
            if (pimpl->is_op("<="))
                jle(label, T_NEAR);
        } else {
            // convert final value to ZF
            test(pimpl->as_r64<XbyakSReg64>(), pimpl->as_r64<XbyakSReg64>());
            jnz(label, T_NEAR);
        }
    } else if (assign_op == 'F') {
        if (pimpl->is_cmp()) {
            if (pimpl->is_op("=="))
                jne(label, T_NEAR);
            if (pimpl->is_op("!="))
                je(label, T_NEAR);
            if (pimpl->is_op(">"))
                jle(label, T_NEAR);
            if (pimpl->is_op(">="))
                jl(label, T_NEAR);
            if (pimpl->is_op("<"))
                jge(label, T_NEAR);
            if (pimpl->is_op("<="))
                jg(label, T_NEAR);
        } else {
            // convert final value to ZF
            test(pimpl->as_r64<XbyakSReg64>().cvt8(), pimpl->as_r64<XbyakSReg64>().cvt8());
            jz(label, T_NEAR);
        }
    }
}
#endif

//
#ifdef __aarch64__
inline const SReg& SReg::operator=(const SReg& rhs) const {
    jit->mov(*reg, rhs);
    return *this;
}
inline const SReg& SReg::operator=(SRegExpr&& expr) const {
    jit->evaluate(expr, this, '=');
    return *this;
}
inline const SReg& SReg::operator+=(SRegExpr&& expr) const {
    jit->evaluate(expr, this, '+');
    return *this;
}
inline const SReg& SReg::operator-=(SRegExpr&& expr) const {
    jit->evaluate(expr, this, '-');
    return *this;
}
inline const SReg& SReg::operator*=(SRegExpr&& expr) const {
    jit->evaluate(expr, this, '*');
    return *this;
}
inline void SReg::operator++() const {
    jit->add(*reg, *reg, 1);
}
inline void SReg::operator--() const {
    jit->sub(*reg, *reg, 1);
}
inline void SReg::operator++(int) const {
    jit->add(*reg, *reg, 1);
}
inline void SReg::operator--(int) const {
    jit->sub(*reg, *reg, 1);
}

inline void SIMDJit::evaluate(SRegExpr& expr,
                              const SReg* pdst,
                              const char assign_op,
                              const Xbyak_aarch64::Label& label) {
    int debug_log = SIMDJIT_DEBUG & 1;
    if (debug_log) {
        std::cout << "\033[32m==========================================\033[0m" << std::endl;
    }
    auto jit_dis = get_disasm(debug_log);

    auto paddr = expr.paddr.get();
    auto pimpl = expr.pimpl.get();
    // do_jump: the expression as condition of control-flow, will not be assigned to any register
    //          instead it will emmit `jump` instruction:
    // assign_op == 'T' jump to label if expression is true
    // assign_op == 'F' jump to label if expression is false
    const bool do_jump = (assign_op == 'T') || (assign_op == 'F');
    const bool do_assign = (pdst != nullptr) && (!do_jump);

    if (debug_log) {
        pimpl->show_rpn();
        if (pdst)
            std::cout << assign_op << " assign-to : r" << pdst->r64().getIdx() << std::endl;
    }

    // short expression optimization
    if (pdst) {
        auto& dst = *pdst;
        auto* lhs = pimpl;
        if (lhs->is_reg()) {
            switch (assign_op) {
            case '=':
                mov(dst, lhs->as_r64<XbyakSReg64>());
                break;
            case '+':
                add(dst, dst, lhs->as_r64<XbyakSReg64>());
                break;
            case '-':
                sub(dst, dst, lhs->as_r64<XbyakSReg64>());
                break;
            case '*':
                mul(dst, dst, lhs->as_r64<XbyakSReg64>());
                break;
            default:
                OPENVINO_ASSERT(false);
                break;
            }
            return;
        }
        if (lhs->is_imm()) {
            auto imm32 = lhs->as_imm32();
            char actual_op = assign_op;
            if (actual_op == '=') {
                mov(dst, imm32);
                return;
            }
            if (actual_op == '+' && imm32 < 0) {
                actual_op = '-';
                imm32 = -imm32;
            }
            if (actual_op == '-' && imm32 < 0) {
                actual_op = '+';
                imm32 = -imm32;
            }
            if (imm32 >= 0 && imm32 <= 4095) {
                if (actual_op == '+') {
                    add(dst, dst, imm32);
                    return;
                }
                if (actual_op == '-') {
                    sub(dst, dst, imm32);
                    return;
                }
            }
            auto imm_reg = get_sreg();
            mov(imm_reg, lhs->as_imm32());

            switch (actual_op) {
            case '+':
                add(dst, dst, imm_reg);
                break;
            case '-':
                sub(dst, dst, imm_reg);
                break;
            case '*':
                mul(dst, dst, imm_reg);
                break;
            default:
                OPENVINO_ASSERT(false);
                break;
            }
            return;
        }
    }
    //
}

#endif
}  // namespace intel_cpu
}  // namespace ov
