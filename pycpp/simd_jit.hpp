// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
//#include <memory>
#include <sstream>
#include <vector>

#define JIT_DEBUG 1
#include "../include/jit.h"
#define DECLARE_CPU_JIT_AUX_FUNCTIONS(x)
static const bool use_avx512 = false;
template <typename... TS>
void _write_all(std::ostream& os, TS&&... args) {
    int dummy[sizeof...(TS)] = {(os << std::forward<TS>(args), 0)...};
    (void)dummy;
}
#define OPENVINO_ASSERT(cond, ...)                                                      \
    if (!(cond)) {                                                                      \
        std::stringstream ss;                                                           \
        _write_all(ss, __FILE__, ":", __LINE__, " ", #cond, " failed:", ##__VA_ARGS__); \
        std::cout << "\033[31m" << ss.str() << "\033[0m" << std::endl;                  \
        asm("int3");                                                                    \
        throw std::runtime_error(ss.str());                                             \
    }

namespace ov {
namespace intel_cpu {

#ifdef _WIN32
#    define abi_param_regs_num 4
#else
#    define abi_param_regs_num 6
#endif
// first few regs contains input arguments passed in through stack
constexpr Xbyak::Operand::Code abi_x86_64_regs[] = {
#ifdef _WIN32
    // args passed in register
    Xbyak::Operand::RCX,
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
    Xbyak::Operand::R15
#else
    // args passed in register
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
    Xbyak::Operand::R15
#endif
};

static const int SIMDJIT_DEBUG = std::getenv("SIMDJIT_DEBUG") ? std::atoi(std::getenv("SIMDJIT_DEBUG")) : 0;

class SIMDJit;

class SRegExpr;

class SReg {
private:
    SIMDJit* jit = nullptr;
    std::shared_ptr<Xbyak::Reg64> reg;

public:
    SReg(SIMDJit* jit, std::shared_ptr<Xbyak::Reg64> reg) : jit(jit), reg(reg) {}

    SReg() = default;

    bool empty() const {
        return !static_cast<bool>(reg);
    }
    operator Xbyak::Reg64&() {
        return *reg;
    }
    operator const Xbyak::Reg64&() const {
        return *reg;
    }
    Xbyak::Reg64& r64() {
        return *reg;
    }
    const Xbyak::Reg64& r64() const {
        return *reg;
    }
    Xbyak::Reg32 r32() {
        return (*reg).cvt32();
    }
    const Xbyak::Reg32 r32() const {
        return (*reg).cvt32();
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
    std::shared_ptr<Xbyak::Xmm> reg;

public:
    VReg(SIMDJit* jit, std::shared_ptr<Xbyak::Xmm> reg) : jit(jit), reg(reg) {}

    VReg() = default;

    bool empty() const {
        return !static_cast<bool>(reg);
    }
    operator Xbyak::Xmm&() {
        return *reg;
    }
    operator const Xbyak::Xmm&() const {
        return *reg;
    }
};

class reg_pool {
public:
    reg_pool(const char* name, size_t max_num_regs) : m_name(name), m_status(max_num_regs, 0) {}

    int allocate(int index = -1) {
        // allocate register with specific index
        if (index >= 0) {
            OPENVINO_ASSERT(static_cast<size_t>(index) < m_status.size());
            OPENVINO_ASSERT(m_status[index] == 0);
            m_status[index] = 1;
            return index;
        }

        auto it = std::find(m_status.begin(), m_status.end(), 0);
        OPENVINO_ASSERT(it != m_status.end(), "regiter pool ", m_name, " exhausted.");
        *it = 1;
        return it - m_status.begin();
    }
    void free(int i) {
        m_status[i] = 0;
    }
    void clear() {
        for (auto& s : m_status)
            s = 0;
    }

private:
    const char* m_name;
    std::vector<int> m_status;
};

struct RegExprImpl {
    // "r" register
    // "i" imm32
    // "+"/"-"/..... normal binary OP
    const char* op;
    int data = -1;
    std::unique_ptr<RegExprImpl> lhs;
    std::unique_ptr<RegExprImpl> rhs;

    Xbyak::Reg64 as_r64() {
        OPENVINO_ASSERT(!is_op("i"));
        return Xbyak::Reg64(data);
    }
    int as_imm32() {
        OPENVINO_ASSERT(is_op("i"));
        return data;
    }

    bool is_leaf() const {
        return (!lhs) && (!rhs);
    }
    bool is_reg() const {
        return is_op("r");
    }
    bool is_imm() const {
        return is_op("i");
    }
    bool is_cmp() const {
        return is_op(">") || is_op(">=") || is_op("<") || is_op("<=") || is_op("==") || is_op("!=");
    }
    bool is_logical_op() const {
        return is_cmp() || is_op("&&") || is_op("||") || is_op("!");
    }
    bool is_op(const char* name = nullptr) const {
        // all nodes other than leaf is op
        if (name == nullptr)
            return !is_leaf();

        // check op type
        if (op[1] == 0)
            return op[0] == name[0] && op[1] == name[1];
        else if (op[2] == 0)
            return op[0] == name[0] && op[1] == name[1] && op[2] == name[2];
        return false;
    }
    std::string to_string() const {
        if (is_leaf()) {
            if (std::string(op) == "i")
                return std::to_string(data);
            return std::string(op) + std::to_string(data);
        }
        return std::string("t") + std::to_string(data) + " = " + lhs->to_string() + op + rhs->to_string();
    }

    std::string name() const {
        if (is_leaf()) {
            if (is_imm()) {
                return std::to_string(data);
            } else {
                return std::string(op) + std::to_string(data);
            }
        }
        return data >= 0 ? std::string("r") + std::to_string(data)
                         : "@" + std::to_string(reinterpret_cast<uintptr_t>(this));
    }

    void show_rpn() const {
        std::cout << "\033[32m::::"
                  << " orignal expression "
                  << "::::\033[0m" << std::endl;
        std::cout << "infix expression: ";
        _show_rpn(this, true);
        std::cout << std::endl;
        std::cout << "suffix expression: ";
        _show_rpn(this, false);
        std::cout << std::endl;
    }
    void _show_rpn(const RegExprImpl* pimpl, bool infix) const {
        if (!pimpl)
            return;
        if (pimpl->is_leaf()) {
            std::cout << pimpl->name();
            return;
        }
        if (infix) {
            if (!pimpl->rhs) {
                std::cout << "(" << pimpl->op;
                _show_rpn(pimpl->lhs.get(), infix);
                std::cout << ")";
            } else {
                std::cout << "(";
                _show_rpn(pimpl->lhs.get(), infix);
                std::cout << pimpl->op;
                _show_rpn(pimpl->rhs.get(), infix);
                std::cout << ")";
            }
        } else {
            std::cout << "(";
            _show_rpn(pimpl->lhs.get(), infix);
            std::cout << ",";
            _show_rpn(pimpl->rhs.get(), infix);
            std::cout << ")" << pimpl->op;
        }
    }

    RegExprImpl(const char* op, int data) : op(op), data(data) {}
    RegExprImpl(const char* op, std::unique_ptr<RegExprImpl>& _lhs) : op(op), lhs(std::move(_lhs)) {}
    RegExprImpl(const char* op, std::unique_ptr<RegExprImpl>& _lhs, std::unique_ptr<RegExprImpl>& _rhs)
        : op(op),
          lhs(std::move(_lhs)),
          rhs(std::move(_rhs)) {}

    // for_each_op all op
    bool for_each_op(const std::function<bool(RegExprImpl* node)>& callback, RegExprImpl* root = nullptr) {
        if (root == nullptr)
            root = this;

        if (root->is_leaf())
            return true;

        if (root->lhs && !root->lhs->is_leaf()) {
            if (!for_each_op(callback, root->lhs.get()))
                return false;  // early terminate
        }
        if (root->rhs && !root->rhs->is_leaf()) {
            if (!for_each_op(callback, root->rhs.get()))
                return false;  // early terminate
        }
        return callback(root);
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

    // convert to address
    operator Xbyak::RegExp() const {
        OPENVINO_ASSERT(paddr);

        if (paddr->base_reg < 0) {
            OPENVINO_ASSERT(paddr->index_reg >= 0);
            return Xbyak::Reg64(paddr->index_reg) * paddr->scale + paddr->disp;
        } else if (paddr->index_reg >= 0)
            return Xbyak::Reg64(paddr->base_reg) + Xbyak::Reg64(paddr->index_reg) * paddr->scale + paddr->disp;
        else
            return Xbyak::Reg64(paddr->base_reg) + paddr->disp;
    }

    inline void evaluate(SIMDJit* jit,
                         const SReg* pdst = nullptr,
                         const char assign_op = '=',
                         const Xbyak::Label& label = {});
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
            auto ret = std::system("objdump -D -b binary -mi386:x86-64 -M intel temp.bin");
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

    SIMDJit(const char* name)
        : jit_generator(name),
          sreg_pool("sreg_pool", sizeof(abi_x86_64_regs) / sizeof(abi_x86_64_regs[0])),
          vreg_pool("vreg_pool", use_avx512 ? 32 : 16) {
        mov(rax, rsp);
        preamble();
    }

    void generate() override{};

    // add an int64_t return value
    template <typename... kernel_args_t>
    int64_t operator()(kernel_args_t... args) const {
        using jit_kernel_func_t = int64_t (*)(const kernel_args_t... args);
        auto* fptr = (jit_kernel_func_t)jit_ker();
        return (*fptr)(std::forward<kernel_args_t>(args)...);
    }

    void return_(int imm32 = 0) {
        mov(rax, imm32);
        postamble();
    }

    void return_(Xbyak::Reg64 return_value = Xbyak::Reg64(Xbyak::Operand::RAX)) {
        if (return_value.getIdx() != rax.getIdx())
            mov(rax, return_value);
        postamble();
    }

    void finalize(Xbyak::Reg64 return_value = Xbyak::Reg64(Xbyak::Operand::RAX)) {
        if (return_value.getIdx() != rax.getIdx())
            mov(rax, return_value);
        postamble();
        create_kernel();
    }

    SReg get_arg(int idx) {
        auto ret = SReg(this, alloc_reg64(idx));
        if (idx >= abi_param_regs_num)
            mov(ret, ptr[rax + (idx - abi_param_regs_num + 1) * 8]);  // load from stack
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

    // simd_xxx helpers have meaning similar to x86 intrinsics
    // it's more well-known than raw instruction can it also can be
    // made cross-platform(avx2/avx512/neon/...)

    void simd_set1_epi32(Xbyak::Xmm vmm, int32_t imm32) {
        // this set1 is not performance critical
        mov(dword[rsp - 4], imm32);
        vpbroadcastd(vmm, dword[rsp - 4]);
    }
    void simd_and(Xbyak::Xmm c, Xbyak::Xmm a, Xbyak::Xmm b) {
        if (use_avx512) {
            vpandd(c, a, b);
        } else {
            vpand(c, a, b);
        }
    }
    void simd_srli_epi32(Xbyak::Xmm vdst, Xbyak::Xmm vsrc, int32_t imm8) {
        vpsrld(vdst, vsrc, imm8);
    }
    void simd_srai_epi32(Xbyak::Xmm vdst, Xbyak::Xmm vsrc, int32_t imm8) {
        vpsrad(vdst, vsrc, imm8);
    }
    void simd_slli_epi32(Xbyak::Xmm vdst, Xbyak::Xmm vsrc, int32_t imm8) {
        vpslld(vdst, vsrc, imm8);
    }
    void simd_setzero_ps(Xbyak::Xmm vmm) {
        if (use_avx512) {
            vpxord(vmm, vmm, vmm);
        } else {
            vpxor(vmm, vmm, vmm);
        }
    }
    void simd_loadu_ps(Xbyak::Xmm vmm, const Xbyak::Address& addr) {
        vmovups(vmm, addr);
    }
    // load packed half into packed single
    void simd_loadu_phps(Xbyak::Xmm vmm, const Xbyak::Address& addr) {
        vcvtph2ps(vmm, addr);
    }
    void simd_load_epu8_epi32(Xbyak::Xmm vmm, const Xbyak::Address& addr) {
        vpmovzxbd(vmm, addr);
    }
    void simd_load_epi8_epi32(Xbyak::Xmm vmm, const Xbyak::Address& addr) {
        vpmovsxbd(vmm, addr);
    }
    void simd_storeu_ps(const Xbyak::Address& addr, Xbyak::Xmm vmm) {
        vmovups(addr, vmm);
    }
    void simd_fmadd_ps(Xbyak::Xmm c, Xbyak::Xmm a, const Xbyak::Operand& b) {
        vfmadd231ps(c, a, b);
    }
    void simd_sub_ps(Xbyak::Xmm c, Xbyak::Xmm a, Xbyak::Xmm b) {
        vsubps(c, a, b);
    }
    void simd_add_ps(Xbyak::Xmm c, Xbyak::Xmm a, Xbyak::Xmm b) {
        vaddps(c, a, b);
    }
    void simd_mul_ps(Xbyak::Xmm c, Xbyak::Xmm a, Xbyak::Xmm b) {
        vmulps(c, a, b);
    }
    void simd_broadcast_ss(Xbyak::Xmm vmm, const Xbyak::Address& addr) {
        vbroadcastss(vmm, addr);
    }
    void simd_cvtepi32_ps(Xbyak::Xmm vmm_dst, Xbyak::Xmm vmm_src) {
        vcvtdq2ps(vmm_dst, vmm_src);
    }

    //***********************************************
    // for_loop(idx, start, stop, step, loop_body) performs following:
    //    for(int idx=start; idx + step <= stop; idx+=step) {
    //       loop_body();
    //    }
    template <typename Fn, typename START, typename STEP>
    void for_loop(Xbyak::Reg64 idx, START start, Xbyak::Reg64 stop, STEP step, const Fn& loop_body) {
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

    //***********************************************
    // while_(rax > 0, loop_body) performs following:
    //    while(rax > 0) {
    //       loop_body();
    //    }
    template <typename Fn>
    void while_(SRegExpr regcmp, const Fn& loop_body) {
        Xbyak::Label loop, exit;

        align(64, false);
        L(loop);

        regcmp.evaluate(this, nullptr, 'F', exit);

        loop_body();

        jmp(loop, T_NEAR);
        L(exit);
    }

    template <typename Fn>
    void do_while_(SRegExpr regcmp, const Fn& loop_body) {
        Xbyak::Label loop;

        align(64, false);
        L(loop);

        loop_body();

        regcmp.evaluate(this, nullptr, 'T', loop);
    }

    void if_(SRegExpr regcmp, const std::function<void()>& then_body, const std::function<void()>& else_body = {}) {
        Xbyak::Label if_else, if_exit;

        regcmp.evaluate(this, nullptr, 'F', if_else);

        then_body();

        if (else_body)
            jmp(if_exit, T_NEAR);

        L(if_else);

        if (else_body)
            else_body();

        L(if_exit);
    }

    template <typename DT>
    static int vmm_width() {
        return (use_avx512 ? 512 : 256) / (sizeof(DT) * 8);
    }

private:
    reg_pool sreg_pool;
    reg_pool vreg_pool;

    std::shared_ptr<Xbyak::Reg64> alloc_reg64(int index) {
        auto reg_index = sreg_pool.allocate(index);
        return std::shared_ptr<Xbyak::Reg64>(new Xbyak::Reg64(abi_x86_64_regs[reg_index]),
                                             [this, reg_index](Xbyak::Reg64* preg) {
                                                 if (preg) {
                                                     sreg_pool.free(reg_index);
                                                     delete preg;
                                                 }
                                             });
    }

    std::shared_ptr<Xbyak::Xmm> alloc_vreg(int index) {
        auto reg_index = vreg_pool.allocate(index);
        if (use_avx512) {
            return std::shared_ptr<Xbyak::Xmm>(new Xbyak::Zmm(reg_index), [this, reg_index](Xbyak::Zmm* preg) {
                vreg_pool.free(reg_index);
                delete preg;
            });
        } else {
            return std::shared_ptr<Xbyak::Xmm>(new Xbyak::Ymm(reg_index), [this, reg_index](Xbyak::Ymm* preg) {
                vreg_pool.free(reg_index);
                delete preg;
            });
        }
    }
};

inline const SReg& SReg::operator=(const SReg& rhs) const {
    jit->mov(*reg, rhs);
    return *this;
}
inline const SReg& SReg::operator=(SRegExpr&& expr) const {
    expr.evaluate(jit, this, '=');
    return *this;
}
inline const SReg& SReg::operator+=(SRegExpr&& expr) const {
    expr.evaluate(jit, this, '+');
    return *this;
}
inline const SReg& SReg::operator-=(SRegExpr&& expr) const {
    expr.evaluate(jit, this, '-');
    return *this;
}
inline const SReg& SReg::operator*=(SRegExpr&& expr) const {
    expr.evaluate(jit, this, '*');
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

inline void SRegExpr::evaluate(SIMDJit* jit, const SReg* pdst, const char assign_op, const Xbyak::Label& label) {
    int debug_log = SIMDJIT_DEBUG & 1;
    if (debug_log) {
        std::cout << "\033[32m==========================================\033[0m" << std::endl;
    }
    auto jit_dis = jit->get_disasm(debug_log);

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
        auto* lhs = pimpl.get();
        if (lhs->is_reg()) {
            switch (assign_op) {
            case '=':
                jit->mov(dst, lhs->as_r64());
                break;
            case '+':
                jit->add(dst, lhs->as_r64());
                break;
            case '-':
                jit->sub(dst, lhs->as_r64());
                break;
            case '*':
                jit->imul(dst, lhs->as_r64());
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
                jit->mov(dst, lhs->as_imm32());
                break;
            case '+':
                jit->add(dst, lhs->as_imm32());
                break;
            case '-':
                jit->sub(dst, lhs->as_imm32());
                break;
            case '*':
                jit->imul(dst, dst, lhs->as_imm32());
                break;
            default:
                OPENVINO_ASSERT(false);
                break;
            }
            return;
        }
        // addressing expression
        if (paddr) {
            if (assign_op == '=') {
                jit->lea(dst, jit->ptr[static_cast<Xbyak::RegExp>(*this)]);
                return;
            } else {
                auto temp = jit->get_sreg();
                jit->lea(temp, jit->ptr[static_cast<Xbyak::RegExp>(*this)]);
                switch (assign_op) {
                case '+':
                    jit->add(dst, temp);
                    break;
                case '-':
                    jit->sub(dst, temp);
                    break;
                case '*':
                    jit->imul(dst, temp);
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
        show(" After const folding");

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
        if (do_jump && p == pimpl.get() && p->is_cmp() && p->lhs->is_reg()) {
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
        show(" After scratch reg allocation & convert to 2-OP form");

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
        show(" After replace dst scratch register");

    // allocate physical registers
    std::map<int, SReg> scratch_regs;
    pimpl->for_each_op([&](RegExprImpl* p) {
        if (p->data >= scratch_reg_base) {
            auto it = scratch_regs.find(p->data);
            if (it != scratch_regs.end()) {
                p->data = it->second.r64().getIdx();
            } else {
                // allocate new scratch reg
                auto sreg = jit->get_sreg();
                scratch_regs.emplace(p->data, sreg);
                p->data = sreg.r64().getIdx();
            }
        }
        return true;
    });

    if (debug_log)
        show(" After allocation of all scratch registers");

    // emmit code
    pimpl->for_each_op([&](RegExprImpl* p) {
        auto dst = Xbyak::Reg64(p->data);
        if (p->is_op(" ")) {
            if (p->lhs->is_imm())
                jit->mov(dst, p->lhs->as_imm32());
            else
                jit->mov(dst, p->lhs->as_r64());
        } else if (p->is_op("+")) {
            if (p->rhs->is_imm())
                jit->add(dst, p->rhs->as_imm32());
            else
                jit->add(dst, p->rhs->as_r64());
        } else if (p->is_op("n+")) {
            jit->neg(dst);
            if (p->rhs->is_imm()) {
                jit->add(dst, p->rhs->as_imm32());
            } else
                jit->add(dst, p->rhs->as_r64());
        } else if (p->is_op("*")) {
            if (p->rhs->is_imm()) {
                // support 3-OP
                jit->imul(dst, p->lhs->as_r64(), p->rhs->as_imm32());
            } else {
                jit->imul(dst, p->rhs->as_r64());
            }
        } else if (p->is_op("-")) {
            if (p->rhs->is_imm())
                jit->sub(dst, p->rhs->as_imm32());
            else
                jit->sub(dst, p->rhs->as_r64());
        } else if (p->is_op(">>")) {
            if (p->rhs->is_imm())
                jit->sar(dst, p->rhs->as_imm32());
            else {
                // only cl register supportted, we need allocate cl
                OPENVINO_ASSERT(false);  // jit->sar(dst, p->rhs->as_r64());
            }
        } else if (p->is_op("<<")) {
            if (p->rhs->is_imm())
                jit->shl(dst, p->rhs->as_imm32());
            else {
                // only cl register supportted, we need allocate cl
                OPENVINO_ASSERT(false);  // jit->shl(dst, p->rhs->as_r64());
            }
        } else if (p->is_op("&")) {
            if (p->rhs->is_imm())
                jit->and_(dst, p->rhs->as_imm32());
            else {
                jit->and_(dst, p->rhs->as_r64());
            }
        } else if (p->is_op("|")) {
            if (p->rhs->is_imm())
                jit->or_(dst, p->rhs->as_imm32());
            else {
                jit->or_(dst, p->rhs->as_r64());
            }
        } else if (p->is_op("&&")) {
            if (p->rhs->is_imm())
                jit->and_(dst, p->rhs->as_imm32() ? 1 : 0);
            else {
                jit->and_(dst, p->rhs->as_r64());
            }
        } else if (p->is_op("||")) {
            if (p->rhs->is_imm())
                jit->or_(dst, p->rhs->as_imm32() ? 1 : 0);
            else {
                jit->or_(dst, p->rhs->as_r64());
            }
        } else if (p->is_op("!")) {
            jit->xor_(dst, 1);
        } else if (p->is_op("^")) {
            if (p->rhs->is_imm())
                jit->xor_(dst, p->rhs->as_imm32());
            else {
                jit->xor_(dst, p->rhs->as_r64());
            }
        } else if (p->is_cmp()) {
            if (p->rhs->is_imm())
                jit->cmp(dst, p->rhs->as_imm32());
            else {
                jit->cmp(dst, p->rhs->as_r64());
            }
            if (!(do_jump && p == pimpl.get())) {
                // generate boolean value based on cmp results
                if (do_assign)
                    jit->mov(dst, 0);  // note only lowest byte is set, clear high bytes
                if (p->is_op("=="))
                    jit->sete(dst.cvt8());
                if (p->is_op("!="))
                    jit->setne(dst.cvt8());
                if (p->is_op(">"))
                    jit->setg(dst.cvt8());
                if (p->is_op(">="))
                    jit->setge(dst.cvt8());
                if (p->is_op("<"))
                    jit->setl(dst.cvt8());
                if (p->is_op("<="))
                    jit->setle(dst.cvt8());
            }
        } else {
            OPENVINO_ASSERT(0, "Unsupported OP: ", p->op);
        }
        return true;
    });

    if (pdst) {
        if (assign_op == '=' && !dst_register_assigned_inplace) {
            jit->mov(*pdst, pimpl->as_r64());
        } else {
            switch (assign_op) {
            case '=':
                break;
            case '+':
                jit->add(*pdst, pimpl->as_r64());
                break;
            case '-':
                jit->sub(*pdst, pimpl->as_r64());
                break;
            case '*':
                jit->imul(*pdst, pimpl->as_r64());
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
                jit->je(label, jit->T_NEAR);
            if (pimpl->is_op("!="))
                jit->jne(label, jit->T_NEAR);
            if (pimpl->is_op(">"))
                jit->jg(label, jit->T_NEAR);
            if (pimpl->is_op(">="))
                jit->jge(label, jit->T_NEAR);
            if (pimpl->is_op("<"))
                jit->jl(label, jit->T_NEAR);
            if (pimpl->is_op("<="))
                jit->jle(label, jit->T_NEAR);
        } else {
            // convert final value to ZF
            jit->test(pimpl->as_r64(), pimpl->as_r64());
            jit->jnz(label, jit->T_NEAR);
        }
    } else if (assign_op == 'F') {
        if (pimpl->is_cmp()) {
            if (pimpl->is_op("=="))
                jit->jne(label, jit->T_NEAR);
            if (pimpl->is_op("!="))
                jit->je(label, jit->T_NEAR);
            if (pimpl->is_op(">"))
                jit->jle(label, jit->T_NEAR);
            if (pimpl->is_op(">="))
                jit->jl(label, jit->T_NEAR);
            if (pimpl->is_op("<"))
                jit->jge(label, jit->T_NEAR);
            if (pimpl->is_op("<="))
                jit->jg(label, jit->T_NEAR);
        } else {
            // convert final value to ZF
            jit->test(pimpl->as_r64().cvt8(), pimpl->as_r64().cvt8());
            jit->jz(label, jit->T_NEAR);
        }
    }
}

}  // namespace intel_cpu
}  // namespace ov

std::vector<std::function<void(void)>> test_exprs;

#define TEST_EXPR(expr_name)                                                                 \
    test_exprs.push_back([] {                                                                \
        auto jit = std::make_shared<ov::intel_cpu::SIMDJit>(__func__);                       \
        {                                                                                    \
            auto dst = jit->get_arg(0);                                                      \
            auto a = jit->get_arg(1);                                                        \
            auto b = jit->get_arg(2);                                                        \
            auto c = jit->get_arg(3);                                                        \
            auto d = jit->get_arg(4);                                                        \
            auto e = jit->get_arg(5);                                                        \
            auto f = jit->get_arg(6);                                                        \
            expr_name(dst, a, b, c, d, e, f);                                                \
            jit->finalize(dst);                                                              \
        }                                                                                    \
        int a = 2;                                                                           \
        int b = 3;                                                                           \
        int c = 4;                                                                           \
        int d = 5;                                                                           \
        int e = 6;                                                                           \
        int f = 7;                                                                           \
        int dst = 10;                                                                        \
        auto result = (*jit)(dst, a, b, c, d, e, f);                                         \
        expr_name(dst, a, b, c, d, e, f);                                                    \
        if (result != dst) {                                                                 \
            std::cout << #expr_name << ":" << result << " != expected " << dst << std::endl; \
            ASSERT(false);                                                                   \
        } else {                                                                             \
            std::cout << #expr_name << ":" << result << " == expected " << dst << std::endl; \
        }                                                                                    \
    });

#define TEST_WHILE_EXPR(cond_expr, body_expr)                                                \
    test_exprs.push_back([] {                                                                \
        auto jit = std::make_shared<ov::intel_cpu::SIMDJit>(__func__);                       \
        {                                                                                    \
            auto dst = jit->get_arg(0);                                                      \
            auto a = jit->get_arg(1);                                                        \
            auto b = jit->get_arg(2);                                                        \
            auto c = jit->get_arg(3);                                                        \
            auto d = jit->get_arg(4);                                                        \
            auto e = jit->get_arg(5);                                                        \
            auto f = jit->get_arg(6);                                                        \
            jit->while_(cond_expr(dst, a, b, c, d, e, f), [&] {                              \
                body_expr(dst, a, b, c, d, e, f);                                            \
            });                                                                              \
            jit->finalize(dst);                                                              \
        }                                                                                    \
        int a = 2;                                                                           \
        int b = 3;                                                                           \
        int c = 4;                                                                           \
        int d = 5;                                                                           \
        int e = 6;                                                                           \
        int f = 7;                                                                           \
        int dst = 10;                                                                        \
        auto result = (*jit)(dst, a, b, c, d, e, f);                                         \
        while (cond_expr(dst, a, b, c, d, e, f)) {                                           \
            body_expr(dst, a, b, c, d, e, f);                                                \
        }                                                                                    \
        if (result != dst) {                                                                 \
            std::cout << #cond_expr << ":" << result << " != expected " << dst << std::endl; \
            ASSERT(false);                                                                   \
        } else {                                                                             \
            std::cout << #cond_expr << ":" << result << " == expected " << dst << std::endl; \
        }                                                                                    \
    });

#define EXPR0(dst, a, b, c, d, e, f) dst = (a + 6)
#define EXPR1(dst, a, b, c, d, e, f) dst = (a - b * 4)
#define EXPR2(dst, a, b, c, d, e, f) dst = (a * ((b << 2) ^ (c >> 1)))
#define EXPR3(dst, a, b, c, d, e, f) dst = (a + b | c - c & 8)
#define EXPR4(dst, a, b, c, d, e, f) dst = (a + b * (c + d) * 8 + e)
#define EXPR5(dst, a, b, c, d, e, f) dst = (a + b * (c - (d + e)) * 8 + e * (f - a))
#define EXPR6(dst, a, b, c, d, e, f) dst += 2
#define EXPR7(dst, a, b, c, d, e, f) dst = a + (dst * b) * 4
#define EXPR8(dst, a, b, c, d, e, f) dst = a * 3 * sizeof(float)
#define EXPR9(dst, a, b, c, d, e, f) dst = dst + 4 + 9 + a + 3 * sizeof(float) + 8
#define EXPRA(dst, a, b, c, d, e, f) dst++
#define EXPRB(dst, a, b, c, d, e, f) dst--
#define EXPRC(dst, a, b, c, d, e, f) dst = ((a > 1) || (b < 2)) && (!(f == 0))

#define COND_EXPR0(dst, a, b, c, d, e, f)  (dst == 100)
#define WHILE_EXPR0(dst, a, b, c, d, e, f) dst += 1

#define COND_EXPR1(dst, a, b, c, d, e, f)  (dst + a * 4) < (80 * f >> 2)
#define WHILE_EXPR1(dst, a, b, c, d, e, f) dst += (a >> 1)

#define COND_EXPR2(dst, a, b, c, d, e, f)  (dst == 100) || (dst == 80) && !(dst < 70)
#define WHILE_EXPR2(dst, a, b, c, d, e, f) dst += (a >> 1)

void unit_test() {
    TEST_EXPR(EXPR0);
    TEST_EXPR(EXPR1);
    TEST_EXPR(EXPR2);
    TEST_EXPR(EXPR3);
    TEST_EXPR(EXPR4);
    TEST_EXPR(EXPR5);
    TEST_EXPR(EXPR6);
    TEST_EXPR(EXPR7);
    TEST_WHILE_EXPR(COND_EXPR0, WHILE_EXPR0);
    TEST_WHILE_EXPR(COND_EXPR1, WHILE_EXPR1);
    TEST_WHILE_EXPR(COND_EXPR2, WHILE_EXPR2);
    TEST_EXPR(EXPR8);
    TEST_EXPR(EXPR9);
    TEST_EXPR(EXPRA);
    TEST_EXPR(EXPRB);
    TEST_EXPR(EXPRC);

    for (auto& func : test_exprs) {
        std::cout << "\n\n";
        func();
    }
}

struct AAA {
    AAA() {
        std::cout << "Default constructor is called\n";
    }
    AAA(const AAA& rhs) {
        std::cout << "Copy constructor is called\n";
    }
    const AAA& operator=(const AAA& rhs) {
        std::cout << "Assign operator is called\n";
    }
};
void test_operator_overload() {
    AAA a;
    AAA b = a;
    AAA c[4] = {a, a, a, a};
    c[0] = a;
}

extern "C" void test() {
    unit_test();
    // test_operator_overload();
}
