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
using XbyakTReg = Xbyak::Tmm;
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

    struct jcout {
        Xbyak::CodeGenerator* jit;
        std::vector<Xbyak::Reg64> preserved_regs;
        std::vector<Xbyak::Xmm> preserved_vmms;
        int vmm_size_byte;
        jcout(Xbyak::CodeGenerator* jit) : jit(jit) {
            preserved_regs = {jit->rbp,
                              jit->rsp,
                              jit->rax,
                              jit->rbx,
                              jit->rcx,
                              jit->rdx,
                              jit->rdi,
                              jit->rsi,
                              jit->r8,
                              jit->r9,
                              jit->r10,
                              jit->r11,
                              jit->r12,
                              jit->r13,
                              jit->r14,
                              jit->r15};
            for (int i = 0; i < 16; i++) {
                preserved_vmms.push_back(Xbyak::Ymm(i));
            }
            vmm_size_byte = preserved_vmms[0].getBit() / 8;
        }
        enum class jout_options {
            as_f32 = 1,
            as_i32 = 2,
            as_hex8 = 3,
            as_i8 = 4,
            as_i16 = 5,
        };
        jout_options m_jout_opt;
        jout_options as_f32 = jout_options::as_f32;
        jout_options as_i32 = jout_options::as_i32;
        jout_options as_hex8 = jout_options::as_hex8;
        jout_options as_x8 = jout_options::as_hex8;
        jout_options as_i8 = jout_options::as_i8;
        jout_options as_i16 = jout_options::as_i16;
        void _jit_cout(jout_options value) {
            m_jout_opt = value;
        }
        static void printf_vec_float(uint32_t* v, int count) {
            printf("[");
            for (int i = count - 1; i >= 0; i--) {
                printf("%.5g", *reinterpret_cast<float*>(&v[i]));
                if (i > 0)
                    printf(",");
            }
            printf("]");
        }
        static void printf_vec_i32(uint32_t* v, int count) {
            printf("[");
            for (int i = count - 1; i >= 0; i--) {
                printf("%08x", v[i]);
                if (i > 0)
                    printf(",");
            }
            printf("]");
        }
        static void printf_vec_hex8(uint8_t* v, int count) {
            printf("[");
            for (int i = 0; i < count; i++) {
                printf("%02x ", v[i]);
            }
            printf("]");
        }
        static void printf_vec_i8(int8_t* v, int count) {
            printf("[");
            for (int i = 0; i < count; i++) {
                printf("%2d ", v[i]);
            }
            printf("]");
        }
        static void printf_vec_i16(int16_t* v, int count) {
            printf("[");
            for (int i = 0; i < count; i++) {
                printf("%4d  ", v[i]);
            }
            printf("]");
        }
        void _jit_cout(Xbyak::Xmm value) {
            auto rbp = jit->rbp;
            auto rsi = jit->rsi;
            auto rdi = jit->rdi;
            int rbp_disp = -1;
            for (int i = 0; i < preserved_vmms.size(); i++) {
                // printf(">>>>>>>>> [%d/%d] %d\n",i, preserved_vmms.size(), preserved_vmms[i].getIdx());
                if (value.getIdx() == preserved_vmms[i].getIdx()) {
                    rbp_disp = ((preserved_vmms.size() - 1) - i) * vmm_size_byte;
                    break;
                }
            }
            assert(rbp_disp >= 0);
            rbp_disp -= (preserved_regs.size() - 1) * 8 + preserved_vmms.size() * vmm_size_byte;

            jit->mov(rsi, 0);
            jit->lea(rdi, jit->ptr[rbp + rbp_disp]);
            jit->mov(jit->esi, value.getBit() / 32);
            if (m_jout_opt == jout_options::as_f32) {
                jit->call(reinterpret_cast<const void*>(printf_vec_float));
            } else if (m_jout_opt == jout_options::as_i32) {
                jit->call(reinterpret_cast<const void*>(printf_vec_i32));
            } else if (m_jout_opt == jout_options::as_hex8) {
                jit->mov(jit->esi, value.getBit() / 8);
                jit->call(reinterpret_cast<const void*>(printf_vec_hex8));
            } else if (m_jout_opt == jout_options::as_i8) {
                jit->mov(jit->esi, value.getBit() / 8);
                jit->call(reinterpret_cast<const void*>(printf_vec_i8));
            } else if (m_jout_opt == jout_options::as_i16) {
                jit->mov(jit->esi, value.getBit() / 16);
                jit->call(reinterpret_cast<const void*>(printf_vec_i16));
            }
        }
        void _jit_cout(Xbyak::Reg64 value) {
            const char* fmt_r64 = "0x%llx";
            // load reg from snapshot on the stack
            jit->mov(jit->rdi, reinterpret_cast<uintptr_t>(fmt_r64));
            bool found = false;
            for (int i = 0; i < preserved_regs.size(); i++) {
                if (value.getIdx() == preserved_regs[i].getIdx()) {
                    jit->mov(jit->rsi,
                             jit->ptr[jit->rbp + (preserved_regs.size() - i) * 8 - preserved_regs.size() * 8]);
                    found = true;
                    break;
                }
            }
            assert(found);
            jit->call(reinterpret_cast<const void*>(printf));
        }
        void _jit_cout(const char* value) {
            const char* fmt_cstr = "%s";
            jit->mov(jit->rdi, reinterpret_cast<uintptr_t>(fmt_cstr));
            jit->mov(jit->rsi, reinterpret_cast<uintptr_t>(value));
            jit->call(reinterpret_cast<const void*>(printf));
        }
        void _jit_cout(int64_t value) {
            const char* fmt = "%lld";
            jit->mov(jit->rdi, reinterpret_cast<uintptr_t>(fmt));
            jit->mov(jit->rsi, value);
            jit->call(reinterpret_cast<const void*>(printf));
        }
        template <typename... Args>
        void operator()(Args... args) {
            // setup frames
            // rbp, rsp, rax,
            auto rbp = jit->rbp;
            auto rsp = jit->rsp;
            auto r15 = jit->r15;
            jit->push(rbp);
            jit->mov(rbp, rsp);  // rbp points to preserved_regs, start with rbp, rsp's value is (rbp - 8)
            jit->add(rbp, 8);
            jit->push(rbp);  // this is the rsp before calling jit_cout
            jit->sub(rbp, 8);
            for (int i = 2; i < preserved_regs.size(); i++)
                jit->push(preserved_regs[i]);

            for (int i = 0; i < preserved_vmms.size(); i++) {
                auto& vmm = preserved_vmms[i];
                jit->vmovdqu(jit->ptr[rsp - (i + 1) * (vmm_size_byte)], vmm);
            }
            jit->sub(rsp, preserved_vmms.size() * vmm_size_byte);

            // align stack for calling printf
            jit->mov(r15, rsp);
            jit->and_(rsp, -16);

            // _jit_cout will not use r15
            int dummy[sizeof...(Args)] = {(_jit_cout(args), 0)...};
            (void)dummy;
            _jit_cout("\n");

            jit->mov(rsp, r15);
            jit->add(rsp, preserved_vmms.size() * preserved_vmms[0].getBit() / 8);
            for (int i = 0; i < preserved_vmms.size(); i++) {
                auto& vmm = preserved_vmms[i];
                jit->vmovdqu(vmm, jit->ptr[rsp - (i + 1) * (vmm_size_byte)]);
            }
            for (int i = preserved_regs.size() - 1; i >= 2; i--)
                jit->pop(preserved_regs[i]);
            jit->pop(rbp);
            jit->pop(rbp);
        }
    } jcout;

    uint32_t vreg_bits() {
        return use_avx512 ? 512 : 256;
    }
    jit_generator(const char* name)
        : m_kernel_name(name),
          sreg_pool("sreg_pool"),
          vreg_pool("vreg_pool"),
          treg_pool("treg_pool"),
          jcout(this) {
        vreg_pool.add_range(0, 15);
        if (use_avx512)
            vreg_pool.add_range(16, 31);
        treg_pool.add_range(0, 7);
#    ifdef _WIN32
        sreg_pool.add_range(std::vector<int>({Xbyak::Operand::RCX,
                                              Xbyak::Operand::RDX,
                                              Xbyak::Operand::R8,
                                              Xbyak::Operand::R9,

                                              // regs for local variables
                                              Xbyak::Operand::RDI,
                                              Xbyak::Operand::RSI,
                                              Xbyak::Operand::RBX,
                                              Xbyak::Operand::RAX,
                                              // Xbyak::Operand::RBP,
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
                                              Xbyak::Operand::RAX,
                                              // Xbyak::Operand::RBP,
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

    struct reg_preserve_status {
        bool is_preserved;
        int offset;
    };
    std::map<int, reg_preserve_status> regs_on_stack;
    int regs_preserve_stack_size;

    static const int num_abi_save_gpr_regs = sizeof(abi_save_gpr_regs) / sizeof(abi_save_gpr_regs[0]);

    void preamble() {
        // regs_on_stack[i]: if i'th register is preserved on stack
        push(rbp);
        mov(rbp, rsp);  // setup stack-frame
        regs_preserve_stack_size = 0;
        for (int i = 0; i < num_abi_save_gpr_regs; ++i) {
            auto reg_idx = static_cast<int>(abi_save_gpr_regs[i]);
            if (abi_save_gpr_regs[i] == Xbyak::Operand::RBP) {
                regs_on_stack[reg_idx] = reg_preserve_status{true, i * 8};
            } else {
                regs_on_stack[reg_idx] = reg_preserve_status{false, i * 8};
            }
            regs_preserve_stack_size += 8;  // increase space
        }
        // lazy preserve: reserve stack space only
        sub(rsp, regs_preserve_stack_size);
    }

    void uni_vzeroupper() {
        // if (mayiuse(avx))
        vzeroupper();
    }
    bool is_post_amble_called = false;
    void postamble() {
        is_post_amble_called = true;
        for (size_t i = 0; i < num_abi_save_gpr_regs; ++i) {
            auto reg_idx = static_cast<int>(abi_save_gpr_regs[i]);
            auto& status = regs_on_stack[reg_idx];
            if (abi_save_gpr_regs[i] != Xbyak::Operand::RBP && status.is_preserved) {
                mov(XbyakSReg64(reg_idx), ptr[rsp + status.offset]);
            }
        }
        add(rsp, regs_preserve_stack_size);
        pop(rbp);
        uni_vzeroupper();
        ret();
    }
    const void* jit_kernel_code = nullptr;
    const void* jit_ker() const {
        return jit_kernel_code;
    }

    void return_(int imm32) {
        mov(rax, imm32);
        postamble();
    }

    void return_(XbyakSReg64 return_value = XbyakSReg64(Xbyak::Operand::RAX)) {
        if (return_value.getIdx() != rax.getIdx())
            mov(rax, return_value);
        postamble();
    }

    int finalize() {
        if (!is_post_amble_called) {
            // throw std::runtime_error("postamble is not generated, besure return_() is called at least once.");
            return_();
        }
        int err_code = Xbyak::GetError();
        if (err_code != Xbyak::ERR_NONE)
            return err_code;
        jit_kernel_code = getCode();
        if (ov::intel_cpu::SIMDJIT_DEBUG > 10) {
            ov::intel_cpu::jit_dump_asm(m_kernel_name, jit_kernel_code, this->getSize());
        }
        return (jit_kernel_code) ? 0 : -1;
    }

protected:
    ov::intel_cpu::reg_pool sreg_pool;
    ov::intel_cpu::reg_pool vreg_pool;
    ov::intel_cpu::reg_pool treg_pool;

    std::shared_ptr<XbyakSReg64> alloc_reg64(int index) {
        auto reg_index = sreg_pool.allocate(index);

        auto it = regs_on_stack.find(reg_index);
        if (it != regs_on_stack.end()) {
            // lazy preserve sreg on stack according to ABI
            auto& status = it->second;
            if (!status.is_preserved) {
                status.is_preserved = true;
                mov(ptr[rsp + status.offset], XbyakSReg64(reg_index));
            }
        }

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

    std::shared_ptr<XbyakTReg> alloc_treg(int index) {
        auto reg_index = treg_pool.allocate(index);
        return std::shared_ptr<XbyakTReg>(new Xbyak::Tmm(reg_index), [this, reg_index](Xbyak::Tmm* preg) {
            treg_pool.free(reg_index);
            delete preg;
        });
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

    uint32_t vreg_bits() {
        return 128;
    }

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
        ready();
        jit_kernel_code = getCode();
        if (ov::intel_cpu::SIMDJIT_DEBUG > 10) {
            ov::intel_cpu::jit_dump_asm(m_kernel_name, jit_kernel_code, this->getSize());
        }
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

struct default_simd_jit_t {
    SIMDJit* cur;
    static default_simd_jit_t& get() {
        static default_simd_jit_t inst;
        return inst;
    }
};

class SReg {
private:
    SIMDJit* jit = nullptr;
    std::shared_ptr<XbyakSReg64> reg;

public:
    SReg(SIMDJit* jit, std::shared_ptr<XbyakSReg64> reg) : jit(jit), reg(reg) {}
    SReg();
    SReg(const SReg& rhs) = default;

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
    inline void load(SRegExpr&& addr) const;
    inline void store(SRegExpr&& addr) const;
    friend class SIMDJit;
    friend class SRegExpr;
};

class VReg {
private:
    SIMDJit* jit = nullptr;
    std::shared_ptr<XbyakVReg> reg;

public:
    VReg(SIMDJit* jit, std::shared_ptr<XbyakVReg> reg) : jit(jit), reg(reg) {}
    VReg();
    VReg(const VReg& rhs) = default;
    bool empty() const {
        return !static_cast<bool>(reg);
    }
    operator XbyakVReg&() {
        return *reg;
    }
    operator const XbyakVReg&() const {
        return *reg;
    }
    Xbyak::Ymm ymm() const {
        return Xbyak::Ymm(reg->getIdx());
    }
    Xbyak::Xmm xmm() const {
        return Xbyak::Xmm(reg->getIdx());
    }
    inline void load(const int32_t brdi32) const;
    inline void load(const float brdf32) const;
    inline void load(const void* pv) const;
    inline void load(const VReg& rhs) const;
    inline void load(SRegExpr&& addr) const;
    inline void store(SRegExpr&& addr) const;
};

// AMX Tile register
class TReg {
private:
    SIMDJit* jit = nullptr;
    std::shared_ptr<XbyakTReg> reg;

public:
    TReg(SIMDJit* jit, std::shared_ptr<XbyakTReg> reg) : jit(jit), reg(reg) {}
    TReg(int id = -1);
    TReg(const TReg& rhs) = default;
    bool empty() const {
        return !static_cast<bool>(reg);
    }
    operator XbyakTReg&() {
        return *reg;
    }
    operator const XbyakTReg&() const {
        return *reg;
    }
    inline void load(SRegExpr&& addr) const;
    inline void store(SRegExpr&& addr) const;
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
        Xbyak::RegExp to_addr() {
            if (base_reg < 0) {
                ASSERT(index_reg >= 0);
                return XbyakSReg64(index_reg) * scale + disp;
            } else if (index_reg >= 0)
                return XbyakSReg64(base_reg) + XbyakSReg64(index_reg) * scale + disp;
            else
                return XbyakSReg64(base_reg) + disp;
        }
        std::string to_string() {
            std::stringstream ss;
            ss << "[";
            if (base_reg >= 0)
                ss << "{r" << base_reg << "}";
            else
                ss << "{}";

            if (index_reg >= 0) {
                ss << " + {r" << index_reg << "} x " << scale;
            }
            ss << " + " << disp << "]";
            return ss.str();
        }
    };
    std::unique_ptr<Addressing> paddr;

    SRegExpr(int data) : pimpl(new RegExprImpl("i", data)) {}

    SRegExpr(SReg r) : pimpl(new RegExprImpl("r", r.r64().getIdx())) {
        paddr.reset(new Addressing(r.r64().getIdx(), -1, 1, 0));
    }

    SRegExpr(const char* type, int data) : pimpl(new RegExprImpl(type, data)) {}
    SRegExpr(const char* op, SRegExpr&& lhs) : pimpl(new RegExprImpl(op, lhs.pimpl)) {}
    SRegExpr(const char* op, SRegExpr&& lhs, SRegExpr&& rhs) : pimpl(new RegExprImpl(op, lhs.pimpl, rhs.pimpl)) {
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
            // std::cout << "pimpl->is_swapped = " << pimpl->is_swapped << std::endl;
            // std::cout << "lhs.paddr = " << (lhs.paddr ? lhs.paddr->to_string() : "N/A") << std::endl;
            // std::cout << "rhs.paddr = " << (rhs.paddr ? rhs.paddr->to_string() : "N/A") << std::endl;

            if (pimpl->is_swapped) {
                paddr = std::move(rhs.paddr);
            } else {
                paddr = std::move(lhs.paddr);
            }
            if (paddr) {
                // merge addressing mode: only (+base) or (+disp) is allowed
                // std::cout << "paddr = " << paddr->to_string() << std::endl;
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
        // show("SRegExpr constructed:");
    }

    void show(std::string title) const {
        std::cout << "\033[32m::::" << title << "::::\033[0m" << std::endl;
        if (paddr) {
            std::cout << "\tAddressing:" << paddr->to_string() << std::endl;
        }
        if (pimpl)
            pimpl->for_each_op([&](RegExprImpl* p) {
                std::cout << "\t" << p->name() << " = " << p->lhs->name() << " " << p->op << " "
                          << (p->rhs ? p->rhs->name() : std::string("( )"));
#ifdef __aarch64__
                if (p->shift_type != RegExprImpl::SHIFT_TYPE::NONE) {
                    auto shift_amount = static_cast<int>(p->shift_amount);
                    switch (p->shift_type) {
                    case RegExprImpl::SHIFT_TYPE::ASR:
                        std::cout << " >>(A) " << shift_amount;
                        break;
                    case RegExprImpl::SHIFT_TYPE::LSR:
                        std::cout << " >>(L) " << shift_amount;
                        break;
                    case RegExprImpl::SHIFT_TYPE::LSL:
                        std::cout << " <<(L) " << shift_amount;
                        break;
                    case RegExprImpl::SHIFT_TYPE::ROR:
                        std::cout << " >>(R) " << shift_amount;
                        break;
                    }
                }
#endif
                std::cout << std::endl;
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
inline SRegExpr operator~(SRegExpr&& lhs) {
    return SRegExpr("~", std::move(lhs));
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
    ASSERT(lhs.pimpl->is_logical_op());
    ASSERT(rhs.pimpl->is_logical_op());
    return SRegExpr("&&", std::move(lhs), std::move(rhs));
}
inline SRegExpr operator||(SRegExpr&& lhs, SRegExpr&& rhs) {
    ASSERT(lhs.pimpl->is_logical_op());
    ASSERT(rhs.pimpl->is_logical_op());
    return SRegExpr("||", std::move(lhs), std::move(rhs));
}
inline SRegExpr operator!(SRegExpr&& lhs) {
    ASSERT(lhs.pimpl->is_logical_op());
    return SRegExpr("!", std::move(lhs));
}

template <typename T>
struct convert_call_arg {
    using type = T;
};
template <>
struct convert_call_arg<int16_t> {
    using type = int64_t;
};
template <>
struct convert_call_arg<uint16_t> {
    using type = int64_t;
};
template <>
struct convert_call_arg<int32_t> {
    using type = int64_t;
};
template <>
struct convert_call_arg<uint32_t> {
    using type = int64_t;
};

// copied from pybind11
template <typename T>
struct remove_class {};
template <typename C, typename R, typename... A>
struct remove_class<R (C::*)(A...)> {
    using type = R(A...);
};
template <typename C, typename R, typename... A>
struct remove_class<R (C::*)(A...) const> {
    using type = R(A...);
};

enum class TMUL_TYPE { SSD = 1, USD = 2, SUD = 3, UUD = 4, FP16 = 5, BF16 = 6 };

//=========================================================================================
class SIMDJit : public jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(SIMDJit);

    template <typename Func, typename Return, typename... Args>
    static std::shared_ptr<SIMDJit> _create(Func&& f, Return (*)(SIMDJit*, Args...)) {
        auto jit = std::make_shared<SIMDJit>();
        default_simd_jit_t::get().cur = jit.get();
        // https://stackoverflow.com/questions/68882421/using-a-pack-expansion-with-an-index-is-it-ub
        auto convertedArgs = std::tuple{jit.get(), static_cast<Args>(jit->get_arg())...};
        std::apply(f, convertedArgs);
        jit->finalize();
        default_simd_jit_t::get().cur = nullptr;
        return jit;
    }

    template <typename Func>
    static std::shared_ptr<SIMDJit> create(Func&& f) {
        using fsig = typename remove_class<decltype(&Func::operator())>::type;
        return _create(f, (fsig*)(nullptr));
    }

    std::shared_ptr<void> get_disasm(int enable) {
        if (enable) {
            auto start = getSize();
            return std::shared_ptr<void>(nullptr, [start, this](void*) {
                jit_dump_asm("", this->getCode() + start, this->getSize() - start);
            });
        }
        return nullptr;
    }

    SIMDJit(const char* name = "") : jit_generator(name) {
        preamble();
    }

    template <typename... kernel_args_t>
    int64_t operator()(kernel_args_t... args) const {
        // all integer value types are converted into int64_t
        using jit_kernel_func_t = int64_t (*)(const typename convert_call_arg<kernel_args_t>::type...);
        auto* fptr = (jit_kernel_func_t)jit_ker();
        // return (*fptr)(std::forward<kernel_args_t>(args)...);
        return (*fptr)(args...);
    }

    int m_arg_id = 0;
    SReg get_arg(int idx = -1) {
        if (idx < 0)
            idx = m_arg_id++;

        auto ret = SReg(this, alloc_reg64(idx));
#ifdef __x86_64__
        // https://en.wikipedia.org/wiki/X86_calling_conventions#x86-64_calling_conventions
        if (idx >= abi_param_regs_num)
            mov(ret, ptr[rbp + (idx - abi_param_regs_num + 2) * 8]);  // load from stack
#endif
#ifdef __aarch64__
        // https://en.wikipedia.org/wiki/Calling_convention#ARM_(A64)
        ASSERT(idx < 8);
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

    TReg get_treg(int id) {
        return TReg(this, alloc_treg(id));
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
    void tmul(const XbyakTReg& x1, const XbyakTReg& x2, const XbyakTReg& x3, TMUL_TYPE type) {
        switch (type) {
        case TMUL_TYPE::SSD:
            tdpbssd(x1, x2, x3);
            break;
        case TMUL_TYPE::USD:
            tdpbusd(x1, x2, x3);
            break;
        case TMUL_TYPE::SUD:
            tdpbsud(x1, x2, x3);
            break;
        case TMUL_TYPE::UUD:
            tdpbuud(x1, x2, x3);
            break;
        case TMUL_TYPE::FP16:
            tdpfp16ps(x1, x2, x3);
            break;
        case TMUL_TYPE::BF16:
            tdpbf16ps(x1, x2, x3);
            break;
        }
    }
#endif

    //***********************************************
    // for_loop(idx, start, stop, step, loop_body) performs following:
    //    for(int idx=start; idx + step <= stop; idx+=step) {
    //       loop_body();
    //    }
    template <typename Fn, typename START, typename STEP>
    void for_loop(XbyakSReg64 idx, START start, XbyakSReg64 stop, STEP step, const Fn& loop_body);

    template <typename Fn>
    void while_(SRegExpr regcmp, const Fn& loop_body);

    template <typename Fn>
    void do_while_(SRegExpr regcmp, const Fn& loop_body);

    inline void if_(SRegExpr regcmp,
                    const std::function<void()>& then_body,
                    const std::function<void()>& else_body = {});

    struct ExprStatus {
        int scratch_reg_cnt;
        int ops_cnt;
        ExprStatus() = default;
        ExprStatus(int scratch_reg_cnt, int ops_cnt) : scratch_reg_cnt(scratch_reg_cnt), ops_cnt(ops_cnt) {}
    } expr_stat;
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

inline SReg::SReg() {
    auto* cur_jit = default_simd_jit_t::get().cur;
    if (cur_jit) {
        // allocate sreg
        jit = cur_jit;
        reg = cur_jit->get_sreg().reg;
    }
}
inline VReg::VReg() {
    auto* cur_jit = default_simd_jit_t::get().cur;
    if (cur_jit) {
        // allocate sreg
        jit = cur_jit;
        reg = cur_jit->get_vreg().reg;
    }
}
inline TReg::TReg(int id) {
    auto* cur_jit = default_simd_jit_t::get().cur;
    if (cur_jit) {
        // allocate sreg
        jit = cur_jit;
        reg = cur_jit->get_treg(id).reg;
    }
}
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
        if (pdst) {
            std::cout << " assign-to : r" << pdst->r64().getIdx() << assign_op << " = ..." << std::endl;
        }
        pimpl->show_rpn();
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
                ASSERT(false);
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
                ASSERT(false);
                break;
            }
            return;
        }
        // addressing expression
        if (paddr) {
            auto to_RegExp = [&] {
                ASSERT(paddr);

                if (paddr->base_reg < 0) {
                    ASSERT(paddr->index_reg >= 0);
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
                    ASSERT(false);
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

        if (p->rhs && !p->rhs->is_leaf() && p->is_op("-")) {
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
        if (p->rhs && !p->rhs->is_leaf())
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
        ASSERT(assign_dst_reg_scratch_sn >= scratch_reg_base);
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

    expr_stat.scratch_reg_cnt = scratch_regs.size();
    expr_stat.ops_cnt = 0;

    // emmit code
    pimpl->for_each_op([&](RegExprImpl* p) {
        auto dst = XbyakSReg64(p->data);
        expr_stat.ops_cnt++;
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
                ASSERT(false);  // sar(dst, p->rhs->as_r64());
            }
        } else if (p->is_op("<<")) {
            if (p->rhs->is_imm())
                shl(dst, p->rhs->as_imm32());
            else {
                // only cl register supportted, we need allocate cl
                ASSERT(false);  // shl(dst, p->rhs->as_r64());
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
        } else if (p->is_op("~")) {
            not_(dst);
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
            ASSERT(0, "Unsupported OP: ", p->op);
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
                ASSERT(false);
                break;
            }
        }
    }

    if (debug_log) {
        std::cout << "expr statistics :  scratch_reg_cnt = " << expr_stat.scratch_reg_cnt
                  << ", ops_cnt = " << expr_stat.ops_cnt << std::endl;
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

inline void VReg::load(const int32_t brdi32) const {
    if (brdi32 == 0) {
        jit->vpxor(*reg, *reg, *reg);
    } else {
        auto s = jit->get_sreg();
        jit->mov(s, brdi32);
        jit->vmovd(Xbyak::Xmm(reg->getIdx()), s.r64().cvt32());
        jit->vpbroadcastd(*reg, Xbyak::Xmm(reg->getIdx()));
    }
}

inline void VReg::load(const float brdf32) const {
    if (brdf32 == 0) {
        jit->vpxor(*reg, *reg, *reg);
    } else {
        auto s = jit->get_sreg();
        jit->mov(s, reinterpret_cast<const uint32_t&>(brdf32));
        jit->vmovd(Xbyak::Xmm(reg->getIdx()), s.r64().cvt32());
        jit->vbroadcastss(*reg, Xbyak::Xmm(reg->getIdx()));
    }
}

inline void VReg::load(const void* pv) const {
    if (pv == nullptr) {
        jit->vpxor(*reg, *reg, *reg);
    } else {
        auto s = jit->get_sreg();
        jit->mov(s, reinterpret_cast<uintptr_t>(pv));
        jit->vmovdqu(*reg, jit->ptr[s.r64()]);
    }
}

inline void VReg::load(const VReg& rhs) const {
    jit->vmovdqa(*reg, rhs);
}

inline void VReg::load(SRegExpr&& addr) const {
    ASSERT(addr.paddr);
    jit->vmovdqu(*reg, jit->ptr[addr.paddr->to_addr()]);
}
inline void VReg::store(SRegExpr&& addr) const {
    ASSERT(addr.paddr);
    jit->vmovdqu(jit->ptr[addr.paddr->to_addr()], *reg);
}
inline void SReg::load(SRegExpr&& addr) const {
    ASSERT(addr.paddr);
    jit->mov(*reg, jit->ptr[addr.paddr->to_addr()]);
}
inline void SReg::store(SRegExpr&& addr) const {
    ASSERT(addr.paddr);
    jit->mov(jit->ptr[addr.paddr->to_addr()], *reg);
}
inline void TReg::load(SRegExpr&& addr) const {
    ASSERT(addr.paddr);
    jit->tileloadd(*reg, jit->ptr[addr.paddr->to_addr()]);
}
inline void TReg::store(SRegExpr&& addr) const {
    ASSERT(addr.paddr);
    jit->tilestored(jit->ptr[addr.paddr->to_addr()], *reg);
}
#endif
// https://courses.cs.washington.edu/courses/cse469/19wi/arm64.pdf
//
#ifdef __aarch64__

template <typename Fn, typename START, typename STEP>
void SIMDJit::for_loop(XbyakSReg64 idx, START start, XbyakSReg64 stop, STEP step, const Fn& loop_body) {
    Xbyak_aarch64::Label loop, exit;
    mov(idx, start);

    align(64);
    L(loop);
    add(idx, idx, step);
    cmp(idx, stop);
    b(Xbyak_aarch64::Cond::GT, exit);
    sub(idx, idx, step);

    loop_body();
    add(idx, idx, step);

    b(loop);
    L(exit);
    // at exit, idx is pointing to tail
    sub(idx, idx, step);
}

template <typename Fn>
void SIMDJit::while_(SRegExpr regcmp, const Fn& loop_body) {
    Xbyak_aarch64::Label loop, exit;

    align(64);
    L(loop);

    evaluate(regcmp, nullptr, 'F', exit);

    loop_body();

    b(loop);
    L(exit);
}

template <typename Fn>
void SIMDJit::do_while_(SRegExpr regcmp, const Fn& loop_body) {
    Xbyak_aarch64::Label loop;

    align(64);
    L(loop);

    loop_body();

    evaluate(regcmp, nullptr, 'T', loop);
}

inline void SIMDJit::if_(SRegExpr regcmp,
                         const std::function<void()>& then_body,
                         const std::function<void()>& else_body) {
    Xbyak_aarch64::Label if_else, if_exit;

    evaluate(regcmp, nullptr, 'F', if_else);

    then_body();

    if (else_body)
        b(if_exit);

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
        if (pdst) {
            std::cout << "r" << pdst->r64().getIdx() << " " << assign_op << (assign_op == '=' ? " " : "= ")
                      << std::endl;
        }
        pimpl->show_rpn();
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
                ASSERT(false);
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
            mov(imm_reg, imm32);

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
                ASSERT(false);
                break;
            }
            return;
        }
    }

    // complex expression: need multiple passes on IR to work
    // assign scratch register & convert to 2-OP instruction form
    pimpl->const_folding();
    if (debug_log)
        expr.show(" After const folding");

    // replace special useless op
    pimpl->for_each_op([&](RegExprImpl* p) {
        if (p->rhs && p->rhs->is_imm()) {
            auto imm32 = p->rhs->as_imm32();
            if (imm32 == 0 && p->is_op("&")) {
                // & 0 ===> = 0
                p->op = " ";
                p->lhs = std::move(p->rhs);
                return true;
            }
            if (imm32 == 0 && (p->is_op("|") || p->is_op("^") || p->is_op("*"))) {
                p->op = "NOP";  // "NOP" : pass-through (dst = lhs)
            }
        }
        if (p->lhs->is_op("NOP")) {
            p->lhs = std::move(p->lhs->lhs);
        }
        if (p->rhs && p->rhs->is_op("NOP")) {
            p->lhs = std::move(p->rhs->lhs);
        }
    });

    // using shifted register when possible
    pimpl->for_each_op([&](RegExprImpl* p) {
        // dst = lhs op (rhs << 3)
        // dst = lhs op (rhs >> 3)
        // dst = lhs op (rhs * 2^n)
        //
        if (!p->rhs)
            return true;
        if (p->is_op("+") || p->is_op("-") || p->is_op("&") || p->is_op("|") || p->is_op("^")) {
            if (p->rhs->is_reg() && p->lhs->is_op() && (!p->is_op("-"))) {
                // most likely optimization happens on lhs
                std::swap(p->lhs, p->rhs);
            }
            if (!p->rhs->rhs)
                return true;
            if (!p->rhs->lhs->is_reg() && !p->rhs->lhs->is_op())
                return true;
            if (!p->rhs->rhs->is_imm())
                return true;
            auto imm32 = p->rhs->rhs->as_imm32();
            if (imm32 < 0)
                return true;

            RegExprImpl::SHIFT_TYPE shift_type = RegExprImpl::SHIFT_TYPE::NONE;
            if (p->rhs->is_op(">>") || p->rhs->is_op("<<")) {
                ASSERT(imm32 >= 0 && imm32 < 64);
                if (p->rhs->is_op(">>"))
                    p->shift_type = RegExprImpl::SHIFT_TYPE::ASR;
                if (p->rhs->is_op("<<"))
                    p->shift_type = RegExprImpl::SHIFT_TYPE::LSL;
                p->rhs = std::move(p->rhs->lhs);
                p->shift_amount = imm32;
                return true;
            }

            if (p->rhs->is_op("*")) {
                p->shift_amount = 0;
                while ((imm32 & 1) == 0) {
                    imm32 = imm32 >> 1;
                    p->shift_amount++;
                }
                if (imm32 == 1) {
                    p->shift_type = RegExprImpl::SHIFT_TYPE::LSL;
                    p->rhs = std::move(p->rhs->lhs);
                    return true;
                }
                // this removes some bits from imm32 : (rhs*imm32)
                if (p->shift_amount > 0) {
                    p->shift_type = RegExprImpl::SHIFT_TYPE::LSL;
                    p->rhs->rhs->data = imm32;
                    return true;
                }
            }
        }
        return true;
    });
    if (debug_log)
        expr.show(" After fusing shift");

    // allocate scratch register
    // ARM instruction is 3-OP, so both lhs & rhs can be reused
    // but only rhs in ADD & SUB can be imm, and it can be 0-4095 only
    static reg_pool scratch_reg_sn_pool("sreg_expr_scratch_registers", 32);
    scratch_reg_sn_pool.clear();
    auto scratch_reg_base = 1000;
    auto alloc_scratch_reg = [&]() {
        return scratch_reg_sn_pool.allocate() + scratch_reg_base;
    };
    auto free_scratch_reg = [&](int reg) {
        scratch_reg_sn_pool.free(reg - scratch_reg_base);
    };

    pimpl->for_each_op([&](RegExprImpl* p) {
        if (p->lhs->is_imm()) {
            // insert mov imm (since only rhs can be imm)
            std::unique_ptr<RegExprImpl> pmov(new RegExprImpl(" ", p->lhs));
            pmov->data = alloc_scratch_reg();
            p->lhs = std::move(pmov);
        }

        // some op (~, !, -) has no rhs
        if (p->rhs && p->rhs->is_imm()) {
            auto imm32 = p->rhs->as_imm32();
            bool is_op_support_imm = p->is_op("+") || p->is_op("-") || p->is_op(">>") || p->is_op("<<") || p->is_cmp();
            if (p->is_op("&") || p->is_op("|") || p->is_op("^")) {
                // check if imm32 is Bitmask immediates
                // https://kddnewton.com/2022/08/11/aarch64-bitmask-immediates.html
                // here for simplicity we only check 64-bits case
                uint32_t temp_imm = reinterpret_cast<uint32_t&>(imm32);
                int num_bit_switch = 0;
                uint32_t last_bit = temp_imm & 1;
                for (int i = 1; i < 32; i++) {
                    uint32_t cur_bit = temp_imm & (1 << i);
                    if (last_bit != cur_bit)
                        num_bit_switch++;
                }
                ASSERT(num_bit_switch > 0);
                if (num_bit_switch == 1 || num_bit_switch == 2) {
                    is_op_support_imm = true;
                }
            }
            if (imm32 < 0 || imm32 > 4095 || (!is_op_support_imm)) {
                auto rhs_temp_reg = alloc_scratch_reg();
                std::unique_ptr<RegExprImpl> pmov(new RegExprImpl(" ", p->rhs));
                pmov->data = rhs_temp_reg;
                p->rhs = std::move(pmov);
            }
        }

        // reuse lhs temp reg
        if (!p->lhs->is_leaf()) {
            p->data = p->lhs->data;
            if (p->rhs && !p->rhs->is_leaf())
                free_scratch_reg(p->rhs->data);
            return true;
        }
        // reuse rhs temp reg
        if (p->rhs && !p->rhs->is_leaf()) {
            //   rhs = lhs + rhs
            p->data = p->rhs->data;
            return true;
        }

        auto new_scratch_reg_sn = alloc_scratch_reg();
        p->data = new_scratch_reg_sn;
        return true;
    });

    if (debug_log)
        expr.show(" After scratch reg allocation & convert to 2-OP form");

    // try to replace last scratch register with assign destination register
    bool dst_register_assigned_inplace = false;
    if (pdst && assign_op == '=') {
        dst_register_assigned_inplace = pimpl->replace_scratch_with_dst(pdst->r64().getIdx());
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

    auto shift_mode = [](RegExprImpl::SHIFT_TYPE sh) {
        switch (sh) {
        case RegExprImpl::SHIFT_TYPE::LSL:
            return Xbyak_aarch64::ShMod::LSL;
        case RegExprImpl::SHIFT_TYPE::LSR:
            return Xbyak_aarch64::ShMod::LSR;
        case RegExprImpl::SHIFT_TYPE::ASR:
            return Xbyak_aarch64::ShMod::ASR;
        case RegExprImpl::SHIFT_TYPE::ROR:
            return Xbyak_aarch64::ShMod::ROR;
        }
        return Xbyak_aarch64::ShMod::NONE;
    };

    expr_stat.scratch_reg_cnt = scratch_regs.size();
    expr_stat.ops_cnt = 0;

    // emmit code
    pimpl->for_each_op([&](RegExprImpl* p) {
        auto dst = XbyakSReg64(p->data);
        expr_stat.ops_cnt++;
        if (p->is_op(" ")) {
            ASSERT(p->lhs->is_imm());
            mov(dst, p->lhs->as_imm32());
        } else if (p->is_op("+")) {
            if (p->rhs->is_imm())
                add(dst, p->lhs->as_r64<XbyakSReg64>(), p->rhs->as_imm32());
            else
                add(dst,
                    p->lhs->as_r64<XbyakSReg64>(),
                    p->rhs->as_r64<XbyakSReg64>(),
                    shift_mode(p->shift_type),
                    p->shift_amount);
        } else if (p->is_op("-")) {
            if (p->rhs->is_imm())
                sub(dst, p->lhs->as_r64<XbyakSReg64>(), p->rhs->as_imm32());
            else
                sub(dst,
                    p->lhs->as_r64<XbyakSReg64>(),
                    p->rhs->as_r64<XbyakSReg64>(),
                    shift_mode(p->shift_type),
                    p->shift_amount);
        } else if (p->is_op("*")) {
            mul(dst, p->lhs->as_r64<XbyakSReg64>(), p->rhs->as_r64<XbyakSReg64>());
        } else if (p->is_op(">>")) {
            if (p->rhs->is_imm())
                asr(dst, p->lhs->as_r64<XbyakSReg64>(), p->rhs->as_imm32());
            else {
                asr(dst, p->lhs->as_r64<XbyakSReg64>(), p->rhs->as_r64<XbyakSReg64>());
            }
        } else if (p->is_op("<<")) {
            if (p->rhs->is_imm())
                lsl(dst, p->lhs->as_r64<XbyakSReg64>(), p->rhs->as_imm32());
            else {
                lsl(dst, p->lhs->as_r64<XbyakSReg64>(), p->rhs->as_r64<XbyakSReg64>());
            }
        } else if (p->is_op("&")) {
            if (p->rhs->is_imm())
                and_(dst, p->lhs->as_r64<XbyakSReg64>(), p->rhs->as_imm32());
            else if (p->shift_amount)
                and_(dst,
                     p->lhs->as_r64<XbyakSReg64>(),
                     p->rhs->as_r64<XbyakSReg64>(),
                     shift_mode(p->shift_type),
                     p->shift_amount);
            else
                and_(dst, p->lhs->as_r64<XbyakSReg64>(), p->rhs->as_r64<XbyakSReg64>());
        } else if (p->is_op("&&")) {
            if (p->rhs->is_imm())
                and_(dst, p->lhs->as_r64<XbyakSReg64>(), p->rhs->as_imm32() ? 1 : 0);
            else if (p->shift_amount)
                and_(dst,
                     p->lhs->as_r64<XbyakSReg64>(),
                     p->rhs->as_r64<XbyakSReg64>(),
                     shift_mode(p->shift_type),
                     p->shift_amount);
            else
                and_(dst, p->lhs->as_r64<XbyakSReg64>(), p->rhs->as_r64<XbyakSReg64>());
        } else if (p->is_op("|")) {
            if (p->rhs->is_imm())
                orr(dst, p->lhs->as_r64<XbyakSReg64>(), p->rhs->as_imm32());
            else if (p->shift_amount)
                orr(dst,
                    p->lhs->as_r64<XbyakSReg64>(),
                    p->rhs->as_r64<XbyakSReg64>(),
                    shift_mode(p->shift_type),
                    p->shift_amount);
            else
                orr(dst, p->lhs->as_r64<XbyakSReg64>(), p->rhs->as_r64<XbyakSReg64>());
        } else if (p->is_op("||")) {
            if (p->rhs->is_imm())
                orr(dst, p->lhs->as_r64<XbyakSReg64>(), p->rhs->as_imm32() ? 1 : 0);
            else if (p->shift_amount)
                orr(dst,
                    p->lhs->as_r64<XbyakSReg64>(),
                    p->rhs->as_r64<XbyakSReg64>(),
                    shift_mode(p->shift_type),
                    p->shift_amount);
            else
                orr(dst, p->lhs->as_r64<XbyakSReg64>(), p->rhs->as_r64<XbyakSReg64>());
        } else if (p->is_op("^")) {
            if (p->rhs->is_imm())
                eor(dst, p->lhs->as_r64<XbyakSReg64>(), p->rhs->as_imm32());
            else if (p->shift_amount)
                eor(dst,
                    p->lhs->as_r64<XbyakSReg64>(),
                    p->rhs->as_r64<XbyakSReg64>(),
                    shift_mode(p->shift_type),
                    p->shift_amount);
            else
                eor(dst, p->lhs->as_r64<XbyakSReg64>(), p->rhs->as_r64<XbyakSReg64>());
        } else if (p->is_op("~")) {
            mvn(dst, p->lhs->as_r64<XbyakSReg64>());
        } else if (p->is_op("!")) {
            eor(dst, p->lhs->as_r64<XbyakSReg64>(), 1);
        } else if (p->is_cmp()) {
            if (p->rhs->is_imm())
                cmp(p->lhs->as_r64<XbyakSReg64>(), p->rhs->as_imm32());
            else
                cmp(p->lhs->as_r64<XbyakSReg64>(), p->rhs->as_r64<XbyakSReg64>());
            if (!(do_jump && p == pimpl)) {
                if (p->is_op("=="))
                    cset(dst, Xbyak_aarch64::Cond::EQ);
                else if (p->is_op("!="))
                    cset(dst, Xbyak_aarch64::Cond::NE);
                else if (p->is_op(">"))
                    cset(dst, Xbyak_aarch64::Cond::GT);
                else if (p->is_op(">="))
                    cset(dst, Xbyak_aarch64::Cond::GE);
                else if (p->is_op("<"))
                    cset(dst, Xbyak_aarch64::Cond::LT);
                else if (p->is_op("<="))
                    cset(dst, Xbyak_aarch64::Cond::LE);
                else
                    ASSERT(false);
            }
        } else {
            ASSERT(false, "Unsupported OP: ", p->op);
        }
        return true;
    });

    if (debug_log) {
        std::cout << "expr statistics :  scratch_reg_cnt = " << expr_stat.scratch_reg_cnt
                  << ", ops_cnt = " << expr_stat.ops_cnt << std::endl;
    }

    if (pdst) {
        if (assign_op == '=' && !dst_register_assigned_inplace) {
            mov(*pdst, pimpl->as_r64<XbyakSReg64>());
        } else {
            switch (assign_op) {
            case '=':
                break;
            case '+':
                add(*pdst, *pdst, pimpl->as_r64<XbyakSReg64>());
                break;
            case '-':
                sub(*pdst, *pdst, pimpl->as_r64<XbyakSReg64>());
                break;
            case '*':
                mul(*pdst, *pdst, pimpl->as_r64<XbyakSReg64>());
                break;
            default:
                ASSERT(false);
                break;
            }
        }
    }

    // generate jump
    if (assign_op == 'T') {
        if (pimpl->is_cmp()) {
            if (pimpl->is_op("=="))
                b(Xbyak_aarch64::Cond::EQ, label);
            if (pimpl->is_op("!="))
                b(Xbyak_aarch64::Cond::NE, label);
            if (pimpl->is_op(">"))
                b(Xbyak_aarch64::Cond::GT, label);
            if (pimpl->is_op(">="))
                b(Xbyak_aarch64::Cond::GE, label);
            if (pimpl->is_op("<"))
                b(Xbyak_aarch64::Cond::LT, label);
            if (pimpl->is_op("<="))
                b(Xbyak_aarch64::Cond::LE, label);
        } else {
            // convert final value to ZF
            cmp(pimpl->as_r64<XbyakSReg64>(), 0);
            b(Xbyak_aarch64::Cond::NE, label);
        }
    } else if (assign_op == 'F') {
        if (pimpl->is_cmp()) {
            if (pimpl->is_op("=="))
                b(Xbyak_aarch64::Cond::NE, label);
            if (pimpl->is_op("!="))
                b(Xbyak_aarch64::Cond::EQ, label);
            if (pimpl->is_op(">"))
                b(Xbyak_aarch64::Cond::LE, label);
            if (pimpl->is_op(">="))
                b(Xbyak_aarch64::Cond::LT, label);
            if (pimpl->is_op("<"))
                b(Xbyak_aarch64::Cond::GE, label);
            if (pimpl->is_op("<="))
                b(Xbyak_aarch64::Cond::GT, label);
        } else {
            // convert final value to ZF
            cmp(pimpl->as_r64<XbyakSReg64>(), 0);
            b(Xbyak_aarch64::Cond::EQ, label);
        }
    }
}

#endif
}  // namespace intel_cpu
}  // namespace ov
