
#pragma once

#include "../thirdparty/xbyak/xbyak/xbyak.h"

#include <cstdlib>
#include <fstream>
#include <vector>
#include <cstring>
#include <cstddef>

#ifdef XBYAK64
constexpr Xbyak::Operand::Code abi_save_gpr_regs[] = {
    Xbyak::Operand::RBP, Xbyak::Operand::RBX, Xbyak::Operand::R12,
    Xbyak::Operand::R13, Xbyak::Operand::R14, Xbyak::Operand::R15,
#ifdef _WIN32
    Xbyak::Operand::RDI, Xbyak::Operand::RSI,
#endif
};

constexpr Xbyak::Operand::Code abi_param_regs[] = {
#ifdef _WIN32
    Xbyak::Operand::RCX, Xbyak::Operand::RDX, Xbyak::Operand::R8,
    Xbyak::Operand::R9
#else
    Xbyak::Operand::RDI,
    Xbyak::Operand::RSI,
    Xbyak::Operand::RDX,
    Xbyak::Operand::RCX,
    Xbyak::Operand::R8,
    Xbyak::Operand::R9
#endif
};

constexpr Xbyak::Operand::Code abi_not_param_reg =
#ifdef _WIN32
    Xbyak::Operand::RDI;
#else
    Xbyak::Operand::RCX;
#endif

#define abi_param1 Xbyak::Reg64(abi_param_regs[0])
#define abi_param2 Xbyak::Reg64(abi_param_regs[1])
#define abi_param3 Xbyak::Reg64(abi_param_regs[2])
#define abi_param4 Xbyak::Reg64(abi_param_regs[3])
#define abi_param5 Xbyak::Reg64(abi_param_regs[4])
#define abi_param6 Xbyak::Reg64(abi_param_regs[5])
#define abi_not_param1 Xbyak::Reg64(abi_not_param_reg)
#endif

// https://en.wikipedia.org/wiki/X86_calling_conventions#Microsoft_x64_calling_convention
//      The registers RAX, RCX, RDX, R8, R9, R10, R11 are considered volatile
//      (caller-saved).[25] The registers RBX, RBP, RDI, RSI, RSP, R12-R15 are
//      considered nonvolatile (callee-saved).[25]
// https://en.wikipedia.org/wiki/X86_calling_conventions#System_V_AMD64_ABI
//      If the callee wishes to use registers RBX, RSP, RBP, and R12–R15, it
//      must restore their original values before returning control to the
//      caller. so register can be used w/o saving/restoring:
//            abi_param1 ~ abi_param6
//            RAX, R10, R11
//
class jit_generator : public Xbyak::CodeGenerator {
 public:
  static std::string& jit_debug() {
    static std::string _jdbg = [](){
        auto* p_jit_debug = std::getenv("JIT_DEBUG");
        if (p_jit_debug == nullptr)
            return std::string{};
        return std::string(p_jit_debug);
    }();
    return _jdbg;
  }

  jit_generator()
      : Xbyak::CodeGenerator(Xbyak::DEFAULT_MAX_CODE_SIZE * 4, (void*)0) {}

  const char * name() {
    return ker_name;
  }
 protected:
  const size_t num_abi_save_gpr_regs =
      sizeof(abi_save_gpr_regs) / sizeof(abi_save_gpr_regs[0]);

  void preamble() {
    for (size_t i = 0; i < num_abi_save_gpr_regs; ++i) {
      push(Xbyak::Reg64(abi_save_gpr_regs[i]));
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
      pop(Xbyak::Reg64(abi_save_gpr_regs[num_abi_save_gpr_regs - 1 - i]));
    uni_vzeroupper();
    ret();
  }

  virtual void generate() { ret(); }

  const Xbyak::uint8* jit_ker_ = nullptr;
  const char* ker_name = "?";
  virtual int create_kernel(const char* name = "?") {
    int err_code = Xbyak::GetError();
    if (err_code != Xbyak::ERR_NONE)
      return err_code;
    generate();
    ker_name = name;
#ifdef JIT_DEBUG
    if (!jit_debug().empty()) {
      std::cout << "jit_generator generate() is done: " << name << std::endl;
      if (jit_debug() == name || jit_debug() == "*") {
        dump();
      }
    }
#endif
    jit_ker_ = getCode();
    return (jit_ker_) ? 0 : -1;
  }

 public:
  template <typename... kernel_args_t>
  int operator()(kernel_args_t... args) const {
    using jit_kernel_func_t = int (*)(const kernel_args_t... args);
    auto* fptr = (jit_kernel_func_t)jit_ker_;
#ifdef JIT_DEBUG
    if (!jit_debug().empty()) {
      if (jit_debug() == ker_name || jit_debug() == "*") {
        std::cout << "jit kernel " << ker_name << " @ 0x" << std::hex
                  << reinterpret_cast<uintptr_t>(jit_ker_)
                  << " is being called.\n";
        asm("int3");
      }
    }
#endif
    return (*fptr)(std::forward<kernel_args_t>(args)...);
  }

  void dump() {
    std::ofstream outfile;
    outfile.open("temp.bin", std::ios_base::binary);
    outfile.write(reinterpret_cast<const char*>(getCode()), getSize());
    outfile.close();
    system("objdump -D -b binary -mi386:x86-64 -M intel temp.bin");
  }

#if 0
  std::vector<uint8_t> log_buffer;
  uint8_t* m_log_addr;
  Xbyak::Reg64 reg_scratch = r9;
  int log_tile_count = 0;

  void log_tile(Xbyak::Tmm tmm, Xbyak::Reg64 reg_stride) {
    auto offset = log_buffer.size();
    log_buffer.resize(offset + 1024, 0xFF);
    m_log_addr = log_buffer.data();
    log_tile_count++;
    // reload base
    mov(reg_scratch, reinterpret_cast<uintptr_t>(&m_log_addr));
    mov(reg_scratch, ptr[reg_scratch]);
    tilestored(ptr[reg_scratch + reg_stride + offset], tmm);
  }

  template <typename T>
  void show_log() {
    T* pdata = reinterpret_cast<T*>(m_log_addr);
    for (int log = 0; log < log_tile_count; log++) {
      std::cout << "========== log " << log << std::endl;
      for (int y = 0; y < 16; y++, pdata += 32) {
        std::cout << "[" << y << "]: ";
        for (int x = 0; x < 32; x++) {
          std::cout << pdata[x] << ",";
        }
        std::cout << "\n";
      }
    }
  }
#endif

  uint8_t log_buffer[4096*1024];
  uint8_t * log_ptr = nullptr;

  void log_reset() {
    log_ptr = log_buffer;
  }
  void log_zmm(const char* fmt, int line_num, Xbyak::Zmm zmm) {
    push(rax);
    push(rbx);
    
    mov(rax, reinterpret_cast<uintptr_t>(&log_ptr));
    mov(rbx, ptr[rax]);
    mov(qword[rbx], reinterpret_cast<uintptr_t>(fmt));    // pointer to fmt
    add(rbx, 8);
    mov(dword[rbx], line_num);                          // DWORD, line-number constant
    add(rbx, 4);
    vmovdqu16(ptr[rbx], zmm);
    add(rbx, 64);
    mov(ptr[rax], rbx);

    pop(rbx);
    pop(rax);
  }
  void log_show() {
    uint8_t* plog = log_buffer;
    int i = 0;
    while(plog < log_ptr) {
        auto * fmt = *reinterpret_cast<const char **>(plog); plog += 8;
        auto line_num = *reinterpret_cast<const int *>(plog); plog += 4;
        printf("line:%d '%s' ", line_num, fmt);
        if (strcmp(fmt, "u8") == 0) {
            for(int k = 0; k < 64; k++)
                printf("%02x,", plog[k]);
        }
        if (strcmp(fmt, "u16") == 0) {
            auto* pu16 = reinterpret_cast<uint16_t *>(plog);
            for(int k = 0; k < 32; k++)
                printf(" %04x,", pu16[k]);
        }
        if (strcmp(fmt, "bf16") == 0) {
            auto* pbf16 = reinterpret_cast<uint16_t *>(plog);
            for(int k = 0; k < 32; k++) {
                auto if32 = static_cast<uint32_t>(pbf16[k]) << 16;
                printf(" %.1f,", reinterpret_cast<float&>(if32));
            }
        }
        if (strcmp(fmt, "f32") == 0) {
            auto* pf32 = reinterpret_cast<float *>(plog);
            for(int k = 0; k < 16; k++) {
                printf(" %8.3f,", pf32[k]);
            }
        }
        printf("\n");
        plog += 64;
    }
  }

};


//===============================================================
inline int getenv(const char * var, int default_value) {
    const char * p = std::getenv(var);
    if (p) default_value = std::atoi(p);
    printf("\e[32mENV:\t %s = %d %s\e[0m\n", var, default_value, p?"":"(default)");
    return default_value;
}

static std::vector<std::string> str_split(const std::string& s, std::string delimiter) {
    std::vector<std::string> ret;
    size_t last = 0;
    size_t next = 0;
    while ((next = s.find(delimiter, last)) != std::string::npos) {
        std::cout << last << "," << next << "=" << s.substr(last, next-last) << "\n";
        ret.push_back(s.substr(last, next-last));
        last = next + 1;
    }
    ret.push_back(s.substr(last));
    return ret;
}

// multiple values separated by ,
inline std::vector<int> getenvs(const char * var, int count = -1, int default_v = 0) {
    std::vector<int> ret;
    const char * p = std::getenv(var);
    if (p) {
        auto vec = str_split(p, ",");
        for(auto& v : vec)
            ret.push_back(std::atoi(v.c_str()));
    }
    while(ret.size() < count)
        ret.push_back(default_v);
    printf("\e[32mENV:\t %s = ", var);
    const char * sep = "";
    for(int v : ret) {
        printf("%s%d", sep, v);
        sep = ",";
    }
    printf("\e[0m\n");
    return ret;
}

//===============================================================
template<typename T>
std::shared_ptr<T> alloc_cache_aligned(int count, T default_value) {
    auto ret = std::shared_ptr<T>(
            reinterpret_cast<T*>(aligned_alloc(64, count * sizeof(T))),
            [](void * p) { ::free(p); });
    
    for(int i = 0; i < count; i++) {
        ret.get()[i] = default_value;
    }
    return ret;
}

//===============================================================
// XTILE initialize on Linux
#ifndef _GNU_SOURCE
#define _GNU_SOURCE /* See feature_test_macros(7) */
#endif
#include <sys/syscall.h> /* For SYS_xxx definitions */
#include <unistd.h>

#define XFEATURE_XTILECFG 17
#define XFEATURE_XTILEDATA 18
#define XFEATURE_MASK_XTILECFG (1 << XFEATURE_XTILECFG)
#define XFEATURE_MASK_XTILEDATA (1 << XFEATURE_XTILEDATA)
#define XFEATURE_MASK_XTILE (XFEATURE_MASK_XTILECFG | XFEATURE_MASK_XTILEDATA)
#define ARCH_GET_XCOMP_PERM 0x1022
#define ARCH_REQ_XCOMP_PERM 0x1023

inline bool initXTILE() {
    unsigned long bitmask = 0;
    long status = syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask);
    if (0 != status)
        return false;
    if (bitmask & XFEATURE_MASK_XTILEDATA)
        return true;

    status = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
    if (0 != status)
        return false; // XFEATURE_XTILEDATA setup is failed, TMUL usage is not allowed
    status = syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask);

    // XFEATURE_XTILEDATA setup is failed, can't use TMUL
    if (0 != status || !(bitmask & XFEATURE_MASK_XTILEDATA))
        return false;

    // XFEATURE_XTILEDATA set successfully, TMUL usage is allowed
    printf("\e[33m""initXTILE success!\n""\e[0m");
    return true;
}

//===============================================================
struct TileConfig {
  uint8_t palette_id;
  uint8_t startRow;
  uint8_t reserved[14];
  uint16_t cols[16];
  uint8_t rows[16];
  void reset(int palette,
             int _startRow,
             const std::vector<std::pair<int, int>>& _rows_columnsBytes) {
    palette_id = palette;
    startRow = _startRow;
    unsigned long i;
    for (i = 0; i < 14; i++) {
      reserved[i] = 0;
    }
    for (i = 0; i < _rows_columnsBytes.size(); i++) {
      rows[i] = _rows_columnsBytes[i].first;
      cols[i] = _rows_columnsBytes[i].second;
    }
    for (; i < 16; i++) {
      cols[i] = 0;
      rows[i] = 0;
    }
  }
} __attribute__((__packed__));

class TileConfiger : public jit_generator {
 public:
  TileConfiger() { create_kernel(); }
  void generate() override {
    Xbyak::Label release;
    test(abi_param1, abi_param1);
    jz(release);
    ldtilecfg(ptr[abi_param1]);
    ret();
    L(release);
    tilerelease();
    ret();
  }
};

// https://stackoverflow.com/questions/23690416/c-template-singleton-static-pointer-initialization-in-header-file
template <typename T>
class Singleton {
 public:
  static T& get() {
    static T instance;
    return instance;
  }
};

class TileConfigScope {
 public:
  TileConfigScope(const TileConfig& cfg) {
    (Singleton<TileConfiger>::get())(&cfg);
  };
  void update(const TileConfig& cfg) { (Singleton<TileConfiger>::get())(&cfg); }
  ~TileConfigScope() { (Singleton<TileConfiger>::get())(nullptr); }
};