
#pragma once


#include "../thirdparty/xbyak/xbyak/xbyak.h"

#include <cstdlib>
#include <fstream>
#include <vector>
#include <cstring>
#include <cstddef>
#include <memory>

#include "../include/misc.hpp"

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
  int64_t operator()(kernel_args_t... args) const {
    using jit_kernel_func_t = int64_t (*)(const kernel_args_t... args);
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

  std::vector<uint8_t> log_buffer;
  uint8_t * log_ptr = nullptr;

  void log_reset() {
    if (log_buffer.empty()) {
        log_buffer.resize(4096*1024);
    }
    log_ptr = &log_buffer[0];
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
    uint8_t* plog = &log_buffer[0];
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
template <typename T>
struct tensor2D {
    int shape[2] = {0};
    int stride_bytes = 0;
    std::shared_ptr<T> data;

    uint64_t size() {
        return shape[0] * stride_bytes;
    }

    tensor2D() = default;

    tensor2D(int rows, int cols, bool initialize = false) {
        shape[0] = rows;
        shape[1] = cols;
        stride_bytes = cols * sizeof(T);
        data = alloc_cache_aligned<T>(rows * cols);
        if (initialize) {
            auto* pdata = data.get();
            for(int i = 0; i < rows * cols; i++)
                pdata[i] = 0;
        }        
    }

    tensor2D<T> clone() {
        // deep copy
        tensor2D<T> ret;
        ret.shape[0] = shape[0];
        ret.shape[1] = shape[1];
        ret.stride_bytes = shape[1] * sizeof(T);
        ret.data = alloc_cache_aligned<T>(shape[0] * shape[1]);
        memcpy(ret.data.get(), data.get(), shape[0] * shape[1] * sizeof(T));
        return ret;
    }

    T* ptr(int i0 = 0, int i1 = 0) const {
        return data.get() + i0 * shape[1] + i1;
    }

    T& operator()(int i0, int i1) const {
        return *ptr(i0, i1);
    }

    // flatten 1D indexing
    T& operator[](int i) const {
        auto i0 = i / shape[1];
        auto i1 = i % shape[1];
        return *ptr(i0, i1);
    }
};

template <typename T>
void show_value(T& v) {
    printf("%6.1f,", static_cast<float>(v));
}

template <>
inline void show_value<int>(int& v) {
    printf("%6d,", v);
}

template <>
inline void show_value<uint32_t>(uint32_t& v) {
    printf("%6d,", v);
}

template <typename T>
void show_tensor2D(const char * name, const tensor2D<T>& t, int rows_limit = 16, int cols_limit = 16) {
    printf("tensor2D<%s(size=%d)> %s(%d, %d)={\n", typeid(T).name(), sizeof(T), name, t.shape[0], t.shape[1]);
    for (int i0 = 0; i0 < t.shape[0]; i0++) {
        if (i0 > rows_limit - 1 && i0 < t.shape[0] - 1) continue;
        printf("\t{");
        for (int i1 = 0; i1 < t.shape[1]; i1++) {
            if (i1 > cols_limit - 1 && i1 < t.shape[1] - 1) continue;
            show_value(t(i0, i1));
            if (i1 == cols_limit - 1) printf("...,");
        }
        printf(" },\\\\ row %d \n", i0);
        if (i0 == rows_limit - 1) printf("\t... ... ... \\\\ row %d ~ %d \n", i0, t.shape[0]-2);
    }
    printf("};\n");
}

template<typename T>
bool compare(tensor2D<T>& ref, tensor2D<T>& cur, bool verbose) {
    if (ref.shape[0] != cur.shape[0] || ref.shape[1] != cur.shape[1]) {
        if (verbose)
            printf("compare tensor2D shape incompatible   ref:(%d,%d)  cur:(%d,%d)\n",
                    ref.shape[0], ref.shape[1], cur.shape[0], cur.shape[1]);
        return false;
    }
    bool all_equal = true;
    for (int m = 0; m < ref.shape[0]; m++) {
        int ndiff = -1;
        for (int n = 0; n < ref.shape[1]; n++) {
            if (ref(m, n) != cur(m, n)) {
                ndiff = n;
                break;
            }
        }
            
        if (ndiff < 0)
            continue;

        all_equal = false;

        if (verbose) {
            int n;
            printf("ref(%d,%d): ..., ", m, ndiff);
            for (n = ndiff; n < ref.shape[1] && n < ndiff + 16; n++) show_value(ref(m, n));
            if (n < ref.shape[1]) printf("...");
            printf("\n");
            printf("cur(%d,%d): ..., ", m, ndiff);
            for (n = ndiff; n < ref.shape[1] && n < ndiff + 16; n++) {
                bool equal = cur(m, n) == ref(m, n);
                if (!equal)
                    printf("\e[31m");
                show_value(cur(m, n));
                if (!equal)
                    printf("\e[0m");
            }
            if (n < ref.shape[1]) printf("...");
            printf("\n");
        }
    }
    if (verbose) {
        printf("compare result: %s\n", all_equal ? "equal" : "unequal");
    }
    return all_equal;
}

//===============================================================
class CLflush : public jit_generator {
    CLflush() {
        create_kernel("CLflush");
    }

public:
    void generate() override {
        //  abi_param1 ~ abi_param6, RAX, R10, R11
        auto start_ptr = abi_param1;
        auto loop_cnt = abi_param2;
        Xbyak::Label loop_begin;
        mfence();
        align(64, false);
        L(loop_begin);

        clflush(ptr[start_ptr]);
        mfence();
        lea(start_ptr, ptr[start_ptr + 64]);
        dec(loop_cnt);
        jne(loop_begin, T_NEAR); // jmp back if sign is set: (start_ptr-end_ptr) < 0
        ret();
    }
    static inline void run(void* pv, int64_t size_bytes) {
        static CLflush jit;
        jit(pv, (size_bytes + 63)/64);
    }
};

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
  TileConfiger() { create_kernel("TileConfiger"); }
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