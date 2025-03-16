/*
  debug toolkit
*/
#pragma once
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>

// https://stackoverflow.com/a/21371401/9292588
inline const char* filename(const char* path) {
    const char* fname = path;
    while (*path != 0) {
        if (*path == '\\' || *path == '/')
            fname = path + 1;
        path++;
    }
    return fname;
}

inline float get_delta_ms() {
    thread_local auto t0 = std::chrono::high_resolution_clock::now();
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> dt = t1 - t0;
    t0 = t1;
    return dt.count();
}

// unit: ns/us/ms/s
struct Nanoseconds {
    double m_tvalue;
    const char* m_unit;
    Nanoseconds(double _ns) : m_tvalue(_ns) {
        const char* unit = "(ns)";
        if (m_tvalue > 1e3) {
            m_tvalue *= 1e-3;
            unit = "(us)";
        }
        if (m_tvalue > 1e3) {
            m_tvalue *= 1e-3;
            unit = "(ms)";
        }
        if (m_tvalue > 1e3) {
            m_tvalue *= 1e-3;
            unit = "(sec)";
        }
        m_unit = unit;
    }
    friend std::ostream& operator<<(std::ostream& os, const Nanoseconds& obj) {
        os << obj.m_tvalue << obj.m_unit;
        return os;
    }
};

struct _log_id {
    static int get_new() {
        static int id = 0;
        return id++;
    }
};

static int DEBUG_LOG_BRK = std::getenv("DEBUG_LOG_BRK") ? atoi(std::getenv("DEBUG_LOG_BRK")) : 0;
static int DEBUG_LOG_COLOR = std::getenv("DEBUG_LOG_COLOR") ? atoi(std::getenv("DEBUG_LOG_COLOR")) : -1;

template <typename... Ts>
void easy_cout(const char* file, const char* func, int line, Ts... args) {
    auto log_id = _log_id::get_new();

    if (DEBUG_LOG_BRK != 0) {
        // DEBUG_LOG_BRK imply silent
        if (DEBUG_LOG_BRK == log_id) {
            std::cout << "breaking at log id [" << log_id << "]\n";
            asm("int3");
        }
        return;
    }
    std::string tag;
    if (file != nullptr) {
        std::string file_path(file);
        std::string file_name(filename(file_path.c_str()));
        std::string file_name_with_line = file_name + ":" + std::to_string(line);
        tag = file_name_with_line + " ";
    }
    if (func)
        tag = tag + func + "()";

    auto dt_value = get_delta_ms();
    std::string dt_unit = "ms";
    if (dt_value > 1000.0f) {
        dt_value /= 1000.0f;
        dt_unit = "sec";
        if (dt_value > 60.0f) {
            dt_value /= 60.0f;
            dt_unit = "min";
        }
    }

    std::stringstream ss;

    if (DEBUG_LOG_COLOR > 0)
        ss << " \033[37;100m+";
    ss << "[" << log_id << "] " << std::fixed << std::setprecision(3) << dt_value << " " << dt_unit;
    if (DEBUG_LOG_COLOR > 0)
        ss << "\033[36;40m";
    ss << " " << tag;
    if (DEBUG_LOG_COLOR > 0)
        ss << " \033[0m ";

    int dummy[sizeof...(Ts)] = {(ss << args, 0)...};
    (void)dummy;

    ss << "" << std::endl;

    std::cout << ss.str();
}

#define DEBUG0(...)        easy_cout(__FILE__, __func__, __LINE__);
#define DEBUG1(x)          easy_cout(__FILE__, __func__, __LINE__, #x, "=", x);
#define DEBUG2(x1, x2)     easy_cout(__FILE__, __func__, __LINE__, #x1, "=", x1, ",", #x2, "=", x2);
#define DEBUG3(x1, x2, x3) easy_cout(__FILE__, __func__, __LINE__, #x1, "=", x1, ",", #x2, "=", x2, ",", #x3, "=", x3);
#define DEBUG4(x1, x2, x3, x4) \
    easy_cout(__FILE__, __func__, __LINE__, #x1, "=", x1, ",", #x2, "=", x2, ",", #x3, "=", x3, ",", #x4, "=", x4);
#define DEBUG5(x1, x2, x3, x4, x5) \
    easy_cout(__FILE__,            \
              __func__,            \
              __LINE__,            \
              #x1,                 \
              "=",                 \
              x1,                  \
              ",",                 \
              #x2,                 \
              "=",                 \
              x2,                  \
              ",",                 \
              #x3,                 \
              "=",                 \
              x3,                  \
              ",",                 \
              #x4,                 \
              "=",                 \
              x4,                  \
              ",",                 \
              #x5,                 \
              "=",                 \
              x5);
#define DEBUG6(x1, x2, x3, x4, x5, x6) \
    easy_cout(__FILE__,                \
              __func__,                \
              __LINE__,                \
              #x1,                     \
              "=",                     \
              x1,                      \
              ",",                     \
              #x2,                     \
              "=",                     \
              x2,                      \
              ",",                     \
              #x3,                     \
              "=",                     \
              x3,                      \
              ",",                     \
              #x4,                     \
              "=",                     \
              x4,                      \
              ",",                     \
              #x5,                     \
              "=",                     \
              x5,                      \
              ",",                     \
              #x6,                     \
              "=",                     \
              x6);

#define GET_MACRO(_0, _1, _2, _3, _4, _5, _6, NAME, ...) NAME
#define DEBUG_LOG(...) \
    GET_MACRO(_0 __VA_OPT__(, ) __VA_ARGS__, DEBUG6, DEBUG5, DEBUG4, DEBUG3, DEBUG2, DEBUG1, DEBUG0)(__VA_ARGS__)


#ifdef XBYAK_XBYAK_H_
template <class J>
struct JitInfo {
    struct jcout {
        J* jit;
        std::vector<Xbyak::Reg64> preserved_regs;
        std::vector<Xbyak::Xmm> preserved_vmms;
        int vmm_size_byte;
        jcout(J* jit) : jit(jit) {
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
        };
        jout_options m_jout_opt;
        jout_options as_f32 = jout_options::as_f32;
        jout_options as_i32 = jout_options::as_i32;
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
        void _jit_cout(Xbyak::Xmm value) {
            auto rbp = jit->rbp;
            auto rsi = jit->rsi;
            auto rdi = jit->rdi;
            int rbp_disp = -1;
            for (size_t i = 0; i < preserved_vmms.size(); i++) {
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
            } else {
                jit->call(reinterpret_cast<const void*>(printf_vec_i32));
            }
        }
        void _jit_cout(Xbyak::Reg64 value) {
            const char* fmt_r64 = "0x%llx";
            // load reg from snapshot on the stack
            jit->mov(jit->rdi, reinterpret_cast<uintptr_t>(fmt_r64));
            bool found = false;
            for (size_t i = 0; i < preserved_regs.size(); i++) {
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
            for (size_t i = 2; i < preserved_regs.size(); i++)
                jit->push(preserved_regs[i]);

            for (size_t i = 0; i < preserved_vmms.size(); i++) {
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
            for (size_t i = 0; i < preserved_vmms.size(); i++) {
                auto& vmm = preserved_vmms[i];
                jit->vmovdqu(vmm, jit->ptr[rsp - (i + 1) * (vmm_size_byte)]);
            }
            for (size_t i = preserved_regs.size() - 1; i >= 2; i--)
                jit->pop(preserved_regs[i]);
            jit->pop(rbp);
            jit->pop(rbp);
        }
    } jcout;

    J* m_pjit;
    struct debug_info {
        int offset;
        std::string tag;
        int scopes_delta;
        debug_info(int offset, std::string tag, int scopes_delta)
            : offset(offset),
              tag(tag),
              scopes_delta(scopes_delta) {}
    };
    std::vector<debug_info> m_info;

    JitInfo(J* pjit) : m_pjit(pjit), jcout(pjit) {}

    int m_scopes_level = 0;
    void info(const char* name = __builtin_FUNCTION(),
              const char* fname = __builtin_FILE(),
              int lineno = __builtin_LINE()) {
        std::string tag = filename(fname);
        tag += ":";
        tag += std::to_string(lineno);
        tag += " ";
        tag += name;
        tag += "() ";
        m_info.emplace_back(m_pjit->getSize(), tag, 0);
    }

    std::shared_ptr<void> scope(const char* name = __builtin_FUNCTION(),
                                const char* fname = __builtin_FILE(),
                                int lineno = __builtin_LINE()) {
        std::string tag = filename(fname);
        tag += ":";
        tag += std::to_string(lineno);
        tag += " ";
        tag += name;
        tag += "() ";
        m_info.emplace_back(m_pjit->getSize(), tag, 1);
        m_scopes_level++;
        return std::shared_ptr<void>(nullptr, [this, tag](void*) {
            m_info.emplace_back(m_pjit->getSize(), "", -1);
            m_scopes_level--;
            if (m_scopes_level == 0) {
                dump();
            }
        });
    }

    void dump() {
        std::ofstream outfile;
        outfile.open("temp.bin", std::ios_base::binary);
        outfile.write(reinterpret_cast<const char*>(m_pjit->getCode()), m_pjit->getSize());
        outfile.close();

        char* line = NULL;
        size_t len = 0;
        ssize_t length;

#ifdef __x86_64__
        FILE* fp = popen("objdump -D -b binary --no-show-raw-insn -mi386:x86-64 -M intel temp.bin", "r");
#endif
#ifdef __aarch64__
        FILE* fp = popen("objdump -D -b binary --no-show-raw-insn -maarch64 temp.bin", "r");
#endif

        size_t jit_ii = 0;
        int scope_level = 0;
        while ((length = getline(&line, &len, fp)) != -1) {
            int offset = 0;
            int i = 0;

            while (line[i] == ' ')
                i++;
            while (i < 6) {
                offset *= 16;
                if (line[i] >= '0' && line[i] <= '9') {
                    offset += line[i] - '0';
                } else if (line[i] >= 'a' && line[i] <= 'f') {
                    offset += line[i] - 'a' + 10;
                } else {
                    offset = offset / 16;
                    break;
                }
                i++;
            }
            if (line[i] != ':') {
                // printf("%s", line);
                continue;
            }

            i++;
            while (line[i] == ' ' || line[i] == '\t')
                i++;

            if (jit_ii < m_info.size() && offset >= m_info[jit_ii].offset) {
                // enter new scope
                auto sname = m_info[jit_ii].tag;
                for (int k = 0; k < scope_level * 4; k++)
                    printf(" ");
                // printf("\e[0;33m %s\e[0m\n", sname.c_str());
                printf(">>>>>>>>>>>>> %s\n", sname.c_str());
                scope_level += m_info[jit_ii].scopes_delta;
                jit_ii++;
            }

            for (int k = 0; k < scope_level * 4; k++)
                printf(" ");
            printf("%x: %s", offset, line + i);
        }
        pclose(fp);
    }
};
#endif