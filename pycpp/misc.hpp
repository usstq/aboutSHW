#pragma once
#include <iostream>
#include <sstream>
#include <stdexcept>

template <typename... TS>
void _write_all(std::ostream& os, TS&&... args) {
    int dummy[sizeof...(TS)] = {(os << std::forward<TS>(args), 0)...};
    (void)dummy;
}

#ifdef __x86_64__
#    define TRAP_INST() __asm__("int3");
#endif

#ifdef __aarch64__
#    define TRAP_INST() __asm__("brk #0x1");
#endif

#define OPENVINO_ASSERT(cond, ...)                                                      \
    if (!(cond)) {                                                                      \
        std::stringstream ss;                                                           \
        _write_all(ss, __FILE__, ":", __LINE__, " ", #cond, " failed:", ##__VA_ARGS__); \
        std::cout << "\033[31m" << ss.str() << "\033[0m" << std::endl;                  \
        TRAP_INST();                                                                    \
        throw std::runtime_error(ss.str());                                             \
    }
