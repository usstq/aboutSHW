
#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <cstring>
union KArg {
    int64_t i;
    float f;
    void* p;
    float* pf32;
    uint8_t* pu8;
    int8_t* pi8;
    uint16_t* pu16;
};

// https://stackoverflow.com/a/21371401/9292588
#if USE_DEBUG
#define DEBUG0(...) std::cout << "===" << __LINE__ << ":" << std::endl;
#define DEBUG1(x) std::cout << "===" << __LINE__ << ":" << #x << "=" << x << std::endl;
#define DEBUG2(x1, x2) std::cout << "===" << __LINE__ << ":" << #x1 << "=" << x1 << "," << #x2 << "=" << x2 << std::endl;
#define DEBUG3(x1, x2, x3) std::cout << "===" << __LINE__ << ":" << #x1 << "=" << x1 << "," << #x2 << "=" << x2 << "," << #x3 << "=" << x3 << std::endl;
#define DEBUG4(x1, x2, x3, x4) std::cout << "===" << __LINE__ << ":" << #x1 << "=" << x1 << "," << #x2 << "=" << x2 << "," << #x3 << "=" << x3 << "," << #x4 << "=" << x4 << std::endl;
#define DEBUG5(x1, x2, x3, x4, x5) std::cout << "===" << __LINE__ << ":" << #x1 << "=" << x1 << "," << #x2 << "=" << x2 << "," << #x3 << "=" << x3 << "," << #x4 << "=" << x4 <<  "," << #x5 << "=" << x5 << std::endl;
#define DEBUG6(x1, x2, x3, x4, x5, x6) std::cout << "===" << __LINE__ << ":" << #x1 << "=" << x1 << "," << #x2 << "=" << x2 << "," << #x3 << "=" << x3 << "," << #x4 << "=" << x4 <<  "," << #x5 << "=" << x5 << "," << #x6 << "=" << x6 << std::endl;

#define GET_MACRO(_0, _1, _2, _3, _4, _5, _6, NAME, ...) NAME
#define DEBUG(...) GET_MACRO(_0 __VA_OPT__(,) __VA_ARGS__,  DEBUG6, DEBUG5, DEBUG4, DEBUG3, DEBUG2, DEBUG1, DEBUG0)(__VA_ARGS__)
#else
#define DEBUG(...)
#endif


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
