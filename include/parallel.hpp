#pragma once

#include "omp.h"

inline int parallel_get_max_threads() {
    return omp_get_max_threads();
}

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

template <typename F>
void parallel_nt_static(int nthr, const F& func) {
    if (nthr == 0)
        nthr = omp_get_max_threads();
#pragma omp parallel
    {
        int ithr = omp_get_thread_num();
        func(ithr, nthr);
    }
}

template <typename T0, typename F>
void parallel_for(const T0& D0, const F& func) {
    parallel_nt_static(0, [&](int ithr, int nthr) {
        T0 n_start, n_end;
        splitter(D0, nthr, ithr, n_start, n_end);
        func(n_start, n_end);
    });
}
