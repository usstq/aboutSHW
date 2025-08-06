/*
 * Copyright (c) 2020-2023, Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#include <cm/cm.h>
#include <cm/cmtl.h>


template<typename T, int N>
void show(const vector<T, N> mat) {
    printf("vector [%d]:\n[", N);
    for(int n = 0; n < N; n ++) {
        printf("%8.4f,", mat[n]);
    }
    printf("]\n");
}

template<typename T, int M, int N>
void show(const matrix<T, M, N> mat) {
    printf("Matrix [%d, %d]:\n", M, N);
    for(int m = 0; m < M; m ++) {
        printf("\t[");
        for(int n = 0; n < N; n ++) {
            printf("%8.4f,", mat[m][n]);
        }
        printf("],\n");
    }
    printf("]\n");
}

// c_max:             [b, hq, m_pad]
// c_max_wg:          [b, hq, m_groups, m_pad]
// c_exp_partial_sum: [b, hq, m_pad, n_after_sum_pad]
// c_sum:             [b, hq, m_pad/TOKEN_IN_BLOCK, n_after_sum_pad]
CM_INLINE void find(int m_block, svmptr_t c_max, svmptr_t c_max_wg, svmptr_t c_exp_partial_sum, svmptr_t c_sum, uint m_pad, uint n_after_sum_pad) {
    constexpr int SG_SIZE = 16;
#ifndef BLOCK_SG_M
    #define BLOCK_SG_M  64
    #define BLOCK_SG_N  32
    #define SG_M  2
    #define SG_N  4
    #define HEAD_SIZE  128
    #define KV_BLOCK_SIZE  256
    #define STRIDE  16
    #define BLOCK_SIZE 128
    #define BLOCK_SHARE_MAX 256
#endif

    const int TOKEN_IN_BLOCK = (BLOCK_SIZE / STRIDE);
    int m = m_block * TOKEN_IN_BLOCK;
    vector<half, TOKEN_IN_BLOCK> max_m;
    max_m.format<int>() = cm_ptr_load<int, TOKEN_IN_BLOCK / 2>((int*)c_max, m * (int)sizeof(half));
    const int TOKEN_SHARE_MAX = BLOCK_SHARE_MAX / TOKEN_IN_BLOCK;
    c_exp_partial_sum += m * n_after_sum_pad * (int)sizeof(half);
    c_max_wg += m * (int)sizeof(half);
    constexpr half log2e = 1.4426950408889634f;
    lsc::block_2d_desc<half, 1, TOKEN_IN_BLOCK, TOKEN_SHARE_MAX> desc_sum{ c_exp_partial_sum, TOKEN_IN_BLOCK - 1, (uint)(n_after_sum_pad * sizeof(half) - 1), (uint)(n_after_sum_pad * sizeof(half) - 1),
        0, 0 };
    matrix<float, TOKEN_IN_BLOCK, TOKEN_SHARE_MAX> sum_m = 0;
    matrix<half, TOKEN_IN_BLOCK, TOKEN_SHARE_MAX> data;
    // compensation: val*exp(local - global)
    for (int j = 0, idx = 0; j < n_after_sum_pad; j += TOKEN_SHARE_MAX, idx++) {
        vector<half, TOKEN_IN_BLOCK> max_m_in_group;
        max_m_in_group.format<int>() = cm_ptr_load<int, TOKEN_IN_BLOCK / 2>((int*)c_max_wg, m_pad * idx * (int)sizeof(half));
        cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached>(data.format<half>(), desc_sum);
        for (int i = 0; i < TOKEN_IN_BLOCK; i++) {
            data.row(i) *= cm_exp((max_m_in_group[i] - max_m[i]) * log2e);
            sum_m.row(i) += data.row(i);
        }
        cm_store(desc_sum, data.format<half>());
        desc_sum.set_block_x(desc_sum.get_block_x() + TOKEN_SHARE_MAX);
    }

    // exp/sum
    vector<half, TOKEN_IN_BLOCK> inv_sum_v;
    for (int i = 0; i < TOKEN_IN_BLOCK; i++) {
        inv_sum_v[i] = 1.0f / cm_sum<float>(sum_m.row(i));
    }
    // compensation: sum(val*inv_sum_v)
    desc_sum.set_block_x(0);
    for (int j = 0; j < n_after_sum_pad; j += TOKEN_SHARE_MAX) {
        cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached>(data.format<half>(), desc_sum);
        data.row(0) *= inv_sum_v[0];
        for (int i = 1; i < TOKEN_IN_BLOCK; i++) {
            data.row(0) += data.row(i) * inv_sum_v[i];
        }
        desc_sum.set_block_x(desc_sum.get_block_x() + TOKEN_SHARE_MAX);
        cm_ptr_store<int, TOKEN_SHARE_MAX / 2>((int*)c_sum, (m_block * n_after_sum_pad + j) * (int)sizeof(half), data.row(0).format<int>());
    }
}

