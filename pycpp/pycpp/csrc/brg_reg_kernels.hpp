template <int row>
void brgemm_4x3(const float* A,
                int A_stride,  // stride in number of element
                const float* B,
                int B_stride,  // stride in number of element
                float* C,
                int C_stride,  // stride in number of element
                int K,
                bool is_accumulate_C) {
    // loop in unit of register blocking: (3x4*8)
    __m256 c00, c01, c02;
    __m256 c10, c11, c12;
    __m256 c20, c21, c22;
    __m256 c30, c31, c32;

    if (is_accumulate_C) {
        c00 = _mm256_loadu_ps(C + 8 * 0);
        c01 = _mm256_loadu_ps(C + 8 * 1);
        c02 = _mm256_loadu_ps(C + 8 * 2);

        if (row > 1) {
            c10 = _mm256_loadu_ps(C + C_stride + 8 * 0);
            c11 = _mm256_loadu_ps(C + C_stride + 8 * 1);
            c12 = _mm256_loadu_ps(C + C_stride + 8 * 2);
        }
        if (row > 2) {
            c20 = _mm256_loadu_ps(C + 2 * C_stride + 8 * 0);
            c21 = _mm256_loadu_ps(C + 2 * C_stride + 8 * 1);
            c22 = _mm256_loadu_ps(C + 2 * C_stride + 8 * 2);
        }
        if (row > 3) {
            c30 = _mm256_loadu_ps(C + 3 * C_stride + 8 * 0);
            c31 = _mm256_loadu_ps(C + 3 * C_stride + 8 * 1);
            c32 = _mm256_loadu_ps(C + 3 * C_stride + 8 * 2);
        }
    } else {
        c00 = _mm256_setzero_ps();
        c01 = _mm256_setzero_ps();
        c02 = _mm256_setzero_ps();

        if (row > 1) {
            c10 = _mm256_setzero_ps();
            c11 = _mm256_setzero_ps();
            c12 = _mm256_setzero_ps();
        }
        if (row > 2) {
            c20 = _mm256_setzero_ps();
            c21 = _mm256_setzero_ps();
            c22 = _mm256_setzero_ps();
        }
        if (row > 3) {
            c30 = _mm256_setzero_ps();
            c31 = _mm256_setzero_ps();
            c32 = _mm256_setzero_ps();
        }
    }

    auto* prefetch_B = B + 64 / sizeof(float) * 10;

    // reducing along k dimension
    //   with -O2 optimization flag, following kernel has 6~7 cycles-per-iteration
    //   which is consistent with FMA's throughput(0.5)
    for (int k = 0; k < K; k++, B += B_stride, A++, prefetch_B += B_stride) {
        // 16-ymm-registers are just enough for 4x3 register blocking
        __m256 b0 = _mm256_loadu_ps(B + 8 * 0);
        __m256 b1 = _mm256_loadu_ps(B + 8 * 1);
        __m256 b2 = _mm256_loadu_ps(B + 8 * 2);

        //_mm_prefetch(prefetch_B, _MM_HINT_T0);

        __m256 a = _mm256_broadcast_ss(A);
        c00 = _mm256_fmadd_ps(a, b0, c00);
        c01 = _mm256_fmadd_ps(a, b1, c01);
        c02 = _mm256_fmadd_ps(a, b2, c02);
        if (row > 1) {
            a = _mm256_broadcast_ss(A + A_stride);
            c10 = _mm256_fmadd_ps(a, b0, c10);
            c11 = _mm256_fmadd_ps(a, b1, c11);
            c12 = _mm256_fmadd_ps(a, b2, c12);
        }
        if (row > 2) {
            a = _mm256_broadcast_ss(A + 2 * A_stride);
            c20 = _mm256_fmadd_ps(a, b0, c20);
            c21 = _mm256_fmadd_ps(a, b1, c21);
            c22 = _mm256_fmadd_ps(a, b2, c22);
        }
        if (row > 3) {
            a = _mm256_broadcast_ss(A + 3 * A_stride);
            c30 = _mm256_fmadd_ps(a, b0, c30);
            c31 = _mm256_fmadd_ps(a, b1, c31);
            c32 = _mm256_fmadd_ps(a, b2, c32);
        }
    }

    // store C back
    _mm256_storeu_ps(C + 8 * 0, c00);
    _mm256_storeu_ps(C + 8 * 1, c01);
    _mm256_storeu_ps(C + 8 * 2, c02);
    if (row > 1) {
        _mm256_storeu_ps(C + C_stride + 8 * 0, c10);
        _mm256_storeu_ps(C + C_stride + 8 * 1, c11);
        _mm256_storeu_ps(C + C_stride + 8 * 2, c12);
    }
    if (row > 2) {
        _mm256_storeu_ps(C + 2 * C_stride + 8 * 0, c20);
        _mm256_storeu_ps(C + 2 * C_stride + 8 * 1, c21);
        _mm256_storeu_ps(C + 2 * C_stride + 8 * 2, c22);
    }
    if (row > 3) {
        _mm256_storeu_ps(C + 3 * C_stride + 8 * 0, c30);
        _mm256_storeu_ps(C + 3 * C_stride + 8 * 1, c31);
        _mm256_storeu_ps(C + 3 * C_stride + 8 * 2, c32);
    }
}

template <int row>
void brgemm_4x1(const float* A,
                int A_stride,  // stride in number of element
                const float* B,
                int B_stride,  // stride in number of element
                float* C,
                int C_stride,  // stride in number of element
                int K,
                bool is_accumulate_C) {
    // loop in unit of register blocking: (3x4*8)
    __m256 c00;
    __m256 c10;
    __m256 c20;
    __m256 c30;

    if (is_accumulate_C) {
        c00 = _mm256_loadu_ps(C + 8 * 0);
        if (row > 1) {
            c10 = _mm256_loadu_ps(C + C_stride + 8 * 0);
        }
        if (row > 2) {
            c20 = _mm256_loadu_ps(C + 2 * C_stride + 8 * 0);
        }
        if (row > 3) {
            c30 = _mm256_loadu_ps(C + 3 * C_stride + 8 * 0);
        }
    } else {
        c00 = _mm256_setzero_ps();
        if (row > 1) {
            c10 = _mm256_setzero_ps();
        }
        if (row > 2) {
            c20 = _mm256_setzero_ps();
        }
        if (row > 3) {
            c30 = _mm256_setzero_ps();
        }
    }

    // reducing along k dimension
    //   with -O2 optimization flag, following kernel has 6~7 cycles-per-iteration
    //   which is consistent with FMA's throughput(0.5)
    for (int k = 0; k < K; k++, B += B_stride, A++) {
        // 16-ymm-registers are just enough for 4x3 register blocking
        __m256 b0 = _mm256_loadu_ps(B + 8 * 0);
        __m256 a = _mm256_broadcast_ss(A);
        c00 = _mm256_fmadd_ps(a, b0, c00);
        if (row > 1) {
            a = _mm256_broadcast_ss(A + A_stride);
            c10 = _mm256_fmadd_ps(a, b0, c10);
        }
        if (row > 2) {
            a = _mm256_broadcast_ss(A + 2 * A_stride);
            c20 = _mm256_fmadd_ps(a, b0, c20);
        }
        if (row > 3) {
            a = _mm256_broadcast_ss(A + 3 * A_stride);
            c30 = _mm256_fmadd_ps(a, b0, c30);
        }
    }
    _mm256_storeu_ps(C + 8 * 0, c00);
    if (row > 1) {
        _mm256_storeu_ps(C + C_stride + 8 * 0, c10);
    }
    if (row > 2) {
        _mm256_storeu_ps(C + 2 * C_stride + 8 * 0, c20);
    }
    if (row > 3) {
        _mm256_storeu_ps(C + 3 * C_stride + 8 * 0, c30);
    }
}

template <int row = 0>
void brgemm_2x6(const float* A,
                int A_stride,  // stride in number of element
                const float* B,
                int B_stride,  // stride in number of element
                float* C,
                int C_stride,  // stride in number of element
                int K,
                bool is_accumulate_C) {
    // loop in unit of register blocking: (3x4*8)
    __m256 c0, c1, c2, c3, c4, c5;
    __m256 c6, c7, c8, c9, ca, cb;

    if (is_accumulate_C) {
        c0 = _mm256_loadu_ps(C + 8 * 0);
        c1 = _mm256_loadu_ps(C + 8 * 1);
        c2 = _mm256_loadu_ps(C + 8 * 2);
        c3 = _mm256_loadu_ps(C + 8 * 3);
        c4 = _mm256_loadu_ps(C + 8 * 4);
        c5 = _mm256_loadu_ps(C + 8 * 5);

        c6 = _mm256_loadu_ps(C + C_stride + 8 * 0);
        c7 = _mm256_loadu_ps(C + C_stride + 8 * 1);
        c8 = _mm256_loadu_ps(C + C_stride + 8 * 2);
        c9 = _mm256_loadu_ps(C + C_stride + 8 * 3);
        ca = _mm256_loadu_ps(C + C_stride + 8 * 4);
        cb = _mm256_loadu_ps(C + C_stride + 8 * 5);
    } else {
        c0 = _mm256_setzero_ps();
        c1 = _mm256_setzero_ps();
        c2 = _mm256_setzero_ps();
        c3 = _mm256_setzero_ps();
        c4 = _mm256_setzero_ps();
        c5 = _mm256_setzero_ps();
        c6 = _mm256_setzero_ps();
        c7 = _mm256_setzero_ps();
        c8 = _mm256_setzero_ps();
        c9 = _mm256_setzero_ps();
        ca = _mm256_setzero_ps();
        cb = _mm256_setzero_ps();
    }
    // reducing along k dimension
    //   with -O2 optimization flag, following kernel has 6~7 cycles-per-iteration
    //   which is consistent with FMA's throughput(0.5)
    for (int k = 0; k < K; k++, B += B_stride, A++) {
        // 16-ymm-registers are just enough for 4x3 register blocking
        __m256 b;
        __m256 a0 = _mm256_broadcast_ss(A);
        __m256 a1 = _mm256_broadcast_ss(A + A_stride);
        // read enough data along current cache fetching line
        b = _mm256_loadu_ps(B + 8 * 0);
        c0 = _mm256_fmadd_ps(a0, b, c0);
        c6 = _mm256_fmadd_ps(a1, b, c6);
        b = _mm256_loadu_ps(B + 8 * 1);
        c1 = _mm256_fmadd_ps(a0, b, c1);
        c7 = _mm256_fmadd_ps(a1, b, c7);
        b = _mm256_loadu_ps(B + 8 * 2);
        c2 = _mm256_fmadd_ps(a0, b, c2);
        c8 = _mm256_fmadd_ps(a1, b, c8);
        b = _mm256_loadu_ps(B + 8 * 3);
        c3 = _mm256_fmadd_ps(a0, b, c3);
        c9 = _mm256_fmadd_ps(a1, b, c9);
        b = _mm256_loadu_ps(B + 8 * 4);
        c4 = _mm256_fmadd_ps(a0, b, c4);
        ca = _mm256_fmadd_ps(a1, b, ca);
        b = _mm256_loadu_ps(B + 8 * 5);
        c5 = _mm256_fmadd_ps(a0, b, c5);
        cb = _mm256_fmadd_ps(a1, b, cb);
    }

    // store C back
    _mm256_storeu_ps(C + 8 * 0, c0);
    _mm256_storeu_ps(C + 8 * 1, c1);
    _mm256_storeu_ps(C + 8 * 2, c2);
    _mm256_storeu_ps(C + 8 * 3, c3);
    _mm256_storeu_ps(C + 8 * 4, c4);
    _mm256_storeu_ps(C + 8 * 5, c5);
    _mm256_storeu_ps(C + C_stride + 8 * 0, c6);
    _mm256_storeu_ps(C + C_stride + 8 * 1, c7);
    _mm256_storeu_ps(C + C_stride + 8 * 2, c8);
    _mm256_storeu_ps(C + C_stride + 8 * 3, c9);
    _mm256_storeu_ps(C + C_stride + 8 * 4, ca);
    _mm256_storeu_ps(C + C_stride + 8 * 5, cb);
}

template <int rows, int prefetch_v = 16>
void brgemm_6x2(const float* A,
                int A_stride,  // stride in number of element
                const float* B,
                int B_stride,  // stride in number of element
                float* C,
                int C_stride,  // stride in number of element
                int K,
                bool is_accumulate_C) {
    __m256 c0, c1, c2, c3, c4, c5;
    __m256 c6, c7, c8, c9, ca, cb;

    if (is_accumulate_C) {
        auto* src = C;
        c0 = _mm256_loadu_ps(src + 8 * 0);
        c1 = _mm256_loadu_ps(src + 8 * 1);
        if (rows > 1) {
            src += C_stride;
            c2 = _mm256_loadu_ps(src + 8 * 0);
            c3 = _mm256_loadu_ps(src + 8 * 1);
        }
        if (rows > 2) {
            src += C_stride;
            c4 = _mm256_loadu_ps(src + 8 * 0);
            c5 = _mm256_loadu_ps(src + 8 * 1);
        }
        if (rows > 3) {
            src += C_stride;
            c6 = _mm256_loadu_ps(src + 8 * 0);
            c7 = _mm256_loadu_ps(src + 8 * 1);
        }
        if (rows > 4) {
            src += C_stride;
            c8 = _mm256_loadu_ps(src + 8 * 0);
            c9 = _mm256_loadu_ps(src + 8 * 1);
        }
        if (rows > 5) {
            src += C_stride;
            ca = _mm256_loadu_ps(src + 8 * 0);
            cb = _mm256_loadu_ps(src + 8 * 1);
        }
    } else {
        c0 = _mm256_setzero_ps();
        c1 = _mm256_setzero_ps();
        c2 = _mm256_setzero_ps();
        c3 = _mm256_setzero_ps();
        c4 = _mm256_setzero_ps();
        c5 = _mm256_setzero_ps();
        c6 = _mm256_setzero_ps();
        c7 = _mm256_setzero_ps();
        c8 = _mm256_setzero_ps();
        c9 = _mm256_setzero_ps();
        ca = _mm256_setzero_ps();
        cb = _mm256_setzero_ps();
    }

    const auto* pA3 = A + 3 * A_stride;
    const auto prefetch_stride = B_stride * prefetch_v;
    int k;
    for (k = 0; k < K; k++, B += B_stride, A++, pA3++) {
        __m256 b0 = _mm256_loadu_ps(B + 8 * 0);
        __m256 b1 = _mm256_loadu_ps(B + 8 * 1);

        if (prefetch_v >= 0)
            _mm_prefetch(B + 16, _MM_HINT_T0);
        if (prefetch_v > 0)
            _mm_prefetch(B + prefetch_stride, _MM_HINT_T0);

        __m256 a0 = _mm256_broadcast_ss(A);
        c0 = _mm256_fmadd_ps(a0, b0, c0);
        c1 = _mm256_fmadd_ps(a0, b1, c1);
        if (rows > 1) {
            a0 = _mm256_broadcast_ss(A + A_stride);
            c2 = _mm256_fmadd_ps(a0, b0, c2);
            c3 = _mm256_fmadd_ps(a0, b1, c3);
        }
        if (rows > 2) {
            a0 = _mm256_broadcast_ss(A + 2 * A_stride);
            c4 = _mm256_fmadd_ps(a0, b0, c4);
            c5 = _mm256_fmadd_ps(a0, b1, c5);
        }

        if (rows > 3) {
            a0 = _mm256_broadcast_ss(pA3);
            c6 = _mm256_fmadd_ps(a0, b0, c6);
            c7 = _mm256_fmadd_ps(a0, b1, c7);
        }
        if (rows > 4) {
            a0 = _mm256_broadcast_ss(pA3 + A_stride);
            c8 = _mm256_fmadd_ps(a0, b0, c8);
            c9 = _mm256_fmadd_ps(a0, b1, c9);
        }
        if (rows > 5) {
            a0 = _mm256_broadcast_ss(pA3 + 2 * A_stride);
            ca = _mm256_fmadd_ps(a0, b0, ca);
            cb = _mm256_fmadd_ps(a0, b1, cb);
        }
    }

    // store C back
    _mm256_storeu_ps(C + 8 * 0, c0);
    _mm256_storeu_ps(C + 8 * 1, c1);
    if (rows > 1) {
        C += C_stride;
        _mm256_storeu_ps(C + 8 * 0, c2);
        _mm256_storeu_ps(C + 8 * 1, c3);
    }
    if (rows > 2) {
        C += C_stride;
        _mm256_storeu_ps(C + 8 * 0, c4);
        _mm256_storeu_ps(C + 8 * 1, c5);
    }
    if (rows > 3) {
        C += C_stride;
        _mm256_storeu_ps(C + 8 * 0, c6);
        _mm256_storeu_ps(C + 8 * 1, c7);
    }
    if (rows > 4) {
        C += C_stride;
        _mm256_storeu_ps(C + 8 * 0, c8);
        _mm256_storeu_ps(C + 8 * 1, c9);
    }
    if (rows > 5) {
        C += C_stride;
        _mm256_storeu_ps(C + 8 * 0, ca);
        _mm256_storeu_ps(C + 8 * 1, cb);
    }
}
