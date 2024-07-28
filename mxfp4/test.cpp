
#include "mxformat.hpp"
#include <sstream>
#include <iostream>

#include <vector>

//#include "../perf/linux_perf.hpp"
//#include "../thirdparty/xbyak/xbyak/xbyak.h"

int test_f32_to_e2m1 = []() {
    float src1[] = {
        0.0f,   0.5f,   1.0f,   1.5f,   2.0f,   3.0f,   4.0f,   6.0f,
        -0.0f,  -0.5f, -1.0f,  -1.5f,  -2.0f,  -3.0f,  -4.0f,  -6.0f
    };
    for(int i = 0; i<sizeof(src1)/sizeof(src1[0]); i++) {
        auto fp4 = mxformat::e2m1(src1[i]);
        auto recover = float(fp4);
        if(recover != src1[i]) {
            printf("%6.4f to 0x%x (e2m1) and back to %6.4f is unexpected!\n",
                    src1[i], static_cast<int>(fp4), recover);
        }
        if (i > 0) {
            auto vsrc = src1[i]-0.1f;
            auto fp4 = mxformat::e2m1(vsrc);
            auto recover = float(fp4);
            if(recover != src1[i]) {
                printf("%6.4f to 0x%x (e2m1) and back to %6.4f is unexpected!\n",
                        vsrc, static_cast<int>(fp4), recover);
            }
        }
    }
    float half_expect[] = {
         0.0f,  1.0f,  1.0f,  2.0f,  2.0f,  4.0f,  4.0f, 3.0f,
        -0.0f, -1.0f, -1.0f, -2.0f, -2.0f, -4.0f, -4.0f
    };
    for(int i = 1; i<sizeof(src1)/sizeof(src1[0]); i++) {

        if (src1[i-1] == 6.0f) continue;

        auto half = (src1[i] + src1[i-1])*0.5f;
        auto fp4 = mxformat::e2m1(half);
        auto recover = float(fp4);
        //printf("half %f to fp4 & back %f\n", half, recover);
        if (recover != half_expect[i-1]) {
            printf("ERROR: half %f to fp4 & back %f\n", half, recover);
        }

        auto half_minus = half - 0.01f;
        auto half_plus = half + 0.01f;
        if (half < 0) {
            half_minus = half + 0.01f;
            half_plus = half - 0.01f;
        }
        auto fp4_minus = mxformat::e2m1(half_minus);
        auto fp4_plus = mxformat::e2m1(half_plus);
        if (fp4_plus.bits != fp4_minus.bits + 1) {
            printf("ERROR: half %f (-/+)delta to %d/%d\n", half, (int)fp4_minus, (int)fp4_plus);
        }
    }

    printf("test_f32_to_e2m1 is done\n");
    return 0;
}();

int test_e8m0 = []() {
    constexpr uint8_t e8m0_bias = 127;
    {
        float expect = 1.0f;
        for(int i = 0; i <= 127; i++) {
            auto got = mxformat::e8m0_to_float(e8m0_bias + i);
            if (got != expect) {
                std::cout << "[" << i << "] " << got << " != " << expect << std::endl;
                abort();
            }
            expect *= 2.0f;
        }
    }
    {
        float expect = 1.0f;
        for(int i = 0; i >= -127; i--) {
            auto got = mxformat::e8m0_to_float(e8m0_bias + i);
            if (got != expect) {
                std::cout << "[" << i << "] " << got << " != " << expect << std::endl;
                abort();
            }
            expect *= 0.5f;
        }
    }
    {
        auto got = mxformat::e8m0_to_float(255);
        if (!std::isnan(got)) {
            std::cout << "expect [Nan] got " << got << std::endl;
            abort();
        }
    }
    std::cout << "test_e8m0 is done\n";
    return 0;
}();


int test_mxfp4 = [](){
    float src[32];
    auto test_accuracy = [&](){
        mxformat::mxfp4 m0(src);
        float avg_rel_error = 0.0f;
        for(int i = 0; i < 32; i++) {
            auto recorver = m0[i];
            auto abs_error = std::abs(src[i] - recorver);
            auto rel_error = abs_error / std::abs(src[i]);
            if (abs_error == 0) continue;
            if (rel_error > 0.05)
                printf("\t%10.4f->%10.4f  error(abs/rel): %f/%f\n", src[i], recorver, abs_error, rel_error);
            avg_rel_error += rel_error;
        }
        avg_rel_error/=32;
        printf("avg_rel_error = %f\n", avg_rel_error);
        return avg_rel_error;
    };

    printf("== test_mxfp4 =============\n");
    float LO = -10.0f;
    float HI = 10.0f;
    for(int i = 0; i < 32; i++) {
        src[i] = LO + static_cast <float> (rand()) /(static_cast <float> (RAND_MAX/(HI-LO)));
    }
    test_accuracy();
    float e2m1s[] = {
        0.0f,   0.5f,   1.0f,   1.5f,   2.0f,   3.0f,   4.0f,   6.0f,
        -0.0f,  -0.5f, -1.0f,  -1.5f,  -2.0f,  -3.0f,  -4.0f,  -6.0f
    };
    auto e2m1s_cnt = sizeof(e2m1s)/sizeof(e2m1s[0]);
    for(int i = 0; i < 32; i++) src[i] = e2m1s[i%e2m1s_cnt];
    test_accuracy();

    for(int i = 0; i < 32; i++) src[i] = e2m1s[i%e2m1s_cnt] * 0.125f;
    test_accuracy();

    for(int i = 0; i < 32; i++) src[i] = e2m1s[i%e2m1s_cnt] * 128.0f;
    test_accuracy();

    return 0;
}();


#if 0
/**************************
now e8m0_to_float/e2m1_to_float can be used as reference to test jit optimization
***************************/
static int8_t get_u4(const uint8_t& val, bool high) {
    return high ? (val >> 4) : (val & 0xF);
}

struct e8m0 {
    uint8_t value;
    operator float() {
        return e8m0_to_float(value);
    }
};

struct e2m1_x2 {
    uint8_t value;
    e2m1_x2(uint8_t x) : value(x) {}
    float operator[](bool high_part) {
        return mxformat::e2m1_to_float(high_part);
    }
};

void MatMul(int M, int K, int N, int kGroupSize,
            float * psrc,
            e2m1_x2 * pwei, bool weightsNonTransposed,
            e8m0 * pscales,
            float * pdst) {
    auto kGroups = K / kGroupSize;
    for (size_t m = 0; m < M; m++) {
        for (size_t n = 0; n < N; n++) {
            size_t dstIdx = m * N + n;
            pdst[dstIdx] = 0.f;

            for (size_t kb = 0; kb < kGroups; kb++) {
                size_t scalesIdx = weightsNonTransposed ? kb * N + n : n * kGroups + kb;
                auto fscale = static_cast<float>(pscales[scalesIdx]);

                for (size_t ki = 0; ki < kGroupSize; ki++) {
                    auto k = kb * kGroupSize + ki;
                    size_t srcIdx = m * K + k;
                    size_t weiIdx = weightsNonTransposed ? k * N + n : n * K + k;

                    auto fwei = static_cast<float>(pwei[weiIdx / 2][weiIdx % 2]);
                    pdst[dstIdx] += psrc[srcIdx] * (fwei * fscale);
                }
            }
        }
    }
}

void simple_test() {
    constexpr int M = 16;
    constexpr int K = 64;
    constexpr int N = 64;

    std::vector<float> src(M*K, 0);
    for(int m = 0; m < M; m++)
        for(int k = 0; k < K; k++)
            src[m*K + k] = m;

    std::vector<e2m1_x2> wei_e2m1(M*K/2, 0);
    std::vector<float> wei(M*K, 0);
    for(int k = 0; k < K; k++)
        for(int n = 0; n < N; n++) {
            wei[k*N + n] = e2m1_to_float(n & 0xF);
        }

}
#endif



int main() {
    return 0;
}
