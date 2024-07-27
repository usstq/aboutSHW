
#include <limits>
#include <cstdint>
#include <sstream>
#include <iostream>
#include <cmath>
#include <vector>

//#include "../perf/linux_perf.hpp"
//#include "../thirdparty/xbyak/xbyak/xbyak.h"

#define ASSERT(cond) if (!(cond)) {\
    std::stringstream ss; \
    ss << __FILE__ << ":" << __LINE__ << " " << #cond << " failed!"; \
    throw std::runtime_error(ss.str()); \
}

// E2M1 to float can be done with LUT easily (no test case is needed)
float e2m1_to_float(uint8_t bits, bool high_4_bit=false) {
    static constexpr float f4e2m1_to_f32_lut[16] = {
        0.0f,   0.5f,
        1.0f,   1.5f,
        2.0f,   3.0f,
        4.0f,   6.0f,
        -0.0f,  -0.5f,
        -1.0f,  -1.5f,
        -2.0f,  -3.0f,
        -4.0f,  -6.0f};
    return f4e2m1_to_f32_lut[high_4_bit ? (bits >> 4) : (bits & 0x0F)];
}

uint8_t float_to_e2m1(float f) {
    uint8_t sign_off = f < 0 ? 8 : 0;

    // clamp
    if (f >= 6.0 || f <= -6.0)
        return sign_off + 7;

    // subnormal of e2m1
    if (f <= 0.25f && f >= -0.25f)
        return sign_off + 0;

    if (f <= 0.5f && f >= -0.5f)
        return sign_off + 1;

    // e2m1 number set is subset of float32
    auto v = reinterpret_cast<uint32_t&>(f);
    auto exponent = static_cast<int>((v >> 23) & 0xFF) - 127;
    auto E = static_cast<uint8_t>(exponent + 1) << 1;
    // valid exponent in e2m1 : -1, 0, 1, 2
    auto mantissa = v & ((1<<23) - 1);
    auto point1 = (0x1<<21);
    auto point3 = (0x3<<21);

    //printf("\t v=%x mantissa=%x\n", v, mantissa);
    if (mantissa <= point1) return sign_off + E;
    if (mantissa < point3) return sign_off + E + 1;
    return sign_off + E + (1<<1);
}

int test_f32_to_e2m1 = []() {
    float src1[] = {
        0.0f,   0.5f,   1.0f,   1.5f,   2.0f,   3.0f,   4.0f,   6.0f,
        -0.0f,  -0.5f, -1.0f,  -1.5f,  -2.0f,  -3.0f,  -4.0f,  -6.0f
    };
    for(int i = 0; i<sizeof(src1)/sizeof(src1[0]); i++) {
        auto fp4 = float_to_e2m1(src1[i]);
        auto recover = e2m1_to_float(fp4);
        if(recover != src1[i]) {
            printf("%6.4f to 0x%x (e2m1) and back to %6.4f is unexpected!\n",
                    src1[i], static_cast<int>(fp4), recover);
        }
    }
    printf("test_f32_to_e2m1 is done\n");
    return 0;
}();

float e8m0_to_float(uint8_t bits) {
    constexpr uint8_t f32_mantissa_bits{23u};
    constexpr uint8_t e8m0_NaN = 0xff;

    if (bits == e8m0_NaN) {
        return std::numeric_limits<float>::quiet_NaN();
    } else if (bits == 0) {
        // this is the only special value needs to use subnormal to express
        return std::numeric_limits<float>::min() / 2;
    } else {
        int value = bits;
        value = (value) << f32_mantissa_bits;
        return reinterpret_cast<float&>(value);
    }
}

int test_e8m0 = []() {
    constexpr uint8_t e8m0_bias = 127;
    {
        float expect = 1.0f;
        for(int i = 0; i <= 127; i++) {
            auto got = e8m0_to_float(e8m0_bias + i);
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
            auto got = e8m0_to_float(e8m0_bias + i);
            if (got != expect) {
                std::cout << "[" << i << "] " << got << " != " << expect << std::endl;
                abort();
            }
            expect *= 0.5f;
        }
    }
    {
        auto got = e8m0_to_float(255);
        if (!std::isnan(got)) {
            std::cout << "expect [Nan] got " << got << std::endl;
            abort();
        }
    }
    std::cout << "test_e8m0 is done\n";
    return 0;
}();

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
        return e2m1_to_float(high_part);
    }
};


struct mxfp4 {
    // block size 32
    uint8_t scale_e8m0;
    uint8_t element_e2m1[16];

    // 32 floats will be converted into a single mxfp4
    // according to 6.3 of OCP MX spec.
    mxfp4(float * src) {
        float max_abs_v = std::fabs(src[0]);
        for(int i = 1; i < 32; i++) {
            auto abs_v = std::fabs(src[i]);
            if (max_abs_v < abs_v)
                max_abs_v = abs_v;
        }
        // largest power-of-two1 less than or equal to max_abs_v
        float p = 1.0f;
        while(p <= max_abs_v) p *= 2.0f;
        while(p > max_abs_v) p *= 0.5f;

        float largest_pow_of_2_src = p;
        float largest_pow_of_2_e2m1 = 4.0f;
        float X = largest_pow_of_2_src / largest_pow_of_2_e2m1;
        std::cout << "X=" << X  << std::endl;
        for(int i = 0; i < 32; i++) {
            float pi = src[i] / X;
            if (pi > 6.0f) pi = 6.0f;
            if (pi < -6.0f) pi = -6.0f;
            // https://stackoverflow.com/questions/12103514/how-do-you-round-off-decimal-places-in-c
            std::rint(pi);
        }
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

int main() {
    return 0;
}
