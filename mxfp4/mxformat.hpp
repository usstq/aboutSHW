#pragma once

#include <limits>
#include <cstdint>
#include <cmath>
#include <cstdio>

namespace mxformat {

inline float e8m0_to_float(uint8_t bits) {
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

inline uint8_t float_to_e8m0(float x) {
    // float32 has same exponent encoding as e8m0 (except sub-normal)
    return ((reinterpret_cast<uint32_t&>(x) >> 23) & 0xFF);
}

// E2M1 to float can be done with LUT easily (no test case is needed)
inline float e2m1_to_float(uint8_t bits, bool high_4_bit=false) {
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

inline uint8_t float_to_e2m1(float f) {
    uint8_t sign_off = f < 0 ? 8 : 0;

    // clamp
    if (f >= 6.0 || f <= -6.0)
        return sign_off + 7;

    // subnormal of e2m1
    //   here we drop -0.0f since it causes additional difficulty
    //   to decompression algo (exponent-adding with zero-guard)
    if (f <= 0.25f && f >= -0.25f)
        return 0;

    if (f < 0.75f && f > -0.75f)
        return sign_off + 1;

    if (f <= 1.25f && f >= -1.25f)
        return sign_off + (1<<1);

    // e2m1 number set is subset of float32
    auto v = reinterpret_cast<uint32_t&>(f);
    auto exponent = static_cast<int>((v >> 23) & 0xFF) - 127;
    auto E = static_cast<uint8_t>(exponent + 1) << 1;
    if (exponent < -1 || exponent > 2) {
        printf("Error: float_to_e2m1(%f) got E=%d\n", f, E);
        abort();
    }
    // valid exponent in e2m1 : -1, 0, 1, 2
    auto mantissa = v & ((1<<23) - 1);
    auto point1 = (0x1<<21);
    auto point3 = (0x3<<21);

    //printf("\t f=%f v=%x mantissa=%x exponent=%d point1=%x point3=%x sign_off=%x E=%d\n",
    //        f, v, mantissa, exponent, point1, point3, sign_off, E);
    if (mantissa <= point1) return sign_off + E;
    if (mantissa < point3) return sign_off + E + 1;
    return sign_off + E + (1<<1);
}

struct e2m1 {
    uint8_t bits;
    e2m1(float f) {
        bits = float_to_e2m1(f);
    }
    operator float() {
        return e2m1_to_float(bits);
    }
    operator int() {
        return bits;
    }
};

struct mxfp4 {
    // block size 32
    uint8_t scale_e8m0;
    uint8_t element_e2m1[16];

    // 32 floats will be converted into a single mxfp4
    // according to 6.3 of OCP MX spec.
    mxfp4() = default;
    mxfp4(float * src) {
        assign(src);
    }
    void assign(float * src) {
        float max_abs_v = std::fabs(src[0]);
        for(int i = 1; i < 32; i++) {
            auto abs_v = std::fabs(src[i]);
            if (max_abs_v < abs_v)
                max_abs_v = abs_v;
        }
        float X = 1.0f;
        // largest power-of-two1 less than or equal to max_abs_v
        if (max_abs_v == 0) {
            scale_e8m0 = 127;
            X = 1.0f;
        } else {
            float p = 1.0f;
            while(p <= max_abs_v) p *= 2.0f;
            while(p > max_abs_v) p *= 0.5f;

            float largest_pow_of_2_src = p;
            float largest_pow_of_2_e2m1 = 4.0f;
            X = largest_pow_of_2_src / largest_pow_of_2_e2m1;

            // E8M0 is an unsigned representation of a conventional biased Float32 exponent
            scale_e8m0 = ((reinterpret_cast<uint32_t&>(X) >> 23) & 0xFF);
        }
        if (scale_e8m0 == 0) {
            printf("Subnormal scale is met, unexpected!  max_abs_v = %f \n", max_abs_v);
            abort();
        }
        for(int i = 0; i < 32; i+=2) {
            float pi_0 = src[i] / X;
            float pi_1 = src[i+1] / X;
            auto e2m1_0 = float_to_e2m1(pi_0);
            auto e2m1_1 = float_to_e2m1(pi_1);
            //auto pi_actual = e2m1_to_float(e2m1);
            element_e2m1[i/2] = (e2m1_1 << 4) | (e2m1_0);
        }
    }

    void show() {
        printf("scale: %f(0x%x) elements:\n", e8m0_to_float(scale_e8m0), scale_e8m0);
        for(int i = 0; i < 32; i++) {
            uint8_t e2m1 = get_e2m1(i);
            printf("     [%2d] %4.1f(0x%x) => %10.5f", i, e2m1_to_float(e2m1), e2m1, (*this)[i]);
            if ((i & 3) == 3) printf("\n");
        }
    }

    uint8_t get_e2m1(int i) {
        uint8_t e2m1 = element_e2m1[i/2];
        return (i&1) ? (e2m1 >> 4) : (e2m1 & 0xF);
    }

    float operator[](int i) {
        // following algorithm is suitable for SIMD optimization
        float element = e2m1_to_float(element_e2m1[i/2], i & 1);
        float result = element;
        if (element != 0.0f) {
            // combine exponent by sum is valid since sub-normals of e2m1 has been converted to normal FP32
            reinterpret_cast<uint32_t&>(result) += static_cast<uint32_t>(scale_e8m0 - 127) << 23;
        }

        // double check correctness
        auto expected = e8m0_to_float(scale_e8m0) * element;
        if (result != expected) {
            printf("Error: mxfp4 convert failed\n");
        }
        return result;
    }
};
};
