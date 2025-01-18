import pycpp
import numpy as np

@pycpp.clib("-march=core-avx2 -g")
def mylib():
    return r'''
#define USE_DEBUG_LOG 1
#include "common.hpp"

// decompress 
inline __m256 decomp_ymm_i4_0(__m256i vzi, __m256i vones, __m256i vmask) {
    return _mm256_castsi256_ps(_mm256_or_si256(vones, _mm256_and_si256(_mm256_slli_epi32(vzi, 23-4), vmask)));
}
inline __m256 decomp_ymm_i4_1(__m256i vzi, __m256i vones, __m256i vmask) {
    return _mm256_castsi256_ps(_mm256_or_si256(vones, _mm256_and_si256(_mm256_slli_epi32(vzi, 23-4*2), vmask)));
}
inline __m256 decomp_ymm_i4_2(__m256i vzi, __m256i vones, __m256i vmask) {
    return _mm256_castsi256_ps(_mm256_or_si256(vones, _mm256_and_si256(_mm256_slli_epi32(vzi, 23-4*3), vmask)));
}
inline __m256 decomp_ymm_i4_3(__m256i vzi, __m256i vones, __m256i vmask) {
    return _mm256_castsi256_ps(_mm256_or_si256(vones, _mm256_and_si256(_mm256_slli_epi32(vzi, 23-4*4), vmask)));
}
inline __m256 decomp_ymm_i4_4(__m256i vzi, __m256i vones, __m256i vmask) {
    return _mm256_castsi256_ps(_mm256_or_si256(vones, _mm256_and_si256(_mm256_slli_epi32(vzi, 23-4*5), vmask)));
}
inline __m256 decomp_ymm_i4_5(__m256i vzi, __m256i vones, __m256i vmask) {
    return _mm256_castsi256_ps(_mm256_or_si256(vones, _mm256_and_si256(_mm256_srli_epi32(vzi, 4*6-23), vmask)));
}
inline __m256 decomp_ymm_i4_6(__m256i vzi, __m256i vones, __m256i vmask) {
    return _mm256_castsi256_ps(_mm256_or_si256(vones, _mm256_and_si256(_mm256_srli_epi32(vzi, 4*7-23), vmask)));
}
inline __m256 decomp_ymm_i4_7(__m256i vzi, __m256i vones, __m256i vmask) {
    return _mm256_castsi256_ps(_mm256_or_si256(vones, _mm256_and_si256(_mm256_srli_epi32(vzi, 4*8-23), vmask)));
}

inline __m256 decomp_ymm_i8_0(__m256i vzi, __m256i vones, __m256i vmask) {
    return _mm256_castsi256_ps(_mm256_or_si256(vones, _mm256_and_si256(_mm256_slli_epi32(vzi, 23-8), vmask)));
}
inline __m256 decomp_ymm_i8_1(__m256i vzi, __m256i vones, __m256i vmask) {
    return _mm256_castsi256_ps(_mm256_or_si256(vones, _mm256_and_si256(_mm256_slli_epi32(vzi, 23-8*2), vmask)));
}
inline __m256 decomp_ymm_i8_2(__m256i vzi, __m256i vones, __m256i vmask) {
    return _mm256_castsi256_ps(_mm256_or_si256(vones, _mm256_and_si256(_mm256_srli_epi32(vzi, 8*3-23), vmask)));
}
inline __m256 decomp_ymm_i8_3(__m256i vzi, __m256i vones, __m256i vmask) {
    return _mm256_castsi256_ps(_mm256_or_si256(vones, _mm256_and_si256(_mm256_srli_epi32(vzi, 8*4-23), vmask)));
}

std::ostream& operator<<(std::ostream& os, const __m256& v) {
    float vec[8];
    _mm256_storeu_ps(vec, v);
    os << "__m256{";
    for(int i = 0; i < 8; i++)
        os << vec[i] << ",";
    os << "}";
    return os;
}

std::ostream& operator<<(std::ostream& os, const __m256i& v) {
    uint32_t vec[8];
    _mm256_storeu_si256((__m256i *)vec, v);
    os << "__m256i{" << std::hex;
    for(int i = 0; i < 8; i++)
        os << "0x" << vec[i] << ",";
    os << "}" << std::dec;
    return os;
}
 

struct WeightArg {
    uint32_t* ptr = nullptr;   // 1xf32/2xf16/4xi8/8xi4
    uint32_t* zps = nullptr;   // 4-i8 or 8-i4 packed into a int32
    float* scales = nullptr;  // scales
    int stride;                     // strides in number of elements
    int IC;
    int OC;
    int ic_group_size;  // 64/128 for i4 case (<=0 means no group in IC, only per-OC quantization is used)
    enum class CompressType { F32, F16, I8S, I8A, I4S, I4A } type;

    void show() {
        const int ele_bits = (type == CompressType::I8S || type == CompressType::I8A) ? 8 : 4;
        const bool symmetric = (type == CompressType::I8S || type == CompressType::I4S);
        const int ele_packed_in_dword = ele_bits == 8 ? 4 : 8;
        
        auto * q = ptr;
        auto * z = zps;
        float * s = scales;
        for (int ic0 = 0; ic0 < IC; ic0 += ic_group_size, z += OC/ele_packed_in_dword, s += OC, q += ic_group_size*OC/ele_packed_in_dword) {
            
            if (!symmetric) {
                auto * src = z;
                std::cout << std::hex << "zps:\t";
                for (int oc = 0; oc < OC; oc+=ele_packed_in_dword) {
                    std::cout << *src++ << ",";
                }
                std::cout << std::endl;
            }
            {
                std::cout << std::hex << "scales:\t";
                for (int oc = 0; oc < OC; oc++) {
                    std::cout << s[oc] << ",";
                }
                std::cout << std::endl;
            }

            for(int ic = 0; ic < ic_group_size; ic++) {
                std::cout << std::hex << "ic:" << ic << "\t";
                auto* src = q + ic*OC/ele_packed_in_dword;
                for (int oc = 0; oc < OC; oc+=ele_packed_in_dword) {
                    std::cout << *src++ << ",";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    }

    ~WeightArg() {
        if (ptr)
            ::free(ptr);
        if (zps)
            ::free(zps);
        if (scales)
            ::free(scales);
    }

    void decomp(float * dst) {
        ASSERT(type != CompressType::F32);
        switch (type) {
            case CompressType::F16: {
                auto* src = reinterpret_cast<int16_t*>(ptr);
                for (int i = 0; i < IC * OC; i += 8) {
                    auto vf16 = _mm_loadu_si128((__m128i const*)(src + i));
                    auto vf32 = _mm256_cvtph_ps(vf16);
                    _mm256_storeu_ps(dst + i, vf32);
                }
            } break;
            case CompressType::I8S:
            case CompressType::I8A:
            case CompressType::I4S:
            case CompressType::I4A: {
                const int ele_bits = (type == CompressType::I8S || type == CompressType::I8A) ? 8 : 4;
                const bool symmetric = (type == CompressType::I8S || type == CompressType::I4S);
                const int ele_packed_in_dword = ele_bits == 8 ? 4 : 8;
                // decompress row by row
                std::vector<float> zps_decomp;
                zps_decomp.resize(OC);
                auto* cur_zps = zps;
                auto* cur_scale = scales;
                auto* cur_q = ptr;
                for(int ic0 = 0; ic0 < IC; ic0 += ic_group_size, cur_zps += OC/ele_packed_in_dword, cur_scale += OC, dst += ic_group_size * OC, cur_q += ic_group_size*OC/ele_packed_in_dword) {
                    auto vmask_i4 = _mm256_set1_epi32(0x0000000F << (23-4));
                    auto vmask_i8 = _mm256_set1_epi32(0x000000FF << (23-8));
                    auto vones = _mm256_castps_si256(_mm256_set1_ps(1.0f));

                    // decompress zp into float values: [(1 + zp/16)*(16s)]
                    if (!symmetric) {
                        auto* pz_dst = zps_decomp.data();
                        if (ele_bits == 8) {
                            for(int oc = 0; oc < OC; oc += 4*8) {
                                auto vpacked = _mm256_loadu_si256((__m256i const *)(cur_zps + oc/4));

                                auto s0 = _mm256_loadu_ps(cur_scale + oc + 8*0);
                                auto s1 = _mm256_loadu_ps(cur_scale + oc + 8*1);
                                auto s2 = _mm256_loadu_ps(cur_scale + oc + 8*2);
                                auto s3 = _mm256_loadu_ps(cur_scale + oc + 8*3);
                                
                                s0 = _mm256_mul_ps(s0, decomp_ymm_i8_0(vpacked, vones, vmask_i8));
                                s1 = _mm256_mul_ps(s1, decomp_ymm_i8_1(vpacked, vones, vmask_i8));
                                s2 = _mm256_mul_ps(s2, decomp_ymm_i8_2(vpacked, vones, vmask_i8));
                                s3 = _mm256_mul_ps(s3, decomp_ymm_i8_3(vpacked, vones, vmask_i8));

                                _mm256_storeu_ps(pz_dst + oc + 8*0, s0);
                                _mm256_storeu_ps(pz_dst + oc + 8*1, s1);
                                _mm256_storeu_ps(pz_dst + oc + 8*2, s2);
                                _mm256_storeu_ps(pz_dst + oc + 8*3, s3);
                            }
                        } else {
                            for(int oc = 0; oc < OC; oc += 8*8) {
                                auto vpacked = _mm256_loadu_si256((__m256i const *)(cur_zps + oc/8));
                                
                                auto s0 = _mm256_loadu_ps(cur_scale + oc + 8*0);
                                auto s1 = _mm256_loadu_ps(cur_scale + oc + 8*1);
                                auto s2 = _mm256_loadu_ps(cur_scale + oc + 8*2);
                                auto s3 = _mm256_loadu_ps(cur_scale + oc + 8*3);
                                auto s4 = _mm256_loadu_ps(cur_scale + oc + 8*4);
                                auto s5 = _mm256_loadu_ps(cur_scale + oc + 8*5);
                                auto s6 = _mm256_loadu_ps(cur_scale + oc + 8*6);
                                auto s7 = _mm256_loadu_ps(cur_scale + oc + 8*7);

                                s0 = _mm256_mul_ps(s0, decomp_ymm_i4_0(vpacked, vones, vmask_i4));
                                s1 = _mm256_mul_ps(s1, decomp_ymm_i4_1(vpacked, vones, vmask_i4));
                                s2 = _mm256_mul_ps(s2, decomp_ymm_i4_2(vpacked, vones, vmask_i4));
                                s3 = _mm256_mul_ps(s3, decomp_ymm_i4_3(vpacked, vones, vmask_i4));
                                s4 = _mm256_mul_ps(s4, decomp_ymm_i4_4(vpacked, vones, vmask_i4));
                                s5 = _mm256_mul_ps(s5, decomp_ymm_i4_5(vpacked, vones, vmask_i4));
                                s6 = _mm256_mul_ps(s6, decomp_ymm_i4_6(vpacked, vones, vmask_i4));
                                s7 = _mm256_mul_ps(s7, decomp_ymm_i4_7(vpacked, vones, vmask_i4));
                                                            
                                _mm256_storeu_ps(pz_dst + oc + 8*0, s0);
                                _mm256_storeu_ps(pz_dst + oc + 8*1, s1);
                                _mm256_storeu_ps(pz_dst + oc + 8*2, s2);
                                _mm256_storeu_ps(pz_dst + oc + 8*3, s3);
                                _mm256_storeu_ps(pz_dst + oc + 8*4, s4);
                                _mm256_storeu_ps(pz_dst + oc + 8*5, s5);
                                _mm256_storeu_ps(pz_dst + oc + 8*6, s6);
                                _mm256_storeu_ps(pz_dst + oc + 8*7, s7);
                            }
                        }
                    }

                    // decompress each row of quantized weight into:
                    //  (1 + q/16)*(16s) - [(1 + zp/16)*(16s)] => (q - zp)*s
                    //
                    // for zp==0 case: (1 + q/16)*(16s) - [16s] => q*s
                    //
                    if (ele_bits == 8) {
                        auto* pz_dst = zps_decomp.data();
                        for(int oc = 0; oc < OC; oc += 4*8) {
                            __m256 zp0, zp1, zp2, zp3;
                            auto s0 = _mm256_loadu_ps(cur_scale + oc + 8*0);
                            auto s1 = _mm256_loadu_ps(cur_scale + oc + 8*1);
                            auto s2 = _mm256_loadu_ps(cur_scale + oc + 8*2);
                            auto s3 = _mm256_loadu_ps(cur_scale + oc + 8*3);
                            if (!symmetric) {
                                zp0 = _mm256_loadu_ps(pz_dst + oc + 8*0);
                                zp1 = _mm256_loadu_ps(pz_dst + oc + 8*1);
                                zp2 = _mm256_loadu_ps(pz_dst + oc + 8*2);
                                zp3 = _mm256_loadu_ps(pz_dst + oc + 8*3);
                            } else {
                                // in symmetric case, just subtract (16*s) out-of the (1+v/16)*(16s)
                                auto zero_point = _mm256_set1_ps(1 + 128.0f/256);
                                
                                zp0 = _mm256_mul_ps(s0, zero_point);
                                zp1 = _mm256_mul_ps(s1, zero_point);
                                zp2 = _mm256_mul_ps(s2, zero_point);
                                zp3 = _mm256_mul_ps(s3, zero_point);
                            }
                            auto* pdst = dst + oc;
                            auto* pq = cur_q + oc/ele_packed_in_dword;
                            for(int ic = 0; ic < ic_group_size; ic++, pdst += OC, pq += OC/ele_packed_in_dword) {
                                auto vpacked = _mm256_loadu_si256((__m256i const *)(pq));
                                auto y0 = decomp_ymm_i8_0(vpacked, vones, vmask_i8);
                                auto y1 = decomp_ymm_i8_1(vpacked, vones, vmask_i8);
                                auto y2 = decomp_ymm_i8_2(vpacked, vones, vmask_i8);
                                auto y3 = decomp_ymm_i8_3(vpacked, vones, vmask_i8);

                                y0 = _mm256_fmsub_ps(y0, s0, zp0);
                                y1 = _mm256_fmsub_ps(y1, s1, zp1);
                                y2 = _mm256_fmsub_ps(y2, s2, zp2);
                                y3 = _mm256_fmsub_ps(y3, s3, zp3);

                                _mm256_storeu_ps(pdst + 8*0, y0);
                                _mm256_storeu_ps(pdst + 8*1, y1);
                                _mm256_storeu_ps(pdst + 8*2, y2);
                                _mm256_storeu_ps(pdst + 8*3, y3);
                            }
                        }
                    } else {
                        // INT4
                    }
                }
            } break;
        }
    }

    WeightArg(const float* org, int _IC, int _OC, CompressType _type, int _ic_group_size = -1, float* fakequant = nullptr)
        : IC(_IC),
          OC(_OC),
          type(_type),
          ic_group_size(_ic_group_size) {
        ASSERT((OC % 8) == 0);
        if (ic_group_size <= 0) {
            // per-OC
            ic_group_size = IC;
        }
        ASSERT((IC % ic_group_size) == 0);

        stride = OC;

        switch (type) {
        case CompressType::F32: {
            ptr = reinterpret_cast<uint32_t*>(aligned_alloc(64, IC * OC * sizeof(uint32_t)));
            memcpy(ptr, org, IC * OC * sizeof(uint32_t));
        } break;
        case CompressType::F16: {
            ptr = reinterpret_cast<uint32_t*>(aligned_alloc(64, IC * OC * sizeof(uint32_t) / 2));
            auto* dst = reinterpret_cast<int16_t*>(ptr);
            for (int i = 0; i < IC * OC; i += 8) {
                auto vf32 = _mm256_loadu_ps(org + i);
                auto vf16 = _mm256_cvtps_ph(vf32, _MM_FROUND_TO_NEAREST_INT);
                _mm_storeu_si128((__m128i*)(dst + i), vf16);
            }
        } break;
        case CompressType::I8S:
        case CompressType::I8A:
        case CompressType::I4S:
        case CompressType::I4A: {
            // 4x8 weights are packed together in following way
            // so they can be decompressed easily into f32
            // 0 | 8 | . | .  : dword0
            // 1 | 9 | . | .  : dword1
            // 2 | a | . | .  : dword2
            // 3 | b | . | .  : dword3
            // 4 | c | . | .  : dword4
            // 5 | d | . | .  : dword5
            // 6 | e | . | .  : dword6
            // 7 | f | . | .  : dword7
            const int ele_bits = (type == CompressType::I8S || type == CompressType::I8A) ? 8 : 4;
            const bool symmetric = (type == CompressType::I8S || type == CompressType::I4S);
            const int ele_packed_in_dword = ele_bits == 8 ? 4 : 8;

            // considering the possible sparsity of input channels, we shouldn't pack elements from different IC into same i32
            // elements packed into an avx2 YMM register in unit of 4x8xi8 or 8x8xi4
            auto pack_ele = [&](uint32_t* dst, int oc, int32_t value) {
                if (ele_bits == 8) {
                    dst += oc/(4*8)*8;
                    auto off = (oc % (4*8));
                    dst += (off % 8);
                    auto b_off = (off / 8) * 8;
                    uint32_t mask = ~(0xFF << b_off);
                    (*dst) = ((*dst) & mask) | ((value & 0xFF) << b_off);
                } else {
                    dst += oc/(8*8)*8;
                    auto off = (oc % (8*8));
                    dst += (off % 8);
                    auto b_off = (off / 8) * 4;
                    uint32_t mask = ~(0xF << b_off);
                    (*dst) = ((*dst) & mask) | ((value & 0xF) << b_off);
                }
            };
            ptr = reinterpret_cast<uint32_t*>(aligned_alloc(64, IC * OC * sizeof(uint32_t) / ele_packed_in_dword));
            scales = reinterpret_cast<float*>(aligned_alloc(64, (IC / ic_group_size) * OC * sizeof(float)));
            if (!symmetric) {
                zps = reinterpret_cast<uint32_t*>(aligned_alloc(64, (IC / ic_group_size) * OC * sizeof(uint32_t) / ele_packed_in_dword));
            }
            
            auto* dst_scales = scales;
            auto* dst_zp = zps;
            for (int ic0 = 0; ic0 < IC; ic0 += ic_group_size, dst_scales += OC, dst_zp += (OC / ele_packed_in_dword)) {
                for (int oc = 0; oc < OC; oc++) {
                    // per-group quantization
                    auto ic1 = std::min(ic0 + ic_group_size, IC);
                    auto* src0 = org + oc + ic0 * stride;
                    auto* src = src0;
                    float v0 = src[0];
                    float v1 = src[0];
                    for (int ic = ic0; ic < ic1; ic++, src += stride) {
                        v0 = std::min(*src, v0);
                        v1 = std::max(*src, v1);
                    }
                    // find scales & zps
                    float s = 0;
                    float zp = 0;
                    float q0 = (ele_bits == 8) ? (0) : (0);
                    float q1 = (ele_bits == 8) ? (255) : (15);
                    if (symmetric) {
                        // v0/s >= (q0 - zp)
                        // v1/s <= (q1 - zp)
                        zp = (ele_bits == 8 ? 128 : 8);
                        auto vmax = std::max(std::abs(v0), std::abs(v1));
                        s = vmax / (q0 - zp);
                        if (vmax == std::abs(v0) && v0 > 0)
                            s = -s;
                        else if (vmax == std::abs(v1) && v1 > 0)
                            s = -s;
                        dst_scales[oc] = s*(ele_bits == 8 ? 256 : 16);  // multiply 256/16 due to special decomp methods
                    } else {
                        //  v0 = (q0 - zp)*s
                        //  v1 = (q1 - zp)*s
                        //
                        s = (q1 - q0) / (v1 - v0);
                        zp = (q0 * v1 - q1 * v0) / (v1 - v0);
                        // s = 1.0; zp = 128;
                        if (zp < q0) zp = q0;
                        if (zp > q1) zp = q1;
                        dst_scales[oc] = s*(ele_bits == 8 ? 256 : 16);  // multiply 256/16 due to special decomp methods
                        pack_ele(dst_zp, oc, zp);
                    }

                    // do quantization
                    auto ds = 1.0f / s;
                    src = src0;
                    auto* fq_dst = fakequant ? (fakequant + oc + ic0 * stride) : (nullptr);
                    for (int ic = ic0; ic < ic1; ic++, src += stride) {
                        auto q = static_cast<int>(std::roundf((*src) * ds + zp));
                        if (q < q0)
                            q = q0;
                        if (q > q1)
                            q = q1;
                        // fake-quantize
                        if (fq_dst) {
                            *fq_dst = (q - zp) * s;
                            fq_dst += stride;
                        }
                        // pack quantized value into int32
                        pack_ele(ptr + ic * OC / ele_packed_in_dword, oc, q);
                    }
                }
            }
        } break;
        }
    }
};

extern "C" WeightArg * setw(const float* org, int IC, int OC, int ic_group_size, float* fakequant) {
    DEBUG_LOG(org, IC, OC, ic_group_size, fakequant);
    auto * pw = new WeightArg(org, IC, OC, WeightArg::CompressType::I8A, ic_group_size, fakequant);
    //pw->decomp(fakequant);
    //pw->show();
    return pw;
}

extern "C" void decomp(WeightArg * pw, float* fakequant) {
    pw->decomp(fakequant);
}
'''


IC = 32
OC = 32
A = np.random.rand(IC, OC).astype(np.float32)
low = -128
high = 128
A = np.random.randint(low, high, [IC, OC]).astype(np.float32)
A2 = np.zeros([IC, OC], dtype=np.float32)
A3 = np.zeros([IC, OC], dtype=np.float32)
group_size = 16
print("A=================")
print(A)

pw = mylib.setw(A, IC, OC, group_size, A2)

print("A2=================")
print(A2)
print("ALL-close: ", np.allclose(A, A2))
print("max diff:", abs(A2 - A).max())

mylib.decomp(pw, A3)
if not np.allclose(A2, A3):
    print("A3=================")
    print(A3)
    print("max diff:", abs(A3 - A).max())
else:
    print("A2 ~= A3")