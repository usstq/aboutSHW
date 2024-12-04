import numpy as np
import pycpp

@pycpp.clib("-march=core-avx2 -g")
def mylib():
    return r'''
#include "common.hpp"

/*
    org: [IC, OC] x float32
    w_quant : [IC, OC] x i8/i4
    zps     : [IC / ic_group_size, OC] x i8/i4
    scales  : [IC / ic_group_size, OC] x f32
*/

extern "C" void compress(float * org, int IC, int OC,
                    int ele_bits, bool symmetric, int ic_group_size,
                    uint32_t * w_quant, uint32_t * zps, float * scales,
                    float* fakequant) {
    const int ele_packed_in_dword = ele_bits == 8 ? 4 : 8;

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
    /*
    ptr = reinterpret_cast<uint32_t*>(aligned_alloc(64, IC * OC * sizeof(uint32_t) / ele_packed_in_dword));
    scales = reinterpret_cast<float*>(aligned_alloc(64, (IC / ic_group_size) * OC * sizeof(float)));
    if (!symmetric) {
        zps = reinterpret_cast<uint32_t*>(aligned_alloc(64, (IC / ic_group_size) * OC * sizeof(uint32_t) / ele_packed_in_dword));
    }
    */
    auto stride = OC;
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
            if (v1 == v0) {
                s = v1;
                zp = 0;
                dst_scales[oc] = s*(ele_bits == 8 ? 256 : 16);
                if (!symmetric) pack_ele(dst_zp, oc, zp);
            } else if (symmetric) {
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
                s = (v1 - v0)/(q1 - q0);
                zp = std::roundf((v1*q0 - v0*q1) / (v1 - v0));

                // zp has limited range, thus s must be fixed to fit the range
                if (zp < q0) {
                    // v1 > v0 > 0
                    zp = q0;
                    s = v1/(q1 - q0);
                } else if (zp > q1) {
                    // 0 > v1 > v0
                    zp = q1;
                    s = v0/(q0 - q1);
                }

                // s = 1.0; zp = 128;
                if (0) {
                    DEBUG_LOG(v0, v1, zp);
                }
                dst_scales[oc] = s*(ele_bits == 8 ? 256 : 16);  // multiply 256/16 due to special decomp methods
                pack_ele(dst_zp, oc, zp);
            }

            // do quantization
            auto ds = 1.0f / s;
            src = src0;
            auto* fq_dst = fakequant ? (fakequant + oc + ic0 * stride) : (nullptr);
            //float* fq_dst = nullptr;
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
                pack_ele(w_quant + ic * OC / ele_packed_in_dword, oc, q);
            }
        }
    }
}

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

extern "C" void decompress(float * dst, int IC, int OC,
                      int ele_bits, bool symmetric, int ic_group_size,
                      uint32_t * w_quant, uint32_t * zps, float * scales) {
    const int ele_packed_in_dword = ele_bits == 8 ? 4 : 8;
    // decompress row by row
    std::vector<float> zps_decomp;
    zps_decomp.resize(OC);
    auto* cur_zps = zps;
    auto* cur_scale = scales;
    auto* cur_q = w_quant;

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

                // decompress:  (q - zp)*s =>  (1 + q/256) *[256*s] - [(1 + zp/256) * 256*s]
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
            auto* pz_dst = zps_decomp.data();
            auto* pq = cur_q;
            auto* pdst = dst;
            // decompress:  (q - zp)*s =>  (1 + q/16) *[16*s] - [(1 + zp/16) * (16*s)]
            
            if (symmetric) {
                for(int ic = 0; ic < ic_group_size; ic++, pdst += OC) {
                    // each ymm has 8x8 i4
                    auto zero_point = _mm256_set1_ps(1 + 8.0f/16);
                    for(int oc = 0; oc < OC; oc += 8*8, pq+=8) {
                        auto vpacked = _mm256_loadu_si256((__m256i const *)(pq));
                        {
                            auto y0 = decomp_ymm_i4_0(vpacked, vones, vmask_i4);
                            auto y1 = decomp_ymm_i4_1(vpacked, vones, vmask_i4);
                            auto y2 = decomp_ymm_i4_2(vpacked, vones, vmask_i4);
                            auto y3 = decomp_ymm_i4_3(vpacked, vones, vmask_i4);
                            auto s0 = _mm256_loadu_ps(cur_scale + oc + 8*0);
                            auto s1 = _mm256_loadu_ps(cur_scale + oc + 8*1);
                            auto s2 = _mm256_loadu_ps(cur_scale + oc + 8*2);
                            auto s3 = _mm256_loadu_ps(cur_scale + oc + 8*3);
                            y0 = _mm256_sub_ps(y0, zero_point);
                            y1 = _mm256_sub_ps(y1, zero_point);
                            y2 = _mm256_sub_ps(y2, zero_point);
                            y3 = _mm256_sub_ps(y3, zero_point);
                            y0 = _mm256_mul_ps(y0, s0);
                            y1 = _mm256_mul_ps(y1, s1);
                            y2 = _mm256_mul_ps(y2, s2);
                            y3 = _mm256_mul_ps(y3, s3);
                            _mm256_storeu_ps(pdst + oc + 8*0, y0);
                            _mm256_storeu_ps(pdst + oc + 8*1, y1);
                            _mm256_storeu_ps(pdst + oc + 8*2, y2);
                            _mm256_storeu_ps(pdst + oc + 8*3, y3);
                        }
                        {
                            auto y0 = decomp_ymm_i4_4(vpacked, vones, vmask_i4);
                            auto y1 = decomp_ymm_i4_5(vpacked, vones, vmask_i4);
                            auto y2 = decomp_ymm_i4_6(vpacked, vones, vmask_i4);
                            auto y3 = decomp_ymm_i4_7(vpacked, vones, vmask_i4);
                            auto s0 = _mm256_loadu_ps(cur_scale + oc + 8*4);
                            auto s1 = _mm256_loadu_ps(cur_scale + oc + 8*5);
                            auto s2 = _mm256_loadu_ps(cur_scale + oc + 8*6);
                            auto s3 = _mm256_loadu_ps(cur_scale + oc + 8*7);
                            y0 = _mm256_sub_ps(y0, zero_point);
                            y1 = _mm256_sub_ps(y1, zero_point);
                            y2 = _mm256_sub_ps(y2, zero_point);
                            y3 = _mm256_sub_ps(y3, zero_point);
                            y0 = _mm256_mul_ps(y0, s0);
                            y1 = _mm256_mul_ps(y1, s1);
                            y2 = _mm256_mul_ps(y2, s2);
                            y3 = _mm256_mul_ps(y3, s3);
                            _mm256_storeu_ps(pdst + oc + 8*4, y0);
                            _mm256_storeu_ps(pdst + oc + 8*5, y1);
                            _mm256_storeu_ps(pdst + oc + 8*6, y2);
                            _mm256_storeu_ps(pdst + oc + 8*7, y3);
                        }
                    }
                }
            } else {
                for(int ic = 0; ic < ic_group_size; ic++, pdst += OC) {
                    // each ymm has 8x8 i4
                    for(int oc = 0; oc < OC; oc += 8*8, pq+=8) {
                        auto vpacked = _mm256_loadu_si256((__m256i const *)(pq));
                        {
                            auto y0 = decomp_ymm_i4_0(vpacked, vones, vmask_i4);
                            auto y1 = decomp_ymm_i4_1(vpacked, vones, vmask_i4);
                            auto y2 = decomp_ymm_i4_2(vpacked, vones, vmask_i4);
                            auto y3 = decomp_ymm_i4_3(vpacked, vones, vmask_i4);
                            auto s0 = _mm256_loadu_ps(cur_scale + oc + 8*0);
                            auto s1 = _mm256_loadu_ps(cur_scale + oc + 8*1);
                            auto s2 = _mm256_loadu_ps(cur_scale + oc + 8*2);
                            auto s3 = _mm256_loadu_ps(cur_scale + oc + 8*3);
                            auto zp0 = _mm256_loadu_ps(pz_dst + oc + 8*0);
                            auto zp1 = _mm256_loadu_ps(pz_dst + oc + 8*1);
                            auto zp2 = _mm256_loadu_ps(pz_dst + oc + 8*2);
                            auto zp3 = _mm256_loadu_ps(pz_dst + oc + 8*3);
                            y0 = _mm256_fmsub_ps(y0, s0, zp0);
                            y1 = _mm256_fmsub_ps(y1, s1, zp1);
                            y2 = _mm256_fmsub_ps(y2, s2, zp2);
                            y3 = _mm256_fmsub_ps(y3, s3, zp3);
                            _mm256_storeu_ps(pdst + oc + 8*0, y0);
                            _mm256_storeu_ps(pdst + oc + 8*1, y1);
                            _mm256_storeu_ps(pdst + oc + 8*2, y2);
                            _mm256_storeu_ps(pdst + oc + 8*3, y3);
                        }
                        {
                            auto y0 = decomp_ymm_i4_4(vpacked, vones, vmask_i4);
                            auto y1 = decomp_ymm_i4_5(vpacked, vones, vmask_i4);
                            auto y2 = decomp_ymm_i4_6(vpacked, vones, vmask_i4);
                            auto y3 = decomp_ymm_i4_7(vpacked, vones, vmask_i4);
                            auto s0 = _mm256_loadu_ps(cur_scale + oc + 8*4);
                            auto s1 = _mm256_loadu_ps(cur_scale + oc + 8*5);
                            auto s2 = _mm256_loadu_ps(cur_scale + oc + 8*6);
                            auto s3 = _mm256_loadu_ps(cur_scale + oc + 8*7);
                            auto zp0 = _mm256_loadu_ps(pz_dst + oc + 8*4);
                            auto zp1 = _mm256_loadu_ps(pz_dst + oc + 8*5);
                            auto zp2 = _mm256_loadu_ps(pz_dst + oc + 8*6);
                            auto zp3 = _mm256_loadu_ps(pz_dst + oc + 8*7);
                            y0 = _mm256_fmsub_ps(y0, s0, zp0);
                            y1 = _mm256_fmsub_ps(y1, s1, zp1);
                            y2 = _mm256_fmsub_ps(y2, s2, zp2);
                            y3 = _mm256_fmsub_ps(y3, s3, zp3);
                            _mm256_storeu_ps(pdst + oc + 8*4, y0);
                            _mm256_storeu_ps(pdst + oc + 8*5, y1);
                            _mm256_storeu_ps(pdst + oc + 8*6, y2);
                            _mm256_storeu_ps(pdst + oc + 8*7, y3);
                        }
                    }
                }
            }
        }
    }
}

extern "C" void decompress_CPI() {
    for(int ic0 = 0; ic0 < IC; ic0 += ic_group_size, 
            cur_zps += OC/ele_packed_in_dword,
            cur_scale += OC,
            dst += ic_group_size * OC,
            cur_q += ic_group_size*OC/ele_packed_in_dword) {
        for(int ic = 0; ic < ic_group_size; ic++, pdst += OC) {
            // each ymm has 8x8 i4
            for(int oc = 0; oc < OC; oc += 8*8, pq+=8) {
                auto vpacked = _mm256_loadu_si256((__m256i const *)(pq));
                {
                    auto y0 = decomp_ymm_i4_0(vpacked, vones, vmask_i4);
                    auto y1 = decomp_ymm_i4_1(vpacked, vones, vmask_i4);
                    auto y2 = decomp_ymm_i4_2(vpacked, vones, vmask_i4);
                    auto y3 = decomp_ymm_i4_3(vpacked, vones, vmask_i4);
                    auto s0 = _mm256_loadu_ps(cur_scale + oc + 8*0);
                    auto s1 = _mm256_loadu_ps(cur_scale + oc + 8*1);
                    auto s2 = _mm256_loadu_ps(cur_scale + oc + 8*2);
                    auto s3 = _mm256_loadu_ps(cur_scale + oc + 8*3);
                    auto zp0 = _mm256_loadu_ps(pz_dst + oc + 8*0);
                    auto zp1 = _mm256_loadu_ps(pz_dst + oc + 8*1);
                    auto zp2 = _mm256_loadu_ps(pz_dst + oc + 8*2);
                    auto zp3 = _mm256_loadu_ps(pz_dst + oc + 8*3);
                    y0 = _mm256_fmsub_ps(y0, s0, zp0);
                    y1 = _mm256_fmsub_ps(y1, s1, zp1);
                    y2 = _mm256_fmsub_ps(y2, s2, zp2);
                    y3 = _mm256_fmsub_ps(y3, s3, zp3);
                    _mm256_storeu_ps(pdst + oc + 8*0, y0);
                    _mm256_storeu_ps(pdst + oc + 8*1, y1);
                    _mm256_storeu_ps(pdst + oc + 8*2, y2);
                    _mm256_storeu_ps(pdst + oc + 8*3, y3);
                }
                {
                    auto y0 = decomp_ymm_i4_4(vpacked, vones, vmask_i4);
                    auto y1 = decomp_ymm_i4_5(vpacked, vones, vmask_i4);
                    auto y2 = decomp_ymm_i4_6(vpacked, vones, vmask_i4);
                    auto y3 = decomp_ymm_i4_7(vpacked, vones, vmask_i4);
                    auto s0 = _mm256_loadu_ps(cur_scale + oc + 8*4);
                    auto s1 = _mm256_loadu_ps(cur_scale + oc + 8*5);
                    auto s2 = _mm256_loadu_ps(cur_scale + oc + 8*6);
                    auto s3 = _mm256_loadu_ps(cur_scale + oc + 8*7);
                    auto zp0 = _mm256_loadu_ps(pz_dst + oc + 8*4);
                    auto zp1 = _mm256_loadu_ps(pz_dst + oc + 8*5);
                    auto zp2 = _mm256_loadu_ps(pz_dst + oc + 8*6);
                    auto zp3 = _mm256_loadu_ps(pz_dst + oc + 8*7);
                    y0 = _mm256_fmsub_ps(y0, s0, zp0);
                    y1 = _mm256_fmsub_ps(y1, s1, zp1);
                    y2 = _mm256_fmsub_ps(y2, s2, zp2);
                    y3 = _mm256_fmsub_ps(y3, s3, zp3);
                    _mm256_storeu_ps(pdst + oc + 8*4, y0);
                    _mm256_storeu_ps(pdst + oc + 8*5, y1);
                    _mm256_storeu_ps(pdst + oc + 8*6, y2);
                    _mm256_storeu_ps(pdst + oc + 8*7, y3);
                }
            }
        }
    }
}

extern "C" void decompress_perf(float * dst, int IC, int OC,
                      int ele_bits, bool symmetric, int ic_group_size,
                      uint32_t * w_quant, uint32_t * zps, float * scales,
                      int rounds) {
    for(int r = 0 ; r < rounds; r++) {
        decompress(dst, IC, OC,
                   ele_bits, symmetric, ic_group_size,
                   w_quant, zps, scales);
    }
}
'''

class Weight:
    def __init__(self, bits, symmetric, ic_group_size, weight, IC, OC):
        assert bits == 8 or bits == 4
        self.bits = bits
        self.symmetric = symmetric
        self.ic_group_size = ic_group_size
        self.ele_per_u32 = 4 if bits == 8 else 8
        self.w_quant = np.zeros([IC, OC//self.ele_per_u32], dtype=np.uint32)
        self.w_zps = np.zeros([IC//ic_group_size, OC//self.ele_per_u32], dtype=np.uint32)
        self.w_scales = np.zeros([IC//ic_group_size, OC], dtype=np.float32)
        self.IC = IC
        self.OC = OC
        self.fakequant = np.zeros([self.IC, self.OC], dtype=np.float32)
        mylib.compress(weight, IC, OC, bits, symmetric, ic_group_size, self.w_quant, self.w_zps, self.w_scales, self.fakequant)
    
    def decomp(self, ROUNDS=1):
        weight = np.zeros([self.IC, self.OC], dtype=np.float32)
        mylib.decompress_perf(weight, self.IC, self.OC, self.bits, self.symmetric, self.ic_group_size, self.w_quant, self.w_zps, self.w_scales, ROUNDS)
        return weight

perf = pycpp.perf(["HW_CPU_CYCLES", "SPLIT_LOADS=0xd0,0x41"])

def test(bits, symmetric, ic_group_size, IC, OC, resolution = 128):
    np.random.seed(0)
    A = np.random.rand(IC, OC).astype(np.float32)
    low = -resolution
    high = resolution
    A = np.random.randint(low, high, [IC, OC]).astype(np.float32)

    wq = Weight(bits, symmetric, ic_group_size, A, IC, OC)
    ROUNDS = 1000
    k_loops = IC * OC // (8*8 if bits == 4 else 8*4)

    with perf.verbose(f"decomp{IC}x{OC}", ROUNDS, k_loops):
        A2 = wq.decomp(ROUNDS)
    #A2 = wq.fakequant

    if not np.allclose(A2, A):
        #print("A2=================")
        #print(A2)
        adiff = abs(A2 - A)
        max_id = adiff.argmax()
        print(f" bits/symmetric/ic_group_size={bits}/{int(symmetric)}/{ic_group_size}/({IC},{OC})    mean/max diff:{adiff.mean():.2f}/{adiff.max():.2f} @ {max_id}")
    else:
        print("A2 ~= A3")

IC = 320
OC = 640
ic_group_size = 64
resolution = 8
test(8, True, ic_group_size, IC, OC, resolution)
test(8, False, ic_group_size, IC, OC, resolution)
#test(4, True, ic_group_size, IC, OC, resolution)
#test(4, False, ic_group_size, IC, OC, resolution)