cl_kernel_sources = r'''
//# [WG_N x (BN x K)] [k-loop x (BK x BN)] [SG_N x (16xBK)]  [SG_M * (16x16)]
//# B += (wg_n * BN * K/2) + (0 * BK*BN/2) + (sg_n * 16*BK/2) + sg_m*(16*16)/2;
int BLayout2Offset(int k, int n, int K, int N) {
    int offset = 0;

    offset += (n / SG_N) * (SG_N * K / 2);
    n = n % SG_N;

    offset += (k / 16) * (16 * SG_N/2);
    k = k % 16;

    //# now inside [16, SG_N] sub-block, we split it into 2 int4 16x8 sub-blocks
    //# and merge them into single int8 16x8 sub-blocks.
    int high_4bit = (n >= 8);
    n = n % 8;
    offset += k*8 + n;

    return (high_4bit) ? (offset | 0x80000000) : (offset);
}

//# quantized weights are stored in blocked format which is suitable for intel_sub_group_block_read_uc16() to load
//# 16x8 uchar block merges two 16x8 quantized 4bit weight matries into a 8bit
//# each 16x8 weight matrix is 
//#
__kernel void quant_I4(__global half * _src, __global uchar * dst, __global half * _scales, __global char * _zps, int N, int K) {
    // ndrange: [K//QUANT_GROUP_SIZE, N//16], [1, 1]
    int kg = get_global_id(0);
    int ng = get_global_id(1);
    int k = kg * QUANT_GROUP_SIZE;

    for (int n_off = 0; n_off < 16; n_off++) {
        int n = ng*16 + n_off;

        __global half * src = _src + n * K + k;

        //# scales & zp are continous in N-dims (for brgemm-style of loading)
        __global half * scales = _scales + kg*N + n;
        __global char * zps = NULL;
        if (_zps != NULL) zps = _zps + kg*N + n;

        half vmin = src[0];
        half vmax = src[0];
        for(int i = 1; i < QUANT_GROUP_SIZE; i++) {
            vmin = min(src[i], vmin);
            vmax = max(src[i], vmax);
        }

        char zp;
        half s;
        if (zps != NULL) {
            // zero-point zp is choosen as interger
            //   vmax =  (7 + zp)*s
            //   vmin = (-8 + zp)*s
            //  
            //
            if (vmax == vmin) {
                zp = 0;
                s = 1;
            } else {
                zp = (7 * vmin + 8 * vmax)/(vmax - vmin);
                s = (vmax - vmin)/15.0f;
            }
            scales[0] = s/16;
            zps[0] = zp;
        } else {
            // no zero-point
            zp = 0;

            // scales can be negative number, so we can always map the max to -8
            if (vmin >= 0) {
                // both >=0, map -8 to vmax
                s = -vmax / 8.0f;
            } else if (vmax <= 0) {
                // both <0, map -8 to vmin
                s = -vmin / 8.0f;
            } else {
                // vmax > 0 & vmin < 0
                if (fabs(vmin) >= fabs(vmax)) {
                    //maps vmin to -8 (using positive scales)
                    s = -vmin / 8.0f;
                } else {
                    //maps vmax to -8 (using negative scales)
                    s = -vmax / 8.0f;
                }
            }
            scales[0] = s/16;
        }

        half rs = 1.0f/s;
        for(int i = 0; i < QUANT_GROUP_SIZE; i++) {
            int offset = BLayout2Offset(k + i, n, K, N);
            uchar v0 = as_uchar(clamp((char)round(src[i] * rs - zp), (char)-8, (char)7));
            if (offset & 0x80000000) {
                offset &= 0x7FFFFFFF;
                dst[offset] = (dst[offset] & 0x0F) | (v0 << 4);
            } else {
                dst[offset] = (dst[offset] & 0xF0) | (v0 & 0xF);
            }

            //# update src as fake-quanted value
            src[i] = (as_char(v0) + zp) * s;
        }
    }
}

__attribute__((intel_reqd_sub_group_size(WG_SGS)))
__kernel void XMX_tput(__global half * A,
                       __global uchar * B,
                       __global half * B_scales,
                       __global float * bias,
                       __global half * C,
                       int M, int K, int N) {
    //# groups/local : [1, N/BN, M/BM] / [WG_SGS, WG_N, WG_M]
    //# global         [8, N/SG_N, M/SG_M]

    //# local memory for A slice & B slice
    __local half buffA[BM * BK];
    __local half buffB[BN * BK];

    //# accumulators 4 x 2 register tile (each reg is 8x8 float)
    float8 c00 = 0.0f;
    float8 c10 = 0.0f;
    float8 c20 = 0.0f;
    float8 c30 = 0.0f;

    float8 c01 = 0.0f;
    float8 c11 = 0.0f;
    float8 c21 = 0.0f;
    float8 c31 = 0.0f;

    __local half * pbuffA;
    __local half * pbuffB;
    const __local uint * pA;
    const __local uint * pB;

    {
        int sg_c = get_local_id(0); //# sub-group local/channel id
        int sg_n = get_local_id(1); //# sub-group id in 'N' dim
        int sg_m = get_local_id(2); //# sub-group id in 'M' dim
        int sg_id = get_sub_group_id();

        int wg_n = get_group_id(1);
        int wg_m = get_group_id(2);
        //# A: [M, K]
        //# B: [N, K]

        //# which b0/b1 should current sg to load into SLM
        //# B is blocked
        //# [WG_N x (BN x K)] [k-loop x (BK x BN)] [SG_N x (16xBK)]  [SG_M * (16x16)]
        B += (wg_n*BN  + sg_n*16) * (K/2) + sg_m*(16*16)/2;

        //# offset in n-dimension
        B_scales += (wg_n*BN + sg_n*16);

        //# SLM target place
        pbuffB = buffB + (sg_n*16*BK) + sg_m*(16*16);

        int x = (sg_n & 1);
        int y = (sg_n >> 1);
        int m = wg_m * BM + sg_m * 32 + y*8 + sg_c;
        if (m >= M) m = M - 1; //# avoid overflow
        A += m * K + x*32;
        pbuffA = buffA + (sg_m*32*BK) + (y + x*8)*(8*16) + sg_c*16;

#if USE_DPASW
        pA = (__local uint *)(buffA + (sg_m * 32 * BK) + (sg_id & 1) * (4*16));
#elif USE_DPAS
        pA = (__local uint *)(buffA + (sg_m * 32 * BK));
#endif
        pB = (const __local uint *)(buffB + (sg_n*16*BK));
    }

    //# outer loop in 'K' dim
    for(int k0 = 0; k0 < K; k0 += BK) {
    //# commented to test performance of [SLM + dpasw] only
    #if 1
        //# load & pack A into buffA:
        //# each sub-group is responsible for :
        //#   - loading part of 32-row A slice into buffA in packed format
        //#   - loading part of 16-row B slice into buffB in packed format
        //#
        {
            //# sg_n (0~7) => [y,x]=[(0~3), (0~1)]
            __global half * src = A + k0;
            uint8 blk0 = *(__global uint8 *)(src);
            uint8 blk1 = *(__global uint8 *)(src + 16);
            *(__local uint8 *)(pbuffA) = blk0;
            *(__local uint8 *)(pbuffA + 4*8*16) = blk1;
        }

        //# load sub-slice B (128*BK) into buffB
        {
            //# quantized weight 16x8 char  => 16x8 half + 16x8 half
            uchar16 q16x8 = intel_sub_group_block_read_uc16((const __global uchar*)(B + k0*16/2));       // 16x8 char
           
            half16 h16x8_b0 = convert_half16(as_char16(q16x8 << 4));                    // low-4bit
            half16 h16x8_b1 = convert_half16(as_char16(q16x8 & (uchar16)(0xF0)));       // high-4bit

            half2 scales = as_half2(intel_sub_group_block_read_us2((const __global ushort*)(B_scales + (k0/QUANT_GROUP_SIZE)*(N))));

            h16x8_b0 *= (half16)scales.s0;
            h16x8_b1 *= (half16)scales.s1;

            // 16x8 ushort sub-group write-out
            intel_sub_group_block_write8((__local uint*)(pbuffB + 0*16), as_uint8(h16x8_b0));
            intel_sub_group_block_write8((__local uint*)(pbuffB + 8*16), as_uint8(h16x8_b1));
        }
    #endif

        barrier(CLK_LOCAL_MEM_FENCE);
        
    #if 1
        //# offset into buff A & B for each sub-group
        //# sub-group 8-rows, 16-cols: each of size 32x16
        //# both are blocked layout:
        //#        each SG in a row share 32xBK buffA slice
        //#        each SG in a col share 16xBK buffB slice

        //# unroll would cause register exhaast & spill
        __attribute__((opencl_unroll_hint(1)))
        for(int k1 = 0; k1 < BK/16; k1 ++) {
            //# limit scopes of temp regs to reuse-registers
#if USE_DPASW
            int4 a0 = as_int4(intel_sub_group_block_read4(pA + 0*8*8));
            int4 a1 = as_int4(intel_sub_group_block_read4(pA + 1*8*8));
            int4 a2 = as_int4(intel_sub_group_block_read4(pA + 2*8*8));
            int4 a3 = as_int4(intel_sub_group_block_read4(pA + 3*8*8));

            //# scatter load is no good here, SLM will be in best tput using blocked sub-group read
            int8 b0 = as_int8(intel_sub_group_block_read8(pB + 0*8*8));
            int8 b1 = as_int8(intel_sub_group_block_read8(pB + 1*8*8));

            c00 = intel_sub_group_f16_f16_split_matrix_mad_k16(a0, b0, c00);
            c10 = intel_sub_group_f16_f16_split_matrix_mad_k16(a1, b0, c10);
            c20 = intel_sub_group_f16_f16_split_matrix_mad_k16(a2, b0, c20);
            c30 = intel_sub_group_f16_f16_split_matrix_mad_k16(a3, b0, c30);

            c01 = intel_sub_group_f16_f16_split_matrix_mad_k16(a0, b1, c01);
            c11 = intel_sub_group_f16_f16_split_matrix_mad_k16(a1, b1, c11);
            c21 = intel_sub_group_f16_f16_split_matrix_mad_k16(a2, b1, c21);
            c31 = intel_sub_group_f16_f16_split_matrix_mad_k16(a3, b1, c31);
#elif USE_DPAS
            int8 a0 = as_int8(intel_sub_group_block_read8(pA + 0*8*8));
            int8 a1 = as_int8(intel_sub_group_block_read8(pA + 1*8*8));
            int8 a2 = as_int8(intel_sub_group_block_read8(pA + 2*8*8));
            int8 a3 = as_int8(intel_sub_group_block_read8(pA + 3*8*8));

            //# scatter load is no good here, SLM will be in best tput using blocked sub-group read
            int8 b0 = as_int8(intel_sub_group_block_read8(pB + 0*8*8));
            int8 b1 = as_int8(intel_sub_group_block_read8(pB + 1*8*8));

            c00 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b0, c00);
            c10 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b0, c10);
            c20 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b0, c20);
            c30 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b0, c30);

            c01 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b1, c01);
            c11 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b1, c11);
            c21 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b1, c21);
            c31 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b1, c31);
#endif
            pA += 4*8*8;
            pB += 2*8*8;
        }
        //# rewind pA & pB after loop to save registers
        pA -= BK/16*4*8*8;
        pB -= BK/16*2*8*8;
    #endif
        //# need to sync again at the end, to avoid faster threads
        //# fetching the next block into the cache before slower threads are done
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    //# store sub-C
    {
        int sg_c = get_local_id(0); //# sub-group local/channel id
        int sg_n = get_local_id(1); //# sub-group id in 'N' dim
        int sg_m = get_local_id(2); //# sub-group id in 'M' dim
        int sg_id = get_sub_group_id();
        
        int n = get_group_id(1)*BN + sg_n*16;
        int m = get_group_id(2)*BM + sg_m*32;
        int channel = get_sub_group_local_id();

        if (bias) {
            float8 bias0 = convert_float8((half8)(bias[n + channel]));
            float8 bias1 = convert_float8((half8)(bias[n + 8 + channel]));
            c00 += bias0; c01 += bias1;
            c10 += bias0; c11 += bias1;
            c20 += bias0; c21 += bias1;
            c30 += bias0; c31 += bias1;
        }

        //# transpose with the help of SLM : using sub-group/scatter load/store
        //# store to SLM in compact format
        //# scatter reads & scatter write to change strides from 8(compat) to N(strided)
        __local half * scratch = buffA + get_sub_group_id() * 8 * 8 * 2;
        __local half * src = (__local half *)(scratch + channel * 8);
#if COMBINE_GATE_UP == 1
        N = N / 2;
        n = n / 2;
#endif

#if COMBINE_GATE_UP == 1
#define STORE_C_REG(creg0, creg1) { \
            if (m >= M) m = M - 1; /*# avoid overflow*/ \
            intel_sub_group_block_write_us8(scratch      , as_ushort8(convert_half8(creg0))); \
            intel_sub_group_block_write_us8(scratch + 8*8, as_ushort8(convert_half8(creg1))); \
            half8 v00 = *(__local half8 *)(src);                                            \
            half8 v01 = *(__local half8 *)(src + 8*8);                                      \
            __global half * dst = (__global half *)(C +  m*N + n);                          \
            v00 = v00 / ((half8)(1.0f) + native_exp(-v00)) * v01;  /* silu(gate_proj) * up_proj */  \
            *(__global half8 *)dst = v00;                                                   \
        }
#else
#define STORE_C_REG(creg0, creg1) { \
            if (m >= M) m = M - 1; /*# avoid overflow*/ \
            intel_sub_group_block_write_us8(scratch      , as_ushort8(convert_half8(creg0))); \
            intel_sub_group_block_write_us8(scratch + 8*8, as_ushort8(convert_half8(creg1))); \
            half8 v00 = *(__local half8 *)(src);                                            \
            half8 v01 = *(__local half8 *)(src + 8*8);                                      \
            __global half * dst = (__global half *)(C +  m*N + n);                          \
            *(__global half8 *)dst = v00;                                                   \
            *(__global half8 *)(dst + 8) = v01;                                             \
        }
#endif
        m += channel;
        STORE_C_REG(c00, c01); m += 8;
        STORE_C_REG(c10, c11); m += 8;
        STORE_C_REG(c20, c21); m += 8;
        STORE_C_REG(c30, c31);
    }
}

__kernel void Matmul_reference(__global half * A, __global half * B, __global float * bias, __global half * C, int M, int K, int N) {
    int n = get_global_id(0);
    int m = get_global_id(1);
    __global half * pA = A + m*K;
    __global half * pB = B + n*K;

    float sum = 0;

    __attribute__((opencl_unroll_hint(8)))
    for(int k = 0; k < K; k ++) {
        sum += pA[k] * pB[k];
    }

    if (bias) sum += bias[n];
    
    C[m*N + n] = sum;
}

__kernel void XMX_interleave(__global half * src0, __global half * src1, __global half * dst, int N, int K) {
    int k = get_global_id(0);
    int n = get_global_id(1);
    
    int n8g = n / 8;
    int n8r = n % 8;
    
    dst[(n8g*16 + n8r) * K     + k] = src0[n*K + k];
    dst[(n8g*16 + n8r + 8) * K + k] = src1[n*K + k];
}

'''

cl_kernel_sources_M1 = r'''
void splitter(const int n, const int team, const int tid, int* n_start, int* n_end) {
    if (team <= 1 || n == 0) {
        *n_start = 0;
        *n_end = n;
    } else {
        int n1 = (n + team - 1) / team;
        int n2 = n1 - 1;
        int T1 = n - n2 * team;
        *n_end = tid < T1 ? n1 : n2;
        *n_start = tid <= T1 ? tid * n1 : T1 * n1 + (tid - T1) * n2;
    }
    *n_end += *n_start;
}

//# M==1, no need to reuse B matrix through SLM
//# 
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void XMX_M1(__global half * A,
                     __global uchar * B,
                     __global half * B_scales,
                     __global float * bias,
                     __global half * C,
                     int M, int K, int N) {
    //# HTs in same work-group will split works along K dimension into 
    //# 
    //# GWS [8*(N//16), 16] LWS [8, 16] :     every work-group (with 16 sub-groups) handles 16 outputs
    //# a: 1x16   b: 16x8   c: 1x8
    int ng = get_group_id(0);

    //# split quantization-groups into multiply `nthr` threads
    int ithr = get_local_id(1);
    int nthr = get_local_size(1); //# fixed 16 HW-threads(each HW-thread is a sub-group of size 8)
    int K_groups = K/QUANT_GROUP_SIZE;
    int gk0, gk1;
    splitter(K_groups, nthr, ithr, &gk0, &gk1);

    //# offset in N direction
    B += ng * (K*8);
    B_scales += ng * 16;

    //# offset in K direction
    B_scales += gk0 * N;
    A += gk0 * QUANT_GROUP_SIZE;
    B += gk0 * QUANT_GROUP_SIZE*8;

    //# do the part assigned to current sub-group
    float c0 = 0.0f;
    float c1 = 0.0f;
    for(int gk = gk0; gk < gk1; gk++) {
        half2 scales = as_half2(intel_sub_group_block_read_us2((const __global ushort*)(B_scales)));
        float cg0 = 0.0f;
        float cg1 = 0.0f;
        for (int k = 0; k < QUANT_GROUP_SIZE; k += 16) {
            uchar16 q16x8 = intel_sub_group_block_read_uc16((const __global uchar*)(B));    // 16x8 char
            half16 h16x8_b0 = convert_half16(as_char16(q16x8 << 4));
            half16 h16x8_b1 = convert_half16(as_char16(q16x8 & (uchar16)(0xF0)));
            //# h16x8_b0 *= (half16)scales.s0;
            //# h16x8_b1 *= (half16)scales.s1;
            int8 b0 = as_int8(h16x8_b0);
            int8 b1 = as_int8(h16x8_b1);

            int a = as_int(intel_sub_group_block_read((const __global uint*)(A)));
            A += 16;
            B += 16*8;

            cg0 = intel_sub_group_f16_f16_matrix_mad_k16(a, b0, cg0);
            cg1 = intel_sub_group_f16_f16_matrix_mad_k16(a, b1, cg1);
        }
        B_scales += N;
        c0 += cg0 * scales.s0;
        c1 += cg1 * scales.s1;
    }

    //# reduce
    __local float all_sum[K_NTHRS][16];     //# [nthr, N16]
    int id_sg_local = get_sub_group_local_id();

    all_sum[ithr][id_sg_local] = c0;
    all_sum[ithr][id_sg_local + 8] = c1;

    //intel_sub_group_block_write((__local uint*)(&all_sum[ithr][0]), as_uint(c0));
    //intel_sub_group_block_write((__local uint*)(&all_sum[ithr][8]), as_uint(c1));

    barrier(CLK_LOCAL_MEM_FENCE);

    //# only the first sg responsible for reducing & post-processing
    if (ithr != 0) return;

    for(int i = 1; i < K_NTHRS; i++) {
        float nc0 = as_float(intel_sub_group_block_read((const __local uint*)(&all_sum[i][0])));
        float nc1 = as_float(intel_sub_group_block_read((const __local uint*)(&all_sum[i][8])));
        c0 += nc0;
        c1 += nc1;
    }

    //# add bias
    if (bias != NULL) {
        int n = ng*16 + get_local_id(0);
        c0 += bias[n]; 
        c1 += bias[n + 8]; 
    }

    half v00 = convert_half(c0);
    half v01 = convert_half(c1);

#if COMBINE_GATE_UP == 1
    v00 = v00 / ((half)(1.0f) + native_exp(-v00)) * v01;  /* silu(gate_proj) * up_proj */
    C += ng*8;
    intel_sub_group_block_write_us((__global ushort*)(C), as_ushort(v00));
#else
    C += ng*16;
    //# store results : 16 x half
    intel_sub_group_block_write_us((__global ushort*)(C), as_ushort(v00));
    intel_sub_group_block_write_us((__global ushort*)(C + 8), as_ushort(v01));
#endif
}
'''

from . import cl
import numpy as np
from .utils import *

KDEBUG=0

QUANT_GROUP_SIZE = 128

#================ Hyper parameter ============================
WG_SGS = 8  # sub-group size (must be 8 for dpasw)
WG_N = 8    # max 16
WG_M = 4    # max 8
assert (WG_N * WG_M) <= 16 * 8, f"number of sub-groups in work-group {WG_N} x {WG_M} exceeds Xe-core capability: {16*8}"

# each subgroup using 4x2 float8 reg tiling, which produces 32 x 16 outputs in C
SG_M = (4*8)
SG_N = (2*8)

# all sub-groups in a work-group handles sub-matrix C of shape [BM, BN]
BM = WG_M * SG_M
BN = WG_N * SG_N
BK = 64 # BK is limited by SLM size
SLM_size = 65536
SLM_use = (BM*BK + BN*BK)*2
assert SLM_use <= SLM_size, f"local memory overflow: ({BM}*{BK} + {BN}*{BK})*2 = {SLM_use} > SLM size {SLM_size}"

num_reg_slm_copyA = SG_M * BK/WG_N/16 # BK is number fo half elements, 16 is half elements per register
num_reg_slm_copyB = SG_N * BK/WG_M/16 # BK is number fo half elements, 16 is half elements per register

assert num_reg_slm_copyA == int(num_reg_slm_copyA), f"num_reg_slm_copyA {num_reg_slm_copyA} is not integer"
assert num_reg_slm_copyB == int(num_reg_slm_copyB), f"num_reg_slm_copyB {num_reg_slm_copyB} is not integer"

assert (BK//16 * SG_M)

K_NTHRS = 8

print(f"{Colors.YELLOW}===== XMX hyper-param ===================")
print(f"BM = {BM} WG_M = {WG_M}")
print(f"BN = {BN} WG_N = {WG_N}")
print(f"BK = {BK} Inner-Loop-Cnt = {BK//16}")
print(f"SG_M x SG_N = {SG_M} x {SG_N}")
print(f"WG HW threads = {WG_N*WG_M*100/(16*8):.1f} %  {WG_N}x{WG_M} = {WG_N*WG_M} ")
print(f"SLM usage     = {SLM_use*100/SLM_size:.1f} %  {SLM_use} bytes(SLM:A {BM*BK*2} + SLM:B {BN*BK*2})")

print(f"num of registers to copy A (per-sub-group per-BK-step) = {num_reg_slm_copyA}")
print(f"num of registers to copy B (per-sub-group per-BK-step) = {num_reg_slm_copyB}")
print(f"============================{Colors.END}")

class Linear_w4x:
    # if weight_up is provided, gate/up combination & silu/mul is fused
    def __init__(self, weight, bias, weight_up = None, do_fakequant_weight = False):
        self.N, self.K = weight.shape # weight: [N, K]
        self.bias = to_cl(bias)
        assert self.N % BN == 0, f"'N' dimension {self.N} is not multiple of BM {BN}"
        assert self.K % BK == 0, f"'K' dimension {self.K} is not multiple of BK {BK}"

        dinfo = cl.dev_info()
        USE_DPAS = int('cl_intel_subgroup_matrix_multiply_accumulate' in dinfo['CL_DEVICE_EXTENSIONS'])
        USE_DPASW = int('cl_intel_subgroup_split_matrix_multiply_accumulate' in dinfo['CL_DEVICE_EXTENSIONS'])

        assert USE_DPAS or USE_DPASW, f"no dpas or dpasw instructions on target device : {dinfo['CL_DEVICE_NAME']}"

        if weight_up is not None:
            self.COMBINE_GATE_UP = 1
        else:
            self.COMBINE_GATE_UP = 0
        self.cl_kernels = kernel_cache(cl_kernel_sources,
                                       options=(f"-DBM={BM} -DBN={BN} -DBK={BK} "
                                                f"-D WG_SGS={WG_SGS} -DWG_N={WG_N} -DWG_M={WG_M} "
                                                f"-D SG_N={SG_N} -D SG_M={SG_M} "
                                                f"-D KDEBUG={KDEBUG} "
                                                f"-D COMBINE_GATE_UP={self.COMBINE_GATE_UP} "
                                                f"-D USE_DPASW={USE_DPASW} -D USE_DPAS={USE_DPAS} "
                                                f"-D QUANT_GROUP_SIZE={QUANT_GROUP_SIZE} -D K_NTHRS={K_NTHRS}"))

        self.cl_kernels_M1 = kernel_cache(cl_kernel_sources_M1,
                                       options=(f"-D COMBINE_GATE_UP={self.COMBINE_GATE_UP} "
                                                f"-D QUANT_GROUP_SIZE={QUANT_GROUP_SIZE} -D K_NTHRS={K_NTHRS}"))
        
        # print("99999999999999", self.cl_kernels_M1.info("XMX_M1", [16, 16], 16))

        if self.COMBINE_GATE_UP:
            # interleaving gate & up proj_matrix
            N = 2*self.N
            assert weight.shape == weight_up.shape
            weight_raw_gate = to_cl(weight.half())
            weight_raw_up = to_cl(weight_up.half())
            weight_raw = cl.tensor([N, self.K], np.dtype(np.float16))
            self.cl_kernels.enqueue("XMX_interleave", [self.K, self.N], [1,1], weight_raw_gate, weight_raw_up, weight_raw, self.N, self.K)
        else:
            N = self.N
            weight_raw = to_cl(weight.half())

        # quantize & repack weight matrix
        self.weight = cl.tensor([N, self.K//2], np.dtype(np.uint8))
        self.weight_scales = cl.tensor([self.K//QUANT_GROUP_SIZE, N], np.dtype(np.float16))
        self.weight_zps = None
        self.cl_kernels.enqueue("quant_I4", [self.K//QUANT_GROUP_SIZE, N//SG_N], [1, 1],
                                weight_raw,
                                self.weight,
                                self.weight_scales,
                                self.weight_zps,
                                N, self.K)
        self.debug = False
        
        if do_fakequant_weight:
            # save fake quantized weight
            self.weight_fq = weight_raw

    def __call__(self, input):
        # shape inference
        i_shape = input.shape
        o_shape = list(i_shape)
        o_shape[-1] = self.N
        output = cl.tensor(o_shape, input.dtype)

        M = input.numel // self.K
        M_padded = (M + (BM - 1))//BM * BM

        N = self.N
        K = self.K
        if self.COMBINE_GATE_UP:
            N *= 2
        assert N % BN == 0, f"'N' dimension {N} is not multiple of BM {BN}"
        assert K % BK == 0, f"'K' dimension {K} is not multiple of BK {BK}"

        if self.debug:
            total_work_groups = (M_padded//BM) * (N//BN)
            total_sub_groups = total_work_groups * WG_M * WG_N
            print(f"{Colors.YELLOW}============================")
            print(f"M = {M}")
            print(f"M_padded = {M_padded} = {M_padded//BM} x BM")
            print(f"N = {N} = {N//BN} x BN")
            print(f"K = {K} = {K//BK} x BK")
            print(f"total_work_groups = {total_work_groups}")
            print(f"total_sub_groups = {total_sub_groups}")
            total_eus_threads = 512 * 8
            print(f"over-subscribe ratio = {total_sub_groups*100/total_eus_threads:.1f} %")
            print(f"============================{Colors.END}")

        if M > 1:
            self.cl_kernels.enqueue("XMX_tput", [WG_SGS, N//SG_N, M_padded//SG_M], [WG_SGS, WG_N, WG_M], input, self.weight, self.weight_scales, self.bias, output, M, K, N)
        else:
            self.cl_kernels_M1.enqueue("XMX_M1", [8*(N//16), K_NTHRS], [8, K_NTHRS], input, self.weight, self.weight_scales, self.bias, output, M, K, N)
        #self.cl_kernels.enqueue("XMX_tput", [WG_SGS, N//SG_N, M_padded//SG_M], [WG_SGS, WG_N, WG_M], input, self.weight, self.bias, output, M, K, N)
        #self.cl_kernels.enqueue("Matmul_reference", [self.N, M], [8, 16], input, self.weight_raw, self.bias, output, M, self.K, self.N)

        return output


def test(M, K, N, REPEATE = 10):
    cl.profiling(True)
    np.set_printoptions(linewidth = 120)
    
    # M, K, N = 128, 128, 128
    print(f"prepare test data  M={M}, K={K}, N={N}...")
    vRANGE = 2
    torch.manual_seed(0)
    A = torch.randint(-vRANGE, vRANGE+1, [M, K]).half()
    B = torch.randint(-vRANGE, vRANGE+1, [N, K]).half()
    bias = torch.randint(-vRANGE, vRANGE+1, [N]).half()
    
    B = B*0.1
    B[0,0] = 4

    layers = []
    for i in range(REPEATE):
        do_fakequant_weight = (i==0)
        layers.append(Linear_w4x(B, bias, do_fakequant_weight = do_fakequant_weight))

    cl_kernels = layers[0].cl_kernels
    input = to_cl(A)

    weight_fq = layers[0].weight_fq
    
    #print(B[:8, :8])
    #print(weight_fq.numpy()[:8, :8])

    output = cl.tensor([M, N], np.dtype(np.float16))
    tbias = to_cl(bias)
    cl_kernels.enqueue("Matmul_reference", [N, M], [1, 1], input, weight_fq, tbias, output, M, K, N)
    ref = output.numpy()

    for i in range(0, REPEATE):
        output = layers[i](input)

    latency_ns = cl.finish()
    for ns in latency_ns:
        flops = (M * K * N * 2)
        rd_bytes = (M*K)*2 + K*N//2
        print(f"Linear_w4x: {flops*1e-9:.1f} Gflop / {ns*1e-3:.1f} us = {flops/ns*1e-3:.3f} TFlops/s      {rd_bytes/ns:.3f} GB/s ")

    res = output.numpy()
    compare(ref, res)


if __name__ == "__main__":
    import sys
    cl.profiling(True)
    batch_seq = 16*1024

    num_hidden_layers = 40
    batch = 2048
    hidden_size = 3072
    intermediate_size = 8192
    num_attention_heads = 24
    num_key_value_heads = 6
    head_size = hidden_size//num_attention_heads
    qkv_proj_size = (head_size)*(num_attention_heads + num_key_value_heads*2)

    flops = 0
    flops += 2*(batch * hidden_size * qkv_proj_size) * num_hidden_layers     # qkv_proj
    flops += 2*(batch * hidden_size * hidden_size) * num_hidden_layers       # out_proj
    flops += 2*(batch * hidden_size * intermediate_size) * num_hidden_layers # gate
    flops += 2*(batch * hidden_size * intermediate_size) * num_hidden_layers # up
    flops += 2*(batch * hidden_size * intermediate_size) * num_hidden_layers # down

    print(f"First Token TFLOP {flops/1e12:.3f}")
    
    #test(M=BM, K=BK*160, N=BN*64, REPEATE = 20); sys.exit(0)
    #hidden_size = 4096
    test(1, hidden_size, hidden_size, REPEATE = 20); sys.exit(0)

    for batch in [7, 1]:
        test(batch, 2048, 2560, REPEATE = 1)
        test(batch, 2048, 2048, REPEATE = 1)
        test(batch, 2048, 5632, REPEATE = 1)
        test(batch, 5632, 2048, REPEATE = 1)
        test(batch, 2048, 32000, REPEATE = 1)
    
    #test(1, hidden_size, hidden_size, REPEATE = 10)
    
