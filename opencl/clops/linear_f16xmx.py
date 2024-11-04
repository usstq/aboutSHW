cl_kernel_sources = r'''ulong __attribute__((overloadable)) intel_get_cycle_counter( void );

__attribute__((intel_reqd_sub_group_size(WG_SGS)))
__kernel void XMX_prepackB(__global half * Bsrc, __global half * Bdst, int N, int K) {
    int sg_c = get_local_id(0); //# sub-group local/channel id
    int sg_n = get_local_id(1); //# sub-group id in 'N' dim
    int sg_m = get_local_id(2); //# sub-group id in 'M' dim
    int sg_id = get_sub_group_id();

    int wg_n = get_group_id(1);
    int wg_m = get_group_id(2);
    //# Bsrc/Bdst: [N, K]
    Bsrc += wg_n * BN * K;
    Bdst += wg_n * BN * K;

    for(int k0 = 0; k0 < K; k0 += BK) {
        int x = sg_m;
        __global half * src = Bsrc + (sg_n * 16 + sg_c)*K + k0 + x*16;
        uint8 blk0 = *(__global uint8 *)(src);
        uint8 blk1 = *(__global uint8 *)(src + 8*K);

        __global half * dst = Bdst + k0*BN + (sg_n*16*BK) + sg_m*(16*16);
        intel_sub_group_block_write8((__global uint*)(dst), blk0);
        intel_sub_group_block_write8((__global uint*)(dst + 8*16), blk1);
    }
}

__attribute__((intel_reqd_sub_group_size(WG_SGS)))
__kernel void XMX_tput(__global half * A, __global half * B, __global float * bias, __global half * C, int M, int K, int N) {
    //# groups/local : [1, N/BN, M/BM] / [WG_SGS, WG_N, WG_M]
    //# global         [8, N/SG_N, M/SG_M]

    //# local memory for A slice & B slice
    __local half buffA[BM * BK];
    __local half buffB[BN * BK];

    int sg_c = get_local_id(0); //# sub-group local/channel id
    int sg_n = get_local_id(1); //# sub-group id in 'N' dim
    int sg_m = get_local_id(2); //# sub-group id in 'M' dim
    int sg_id = get_sub_group_id();


    //# accumulator (32x16)
    float8 c[4][2] = {0.0f};
    {
        int wg_n = get_group_id(1);
        int wg_m = get_group_id(2);
        //# A: [M, K]
        //# B: [N, K]
        B += wg_n * BN * K;
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
            int wg_m = get_group_id(2);
            int x = (sg_n & 1);
            int y = (sg_n >> 1);
            int m = wg_m * BM + sg_m * 32 + y*8 + sg_c;
            if (m >= M) m = M - 1; //# avoid overflow
            __global half * src = A + m * K + k0 + x*32;
            uint8 blk0 = *(__global uint8 *)(src);
            uint8 blk1 = *(__global uint8 *)(src + 16);
            __local half * dst = buffA + (sg_m*32*BK) + (y + x*8)*(8*16) + sg_c*16;
            *(__local uint8 *)(dst) = blk0;
            *(__local uint8 *)(dst + 4*8*16) = blk1;
        }

        //# load sub-slice B (256*BK) into buffB : (256*BK)/(16*8) = 128 half (8 x regs)
        //#  suppose B has been turned into blocked layout : [K/BK, 256, BK]
        {
            #if 1
                //# given B is prepacked, sub-group block read is faster
                __global half * src = B + k0*BN + (sg_n*16*BK) + sg_m*(16*16);
                uint8 blk0 = intel_sub_group_block_read8((__global uint *)(src));
                uint8 blk1 = intel_sub_group_block_read8((__global uint *)(src + 8*16));
                __local half * dst = buffB + (sg_n*16*BK) + sg_m*(16*16);
                intel_sub_group_block_write8((__local uint*)(dst), blk0);
                intel_sub_group_block_write8((__local uint*)(dst + 8*16), blk1);
            #else
                int x = sg_m;
                __global half * src = B + (sg_n * 16 + sg_c)*K + k0 + x*16;
                uint8 blk0 = *(__global uint8 *)(src);
                uint8 blk1 = *(__global uint8 *)(src + 8*K);
                __local half * dst = buffB + (sg_n*16*BK) + sg_m*(16*16);
                intel_sub_group_block_write8((__local uint*)(dst), blk0);
                intel_sub_group_block_write8((__local uint*)(dst + 8*16), blk1);
            #endif
        }
    #endif
        barrier(CLK_LOCAL_MEM_FENCE);
    #if 1
        //# offset into buff A & B for each sub-group
        //# sub-group 8-rows, 16-cols: each of size 32x16
        //# both are blocked layout:
        //#        each SG in a row share 32xBK buffA slice
        //#        each SG in a col share 16xBK buffB slice
        const __local uint * pA = (const __local uint *)(buffA + (sg_m * 32 * BK) + (sg_id & 1) * (4*16));

        //# each sg_n contains two int8x8(half8x16) (B00,B01,B10,B11,....,)
        const __local uint * pB = (const __local uint *)(buffB + (sg_n * 16) * BK);

        //# unroll would cause register exhaast & spill
        __attribute__((opencl_unroll_hint(1)))
        for(int k1 = 0; k1 < BK/16; k1 ++) {
            //# limit scopes of temp regs to reuse-registers
            int4 a0 = as_int4(intel_sub_group_block_read4(pA + 0*8*8));
            int4 a1 = as_int4(intel_sub_group_block_read4(pA + 1*8*8));
            int4 a2 = as_int4(intel_sub_group_block_read4(pA + 2*8*8));
            int4 a3 = as_int4(intel_sub_group_block_read4(pA + 3*8*8));

            //# scatter load is no good here, SLM will be in best tput using blocked sub-group read
            int8 b0 = as_int8(intel_sub_group_block_read8(pB + 0*8*8));
            int8 b1 = as_int8(intel_sub_group_block_read8(pB + 1*8*8));

            c[0][0] = intel_sub_group_f16_f16_split_matrix_mad_k16(a0, b0, c[0][0]);
            c[1][0] = intel_sub_group_f16_f16_split_matrix_mad_k16(a1, b0, c[1][0]);
            c[2][0] = intel_sub_group_f16_f16_split_matrix_mad_k16(a2, b0, c[2][0]);
            c[3][0] = intel_sub_group_f16_f16_split_matrix_mad_k16(a3, b0, c[3][0]);

            c[0][1] = intel_sub_group_f16_f16_split_matrix_mad_k16(a0, b1, c[0][1]);
            c[1][1] = intel_sub_group_f16_f16_split_matrix_mad_k16(a1, b1, c[1][1]);
            c[2][1] = intel_sub_group_f16_f16_split_matrix_mad_k16(a2, b1, c[2][1]);
            c[3][1] = intel_sub_group_f16_f16_split_matrix_mad_k16(a3, b1, c[3][1]);

            pA += 4*8*8;
            pB += 2*8*8;
        }
    #endif
        //# need to sync again at the end, to avoid faster threads
        //# fetching the next block into the cache before slower threads are done
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    //# store sub-C
    {
        int n = get_group_id(1)*BN + sg_n*16;
        int m0 = get_group_id(2)*BM + sg_m*32;
        int channel = get_sub_group_local_id();

        if (bias) {
            float8 bias0 = convert_float8((half8)(bias[n + channel]));
            float8 bias1 = convert_float8((half8)(bias[n + 8 + channel]));
            for(int i = 0; i < 4; i++) {
                c[i][0] += bias0;
                c[i][1] += bias1;
            }
        }

        //# transpose with the help of SLM : using sub-group/scatter load/store
        //# store to SLM in compact format
        //# scatter reads & scatter write to change strides from 8(compat) to N(strided)
#if KDEBUG == 1
        if(m == 32 && n == 0 && channel < 8) {
            half8 v00 = convert_half8(c[0][0]);
            printf("]]]]] channel %d:  %f\n", channel, v00.s0);
            //printf("]]]]] %f\n", v00.s1);
            //printf("]]]]] %f\n", v00.s2);
            //printf("]]]]] %f\n", v00.s3);
            //printf("]]]]] %f\n", v00.s4);
            //printf("]]]]] %f\n", v00.s5);
            //printf("]]]]] %f\n", v00.s6);
            //printf("]]]]] %f\n", v00.s7);
        }
#endif
        __local half * scratch = buffA + get_sub_group_id() * 8 * 8 * 2;
        __local half * src = (__local half *)(scratch + channel * 8);

        for(int i = 0; i < 4; i++) {
            int m = m0 + channel + i*8;
            if (m >= M) m = M - 1; //# no overflow
            __global half * dst = (__global half *)(C +  m*N + n);
            intel_sub_group_block_write_us8(scratch      , as_ushort8(convert_half8(c[i][0])));
            intel_sub_group_block_write_us8(scratch + 8*8, as_ushort8(convert_half8(c[i][1])));
            half8 v00 = *(__local half8 *)(src);
            half8 v01 = *(__local half8 *)(src + 8*8);
            *(__global half8 *)dst = v00;
            *(__global half8 *)(dst + 8) = v01;
        }
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

'''
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

KDEBUG = 0

if KDEBUG:
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

from . import cl
import numpy as np
from .utils import *


class Linear_f16xmx:
    def __init__(self, weight, bias):
        self.N, self.K = weight.shape # weight: [N, K]
        self.bias = to_cl(bias)
        assert self.N % BN == 0, f"'N' dimension {self.N} is not multiple of BM {BN}"
        assert self.K % BK == 0, f"'K' dimension {self.K} is not multiple of BK {BK}"

        self.cl_kernels = kernel_cache(cl_kernel_sources,
                                       options=f"-DBM={BM} -DBN={BN} -DBK={BK} -D WG_SGS={WG_SGS} -DWG_N={WG_N} -DWG_M={WG_M} -D KDEBUG={KDEBUG}")

        self.weight_raw = to_cl(weight.half())
        self.weight = to_cl(weight.half())
        self.cl_kernels.enqueue("XMX_prepackB", [WG_SGS, self.N//SG_N, BM//SG_M], [WG_SGS, WG_N, WG_M], self.weight_raw, self.weight, self.N, self.K)

    def __call__(self, input):
        # shape inference
        i_shape = input.shape
        o_shape = list(i_shape)
        o_shape[-1] = self.N
        output = cl.tensor(o_shape, input.dtype)

        M = input.numel // self.K
        M_padded = (M + (BM - 1))//BM * BM

        self.cl_kernels.enqueue("XMX_tput", [WG_SGS, self.N//SG_N, M_padded//SG_M], [WG_SGS, WG_N, WG_M], input, self.weight, self.bias, output, M, self.K, self.N)
        #self.cl_kernels.enqueue("Matmul_reference", [self.N, M], [8, 16], input, self.weight_raw, self.bias, output, M, self.K, self.N)
        
        return output


def test_mm(M, K, N, compare_ref = False, REPEATE = 2):
    assert N % BN == 0, f"'N' dimension {N} is not multiple of BM {BN}"
    assert K % BK == 0, f"'K' dimension {K} is not multiple of BK {BK}"

    M_padded = (M + (BM - 1))//BM * BM

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

    print("prepare test data...")
    vRANGE = 2
    np.random.seed(0)
    A = np.random.randint(-vRANGE, vRANGE+1, [M, K]).astype(np.float16)
    B = np.random.randint(-vRANGE, vRANGE+1, [N, K]).astype(np.float16)
    bias = np.random.randint(-vRANGE, vRANGE+1, [N]).astype(np.float16)
    k = kernel_cache(cl_kernel_sources,
                     options=f"-DBM={BM} -DBN={BN} -DBK={BK} -D WG_SGS={WG_SGS} -DWG_N={WG_N} -DWG_M={WG_M} -D KDEBUG={KDEBUG}")

    if KDEBUG:
        for ni in range(8):
            for ki in range(16):
                A[ni, ki] = ni + ki*0.1
                B[ni, ki] = ni + ki*0.1
        print(f"A[32:40, :16]=\n{A[32:40, :16]}")
        print(f"B=\n{B[:8, :16]}")

    tA = cl.tensor(A)
    tB = cl.tensor(B)
    tbias = cl.tensor(bias)
    C = cl.tensor([M, N], np.dtype(np.float16))
    print("prepare reference result...")
    if compare_ref:
        k.enqueue("Matmul_reference", [N, M], [1, 1], tA, tB, tbias, C, M, K, N)
        cl.finish()
    print("done.")

    if True:
        all_linear = [Linear_f16xmx(torch.from_numpy(B), torch.from_numpy(bias)) for _ in range(REPEATE)]
        for i in range(0, REPEATE):
            tC = all_linear[i](tA)
    else:
        Bsrc = cl.tensor(B)
        Bdst = cl.tensor(B)
        k.enqueue("XMX_prepackB", [WG_SGS, N//SG_N, BM//SG_M], [WG_SGS, WG_N, WG_M], Bsrc, Bdst, N, K)
        B_prepacked = Bdst.numpy()

        tBs = [cl.tensor(B_prepacked) for _ in range(REPEATE)]
        tbias = [cl.tensor(bias) for _ in range(REPEATE)]
        tC = cl.tensor([M, N], np.dtype(np.float16))

        for i in range(0, REPEATE):
            k.enqueue("XMX_tput", [WG_SGS, N//SG_N, M_padded//SG_M], [WG_SGS, WG_N, WG_M], tA, tBs[i], tbias[i], tC, M, K, N)

    latency_ns = cl.finish()
    for ns in latency_ns:
        flops = (M * K * N * 2)
        rd_bytes = (M*K + K*N)*2
        actual_rd = total_work_groups * (BM*K + BN*K)*2
        print(f"XMX_tput: {flops*1e-9:.1f} Gflop / {ns*1e-3:.1f} us = {flops/ns*1e-3:.3f} TFlops/s    {rd_bytes/1e6:.1f}=>{actual_rd/1e6:.1f} MB  / {ns*1e-3:.1f} us = {actual_rd/ns:.1f} GB/s")

    C = C.numpy()
    tC = tC.numpy()
    #print(C[30:35, [0,1,2,3,8,9,10,11]])
    #print("=========================")
    #print(tC[30:35,[0,1,2,3,8,9,10,11]])
    if compare_ref:
        compare(C, tC)

if __name__ == "__main__":
    import sys
    cl.profiling(True)
    batch_seq = 1
    test_mm(M=128, K=896, N=151936, compare_ref =True, REPEATE = 1)
    test_mm(M=batch_seq, K=896, N=1152, compare_ref =True, REPEATE = 10)
    test_mm(M=batch_seq, K=896, N=896, compare_ref =True, REPEATE = 10)
    test_mm(M=batch_seq, K=896, N=4864, compare_ref =True, REPEATE = 10)
    test_mm(M=batch_seq, K=4864, N=896, compare_ref =True, REPEATE = 10)

