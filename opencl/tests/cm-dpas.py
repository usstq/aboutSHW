#!/usr/bin/python3
from clops import cl
import numpy as np
import time
from clops import compare


'''
Xe2 register width: 512 bit
16x 32bits accumulation destinations

each systolic step do a VNNI-style dot-product:
    // bf16/fp16
    dest[i] +=  src2[i*2+0] * src1[i*2+0]
              + src2[i*2+1] * src1[i*2+1]

    // s8/u8
    dest[i] +=  src2[i*4+0] * src1[i*4+0]
              + src2[i*4+1] * src1[i*4+1]
              + src2[i*4+2] * src1[i*4+2]
              + src2[i*4+3] * src1[i*4+3]

The systolic depth must be equal to 8 for all the XMX functions.
so K dimension is 8*2 for bf16/fp16 and 8*4 for s8/u8

https://www.techpowerup.com/gpu-specs/arc-b580.c4244
 - 160 EU
 - SIMD-16 ALU (32 flops)
 - frequency: 2.67HGz, clinfo MAX 2.9GHz
 - SLM: 131072 (128*1024 = 128 KiB)

FP32    : 160*2.67*16(SIMD-width)*2(MAD)/1e3 = 13.67 TFLOPS
FP16    : 160*2.67*32(SIMD-width)*2(MAD)/1e3 = 27.34 TFLOPS
XMX-FP16: 160*2.67*32(SIMD-width)*2(MAD)/1e3*4 = 109.36 TFLOPS
          160*2.9*32*2/1e3*4 = 118.78 TFLOPS

'''
devinfo = cl.dev_info()
num_EUs = devinfo["CL_DEVICE_MAX_COMPUTE_UNITS"]
max_freqMHz = devinfo["CL_DEVICE_MAX_CLOCK_FREQUENCY"]

SIMD_WIDTH = 16
MAC_PER_LANE = 2 #(2xfp16 VNNI)
DEPTH = 4 #?
OPS_PER_MAC = 2
max_TFLOPS = num_EUs * SIMD_WIDTH * MAC_PER_LANE * DEPTH * OPS_PER_MAC * max_freqMHz * 1e-6
# 118 TFLOPS

regM = 2
regN = 2

src_op_bits = 16
SystolicDepth = 8
VNNI_WIDTH = 32//src_op_bits  # 2 or 4
RepeatCount = 8
M = RepeatCount * regM  # RepeatCount 1~8
K = SystolicDepth * VNNI_WIDTH   # 16 x half  = 256 bits
N = 16 * regN           # 16 x float = 512 bits

AccSize = RepeatCount * 16
Src1Size = K*16//2
Src2Size = RepeatCount*K//2

print(f" {M=} {K=} {N=} {Src1Size=} {Src2Size=} ")

def repack_B(B):
    K, N = B.shape
    newB = np.zeros([K//2, N*2], np.float16)
    for k in range(K//2):
        for n in range(N):
            newB[k, 2*n+0] = B[2*k+0, n]
            newB[k, 2*n+1] = B[2*k+1, n]
    return newB

def pyeval(src):
    result_src = ""
    for line in src.splitlines():
        if line.startswith("#pyeval"):
            new_line = eval(line[8:])
            result_src += new_line + "\n"
            # print(f"[pyeval] {new_line}")
        else:
            result_src += line + "\n"
    return result_src

# MAX-workgroups size for Xe2 is 64: 8 EUs x 8 HW-threads
# when total number of HW-worker-threads are not enough to fully occupy all Xe-Cores
# smaller work-group size allows driver to distribute threads more evenly among all Xe-cores
# thus work-group size must be easily configurable
wgM = 2
wgN = 2

# keep work-group size fixed, GWS is work-group size x work-group count
# each work-group produces BM x BN output :
#   BM = wgM * regM * RepeatCount
#   BN = wgN * regN * 16
BM = wgM * regM * RepeatCount
BN = wgN * regN * 16

# sometime M * N are small but K is big, choose small wgM & wgN and split along K dimension
# can produce more threads and use all Xe-cores

# we need enough of threads to use all Xe-cores
nBM = 20
nBN = 16

total_M = nBM * wgM * regM * RepeatCount
total_N = nBN * wgN * regN * 16

GWS = [nBM * wgM, nBN * wgN]
LWS = [wgM, wgN]

total_threads = GWS[0] * GWS[1]

src = r'''
#include <cm/cm.h>
#include <cm/cmtl.h>

#ifndef CM_HAS_LSC_UNTYPED_2D
#error "CM_HAS_LSC_UNTYPED_2D is required!!!!"
#endif

// due to C++ headers cm.h, passing macro through compiler may cause naming conflict

#pyeval f"#define wgM {wgM}"
#pyeval f"#define wgN {wgN}"
#pyeval f"#define regM {regM}"
#pyeval f"#define regN {regN}"
#pyeval f"#define Src1Size {Src1Size}"
#pyeval f"#define Src2Size {Src2Size}"
#pyeval f"#define RepeatCount {RepeatCount}"
#pyeval f"#define SystolicDepth {SystolicDepth}"
#pyeval f"#define AccSize {AccSize}"
#pyeval f"#define total_N {total_N}"
#pyeval f"#define total_K {total_K}"
#pyeval f"#define numBK {numBK}"

extern "C" _GENX_MAIN_ void cm_dpas_gemm(
    svmptr_t pdst [[type("svmptr_t")]],     // C-matrix
    svmptr_t psrc1 [[type("svmptr_t")]],    // B-matrix
    svmptr_t psrc2 [[type("svmptr_t")]]     // A-matrix
    ) {
    constexpr uint REGM_SIZE = regM * RepeatCount * SystolicDepth * sizeof(int32_t);
    constexpr uint REGN_SIZE = regN * 16 * SystolicDepth * sizeof(int32_t);

    cm_slm_init(wgM * REGM_SIZE + wgN * REGN_SIZE);

    auto slmA = cm_slm_alloc(wgM * REGM_SIZE);
    auto slmB = cm_slm_alloc(wgN * REGN_SIZE);

    auto local_id_M = cm_local_id(0);
    auto local_id_N = cm_local_id(1);

    auto group_id_M = cm_group_id(0);
    auto group_id_N = cm_group_id(1);

    uint M_off = (group_id_M * wgM + local_id_M) * regM * RepeatCount;
    uint N_off = (group_id_N * wgN + local_id_N) * regN * 16;

    //# output
    pdst += (M_off * total_N + N_off) * sizeof(float);

    matrix<float, regM * regN, AccSize> acc;// The matrix C
    matrix<int32_t, regN, Src1Size> src1; // The matrix B
    matrix<int32_t, regM, Src2Size> src2; // The matrix A

    //cm_svm_block_read(psrc1, src1);
    //cm_svm_block_read(psrc2, src2);

    acc = 0.0f;

    #pragma unroll (1)
    for(int nbK = 0; nbK < numBK; nbK++) {
        //# copy from DDR into SLM
        //# sync

        cm_slm_block_read(slmA, GENX_DWALIGNED, local_id_M*REGM_SIZE, src2.format<int32_t>());
        cm_slm_block_read(slmB, GENX_DWALIGNED, local_id_N*REGN_SIZE, src1.format<int32_t>());

        //# insert more work-loads between barrier's signal & wait stage
        //# helps to reduce the synchronization overhead, since the signal event
        //# is non-blocking, big work-loads before wait ensure that when wait
        //# is called, all signal events have been done, thus wait can be done very fast.
        cm_sbarrier(1);

        #pragma unroll (regM)
        for(int m = 0; m < regM; m++) {
            #pragma unroll (regN)
            for(int n = 0; n < regN; n++) {
                acc.row(m*regN + n) = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount>(
                                                acc.row(m*regN + n),
                                                src1.row(n),
                                                src2.row(m));
            }
        }
        //# wait
        cm_sbarrier(0);
    }
    cm_svm_block_write(pdst, acc);

    //printf("Hello,CM\n");
}
'''

def tput_WO_SLM_FP16():
    src =  r'''
#ifdef CM_HAS_LSC_UNTYPED_2D
#pragma message (">>>>>> CM_HAS_LSC_UNTYPED_2D OK")
#else
#error "----------------------------"
#endif
#define SystolicDepth 8
#define regN 16
#define regK 16

    extern "C" _GENX_MAIN_ void gemm(svmptr_t A [[type("svmptr_t")]],
                                  svmptr_t B [[type("svmptr_t")]],
                                  svmptr_t C [[type("svmptr_t")]],
                                  unsigned int M, unsigned int K, unsigned int N) {
        int m_idx = cm_global_id(0) * regM * tileM;
        int n_idx = cm_global_id(1) * regN * tileN;
        matrix<half, tileM, regM * regK> matA;
        matrix<half, tileN, regK*regN> matB;
        matrix<half, tileM*tileN, regM*regN> matC = 0;
        lsc::block_2d_desc<half, 1, regM, regK> descA{(half*)A, M-1, K*2-1, K*2-1, 0, m_idx};
        lsc::block_2d_desc<half, tileN, regK, regN> descB{(half*)B, K-1, 2*N-1, 2*N-1, n_idx, 0};
        lsc::block_2d_desc<half, 1, regM, regN> descC{(half*)C, M-1, 2*N-1, 2*N-1, n_idx, m_idx};

        for (int kb = 0; kb < K; kb+= BK) {
            #pragma unroll
            for (int ki = 0; ki < BK ; ki+=regK) {
                descB.set_block_y(kb+ki);
                cm_load<lsc::LoadOp::VNNI>(matB.format<half>(), descB);
                descA.set_block_x(kb+ki);
                #pragma unroll
                for (int mt_idx = 0; mt_idx <  tileM; mt_idx ++) {
                    descA.set_block_y(m_idx + mt_idx* regM);
                    cm_load<lsc::LoadOp::Normal>(matA[mt_idx].format<half>(), descA);
                }
                int tileC_idx = 0;
                #pragma unroll
                for (int mt_idx = 0; mt_idx <  tileM; mt_idx ++) {
                    #pragma unroll
                    for (int nt_idx = 0; nt_idx <  tileN; nt_idx ++) {
                        matC.row(mt_idx*tileN + nt_idx).format<half>() = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, regM>(
                                matC.row(tileC_idx).format<half>(),
                                matB[nt_idx].format<int32_t>(),
                                matA[mt_idx].format<int32_t>());
                        tileC_idx++;
                    }
                }
            }
        }
        #pragma unroll
        for (int mt_idx = 0; mt_idx <  tileM; mt_idx ++) {
            descC.set_block_y(m_idx + mt_idx*regM);
            #pragma unroll
            for (int nt_idx = 0; nt_idx <  tileN; nt_idx ++) {
                descC.set_block_x(n_idx + nt_idx*regN);
                cm_store(descC, matC[mt_idx*tileN + nt_idx].format<half>());
            }
        }
    }
    '''
    REPEAT = 10
    regM = 8
    regN = 16
    BK = 64
    tileM = 8
    tileN = 2

    nthrM = 8
    nthrN = 8

    BM = nthrM* regM * tileM
    BN = nthrN* regN * tileN

    WGS_M = 8
    WGS_N = 8
    global_nthrM = nthrM*WGS_M
    global_nthrN = nthrN*WGS_N

    M = regM*tileM*global_nthrM
    N = regN*tileN*global_nthrN
    K = BK*32
    # np.random.seed(0)
    vRANGE = 2
    A = np.random.randint(-vRANGE, vRANGE+1, [M, K]).astype(np.float16)
    B = np.random.randint(-vRANGE, vRANGE+1, [K, N]).astype(np.float16)
    C_ref = np.matmul(A, B)
    tA_list = [cl.tensor(A) for _ in range(REPEAT)]
    tB_list = [cl.tensor(B) for _ in range(REPEAT)]
    tC_list = [cl.tensor([M, N], np.dtype(np.float16)) for _ in range(REPEAT)]

    SG_SZ = 16
    kernel =  cl.kernels(src, options=f"-cmc -mdump_asm -g2 -DregM={regM} -DregN={regN} -DtileM={tileM} -DtileN={tileN} -DBK={BK} -BM={BM} -BN={BN}")

    for i in range(0, REPEAT):
        kernel.enqueue("gemm", [global_nthrM, global_nthrN],[nthrM, nthrN], tA_list[i], tB_list[i],tC_list[i], M, K, N)
    ns = cl.finish()
    flops = M * N * K * 2
    print("----------------------------------------------------")
    print(f'M:{M}, N:{N}, K:{K}')
    print("----------------------------------------------------")
    for time_opt in ns:
        print(f'TPUT: [W/O SLM]:{flops/time_opt:.1f} GFLOPS, us: {time_opt*1e-3:.1f}')

    compare(C_ref, tC_list[0].numpy())


cl.profiling(True)


dt_ker_overhead = None

# set LoopCount to 0 to estimate kernel overhead
for numBK in [0, 100, 256, 512]:
#for LoopCount in [0, 0, 0]:
    total_K = numBK * K
    B = np.random.randint(-4,5,[total_K, total_N]).astype(np.float16)
    A = np.random.randint(-4,5,[total_M, total_K]).astype(np.float16)
    if numBK < 200:
        C = A @ B
    else:
        C = None

    tA = cl.tensor(A)
    #tB = cl.tensor(repack_B(B))
    tB = cl.tensor(B)
    tC = cl.tensor(np.zeros([total_M, total_N], np.float32))

    # increase loop count to reduce overhead
    kernels = cl.kernels(pyeval(src), f"-cmc -mdump_asm -g2 ")

    print(f"======== {numBK=} =============")
    valid_test_cnt = 0
    warm_up_time_thr = 0.1
    tbase = time.time()
    while valid_test_cnt < 1:
        t0 = time.time()
        kernels.enqueue("cm_dpas_gemm", GWS, LWS, tC, tB, tA)
        ns = cl.finish()[0]
        t1 = time.time()

        if (t1 - tbase > warm_up_time_thr):
            dt = (t1 - t0)
            dt_ker = ns / 1e9
            if not dt_ker_overhead:
                dt_ker_overhead = dt_ker * 0.95
            dt_ker -= dt_ker_overhead
            dt_host_overhead = dt - dt_ker - dt_ker_overhead

            if C is not None:
                C1 = tC.numpy()
                if not np.allclose(C, C1):
                    #print(C)
                    #print(C1)
                    accinfo = "failed."
                else:
                    accinfo = "passed."
            else:
                accinfo = "______"

            total_K = K * numBK
            tflops = M*N*K*2*numBK*total_threads/dt_ker*1e-12
            #tflops = total_M*total_N*total_K*2/dt*1e-12
            print(f"[{accinfo}]  M,N,K={total_M},{total_N},{total_K}  {dt*1e6:.2f} = {dt_ker*1e6:.2f} + {dt_ker_overhead*1e6:.2f} + {dt_host_overhead*1e6:.2f} uS    {tflops:.3f} TFLOPS  /  {max_TFLOPS:.3f} TFLOPS = {tflops*100/max_TFLOPS:.1f}% ")
            valid_test_cnt += 1


print(f"{total_threads=}   Hyper-Threads: x {total_threads/num_EUs:.2f}")




# test tput.
tput_WO_SLM_FP16()