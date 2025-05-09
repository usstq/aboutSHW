src=r'''
#include <cm/cm.h>
#include <cm/cmtl.h>

extern "C" _GENX_MAIN_ void trans(svmptr_t srcA [[type("svmptr_t")]],
                                  svmptr_t srcB [[type("svmptr_t")]], int M, int K) {
    int k = cm_global_id(0);
    half* src = reinterpret_cast<half*>(srcA) + k;
    half* dst = reinterpret_cast<half*>(srcB) + k*M;
    for(int m = 0; m < M; m++) {
        dst[m] = src[m*K];
    }
}

extern "C" _GENX_MAIN_ void sgemm(svmptr_t srcA [[type("svmptr_t")]],
                                  svmptr_t srcB [[type("svmptr_t")]],
                                  svmptr_t dstC [[type("svmptr_t")]],
                                  int M, int N, int K) {
    constexpr int BK = SGEMM_BLOCK_K;
    cm_slm_init(1024*64);

    // the offset returned is the same for all threads within current group

    auto nthrM = cm_local_size(0);
    auto nthrN = cm_local_size(1);

    auto ithrM = cm_local_id(0);
    auto ithrN = cm_local_id(1);

    uint slmA0 = cm_slm_alloc(nthrM*BK*regM * sizeof(half));
    uint slmB0 = cm_slm_alloc(nthrN*BK*regN * sizeof(half));

    uint slmA1 = cm_slm_alloc(nthrM*BK*regM * sizeof(half));
    uint slmB1 = cm_slm_alloc(nthrN*BK*regN * sizeof(half));

    //SLM A: [nthrM, BK, regM] => block-read a column
    //SLM B: [nthrN, BK, regN] => block-read a row

    slmA0 += ithrM * (BK*regM) * sizeof(half);
    slmB0 += ithrN * (BK*regN) * sizeof(half);

    slmA1 += ithrM * (BK*regM) * sizeof(half);
    slmB1 += ithrN * (BK*regN) * sizeof(half);

    srcB += cm_global_id(1) * (regN * sizeof(half));
    srcA += cm_global_id(0) * (regM * sizeof(half));

    dstC += cm_global_id(0) * (regM * sizeof(half) * N);
    dstC += cm_global_id(1) * (regN * sizeof(half));

    matrix<half, regM, regN> C;
    vector<half, regM> A;
    vector<half, regN> B;
    vector<half, regM> tempA;
    vector<half, regN> tempB;
    C = 0;

    uint slm_id = 0;
    uint readA = slm_id & 1? slmA0 : slmA1;
    uint readB = slm_id & 1? slmB0 : slmB1;

    // load first block into SLM
    uint writeA = readA;
    uint writeB = readB;
    for(int k = 0; k < BK; k++) {
        // copy data from VRAM to SLM is hidden
        cm_svm_block_read_unaligned(srcA, tempA);
        cm_svm_block_read_unaligned(srcB, tempB);
        srcA += M*sizeof(half);
        srcB += N*sizeof(half);
        cm_slm_block_write(writeA, 0, tempA);
        cm_slm_block_write(writeB, 0, tempB);
        writeA += regM*sizeof(half);
        writeB += regN*sizeof(half);
    }

    writeA = slm_id & 1? slmA1 : slmA0;
    writeB = slm_id & 1? slmB1 : slmB0;

    for(int k0 = 0; k0 < K - BK; k0 += BK) {
        for(int k = 0; k < BK; k++) {
            // load from SLM also has latency
            cm_slm_block_read(readA, 0, A);
            cm_slm_block_read(readB, 0, B);

            // copy data from VRAM to SLM is hidden
            cm_svm_block_read_unaligned(srcA, tempA);
            cm_svm_block_read_unaligned(srcB, tempB);
            srcA += M*sizeof(half);
            srcB += N*sizeof(half);

            #pragma unroll
            for(int m = 0; m < regM; m++) {
                C.row(m) += A.replicate<regN, 1>(m) * B;
            }

            cm_slm_block_write(writeA, 0, tempA);
            cm_slm_block_write(writeB, 0, tempB);
            writeA += regM*sizeof(half);
            writeB += regN*sizeof(half);

            readA += regM*sizeof(half);
            readB += regN*sizeof(half);
        }

        // post-log
        cm_sbarrier(1);
        slm_id ++;
        readA = slm_id & 1? slmA0 : slmA1;
        readB = slm_id & 1? slmB0 : slmB1;
        writeA = slm_id & 1? slmA1 : slmA0;
        writeB = slm_id & 1? slmB1 : slmB0;
        cm_slm_fence(CM_GLOBAL_COHERENT_FENCE);
        cm_sbarrier(0);
    }

    // last Block K w/o read data into SLM
    for(int k = 0; k < BK; k++) {
        // load from SLM also has latency
        cm_slm_block_read(readA, 0, A);
        cm_slm_block_read(readB, 0, B);
        #pragma unroll
        for(int m = 0; m < regM; m++) {
            C.row(m) += A.replicate<regN, 1>(m) * B;
        }
        readA += regM*sizeof(half);
        readB += regN*sizeof(half);
    }

    #pragma unroll
    for(int m = 0; m < regM; m++) {
        cm_svm_block_write<half, regN>(dstC, C.row(m));
        dstC += N*sizeof(half);
    }
}
'''

# export CM_FE_DIR=~/tingqian/

r'''
160 EU, SIMD-16 ALU (32 flops)
Local memory size                               131072 (128*1024 = 128 KiB)

frequency: 2.67HGz

https://www.techpowerup.com/gpu-specs/arc-b580.c4244
FP32 : 160*2.67*16(SIMD-width)*2(MAD)/1e3 = 13.67 TFLOPS
FP16 : 160*2.67*32(SIMD-width)*2(MAD)/1e3 = 27.34 TFLOPS

SIMD-16 means GRF width: 16x4= 64bytes
GRF size per HW-thread: 8KB
GRF height(rows or count): 8192/64 = 128

total_threads: 160:       4.3 TFLOPS
total_threads: 160*2:     8.6 TFLOPS
total_threads: 160*4:    17.0 TFLOPS
total_threads: 160*8:    25.0 TFLOPS
       read from SLM:    26.3 TFLOPS
HW-threads are dispatched to local slice as much as possible first?

'''

from clops import cl
import numpy as np

cl.profiling(True)

def test(K, chk_accuracy = False):
    regM = 16
    regN = 48

    nthrM = 8
    nthrN = 8
    global_nthrM = nthrM*5
    global_nthrN = nthrN*4
    M = regM*global_nthrM
    N = regN*global_nthrN
    SGEMM_BLOCK_K = 32

    SLMA = nthrM*SGEMM_BLOCK_K*regM *2
    SLMB = nthrN*SGEMM_BLOCK_K*regN *2

    print(f"M={M} N={N} K={K}")
    print(f"SLMA={SLMA/1024:.1f}K SLMB={SLMB/1024:.1f}K")

    np.random.seed(0)
    A = np.random.randint(-1,2,[M, K]).astype(np.float16)
    B = np.random.randint(-1,2,[K, N]).astype(np.float16)

    if chk_accuracy:
        C = A @ B
        print(f"refernce is calculated")

    k = cl.kernels(src, f"-cmc -march=BMG -mdump_asm -g2 -DregM={regM} -DregN={regN} -DSGEMM_BLOCK_K={SGEMM_BLOCK_K}")

    tA0 = cl.tensor(A)
    tA = cl.tensor(np.zeros([K, M], dtype=np.float16))
    tB = cl.tensor(B)
    tC = cl.tensor(np.zeros([M, N], dtype=np.float16))

    if chk_accuracy:
        k.enqueue("trans", [K],[1], tA0, tA, M, K)
        print(f"A is transposed")

    #
    for _ in range(20):
        k.enqueue("sgemm", [global_nthrM, global_nthrN],[nthrM, nthrN], tA, tB, tC, M, N, K)

    times = cl.finish()
    for ns in times:
        print(f"{ns/1e3:.3f} us  {M*N*K*2/ns:.2f} GFLOPS/s   ")

    if chk_accuracy:
        if not np.allclose(C, tC.numpy()):
            print("===============C")
            print(C)
            print("===============tC")
            print(tC.numpy())
            assert False

test(256, chk_accuracy=True)
#test(4096, chk_accuracy=False)
# test(40960, chk_accuracy=False)