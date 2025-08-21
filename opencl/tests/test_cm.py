#!/usr/bin/python3
from clops import cl
import numpy as np

z = np.zeros([2, 3], dtype=np.float32)
z[0,0] = 1
z[0,1] = 2
z[1,0] = 3
z[1,1] = 4

t = cl.tensor(z)
# print(f"t.shape={t.shape}, t.numpy()=\n{t.numpy()}")


# the decorator `cl.source` turns testcm() into a compiled opencl kernel object
# by invoking clbuild inside.
#
# thanks to `-mdump_asm -g2`, we can find following dumps after running:
#    vISA asm file:   cm_check.visaasm
#    GEN asm file:    cm_check.asm (with interleaved src-code & line-number)
@cl.source("-cmc -mdump_asm -g2")
def testcm():
    return r'''
#include <cm/cm.h>
#include <cm/cmtl.h>




template<typename T, int M, int N>
void show(const matrix<T, M, N> mat) {
    printf("Matrix [%d, %d]:\n", M, N);
    for(int m = 0; m < M; m ++) {
        printf("\t[");
        for(int n = 0; n < N; n ++) {
            printf("%d,", mat[m][n]);
        }
        printf("],\n");
    }
    printf("]\n");
}


extern "C" _GENX_MAIN_ void cm_check(svmptr_t dst) {
    matrix<int, 16, 16> atten(1);
    vector<int, 16> q;
    cmtl::cm_vector_assign(q, 14, 1);
    *(int*)(dst) = 0;
    show(q.format<int, 1, 16>());
    for (int kv = 0; kv < 16; kv++) {
        auto vec = atten.row(kv%16);
    #if 1
        SIMD_IF_BEGIN(q < kv) {
            vec = 0;
        } SIMD_IF_END;
    #endif
    }

    show(atten.format<int, 16, 16>());

    atten = 1;

    for (int kv = 16; kv < 32; kv++) {
        auto vec = atten.row(kv%16);
    #if 1
        SIMD_IF_BEGIN(q < kv) {
            vec = 0;
        } SIMD_IF_END;
    #endif
    }
    show(atten.format<int, 16, 16>());

    atten = 1;

    for (int kv = 32; kv < 48; kv++) {
        auto vec = atten.row(kv%16);
    #if 1
        SIMD_IF_BEGIN(q < kv) {
            vec = 0;
        } SIMD_IF_END;
    #endif
    }
    show(atten.format<int, 16, 16>());

}
'''

testcm.enqueue("cm_check", [1,1],[1,1], t)

cl.finish()

# print(t.numpy())

