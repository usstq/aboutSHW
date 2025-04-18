#!/usr/bin/python3
from clops import cl
import numpy as np

z = np.zeros([2, 3], dtype=np.float32)
z[0,0] = 1
z[0,1] = 2
z[1,0] = 3
z[1,1] = 4

t = cl.tensor(z)
print(f"t.shape={t.shape}, t.numpy()=\n{t.numpy()}")


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

extern "C" _GENX_MAIN_ void cm_check(svmptr_t dst) {
    //unsigned int id = cm_linear_global_id();
    vector<float, 4> v1;

    v1 = 1.23f;

    cm_svm_block_write(dst, v1);
    printf("Hello,CM\n");
}
'''

testcm.enqueue("cm_check", [1,2],[1,1], t)

cl.finish()

print(t.numpy())

