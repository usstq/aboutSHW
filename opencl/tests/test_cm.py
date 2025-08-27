#!/usr/bin/python3
from clops import cl
import numpy as np
import torch

z = np.zeros([2, 3], dtype=np.float32)
z[0,0] = 1
z[0,1] = 2
z[1,0] = 3
z[1,1] = 4

# t = cl.tensor(z)
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

extern "C" _GENX_MAIN_ void cm_check(svmptr_t src, svmptr_t dest, svmptr_t scale, svmptr_t out) {

    matrix<half, 16, 16> mat;
    vector<half, 16> vec_scales;
    cm_svm_block_read(scale, vec_scales.format<half>());

    auto temp = mat.format<half, 2, 128>().row(1).format<uint8_t, 8, 32>();
    auto input = mat.format<half, 2, 128>().row(0).format<uint8_t, 16, 16>();
    cm_svm_block_read(src, input.format<uint8_t>());
    auto dq_out = mat.format<half, 8, 32>();


    temp.row(0).select<16, 2>(0) = input.row(0);
    temp.row(0).select<16, 2>(1) = input.row(1);

    temp.row(1).select<16, 2>(0) = input.row(2);
    temp.row(1).select<16, 2>(1) = input.row(3);

    temp.row(2).select<16, 2>(0) = input.row(4);

    temp.row(2).select<16, 2>(1) = input.row(5);

    temp.row(3).select<16, 2>(0) = input.row(6);
    temp.row(3).select<16, 2>(1) = input.row(7);


    temp.row(4).select<16, 2>(0) = input.row(8);
    temp.row(4).select<16, 2>(1) = input.row(9);


    temp.row(5).select<16, 2>(0) = input.row(10);
    temp.row(5).select<16, 2>(1) = input.row(11);


    temp.row(6).select<16, 2>(0) = input.row(12);
    temp.row(6).select<16, 2>(1) = input.row(13);


    temp.row(7).select<16, 2>(0) = input.row(14);
    temp.row(7).select<16, 2>(1) = input.row(15);

    cm_svm_block_write(dest, temp.format<uint8_t>());

    vector<half, 32> bk = temp.row(7);

    for (int r = 0; r < 7; r++) {
            dq_out[r].select<16, 2>(0) = cm_mul<half>(temp[r].select<16, 2>(0), vec_scales[r*2]);
            dq_out[r].select<16, 2>(1) = cm_mul<half>(temp[r].select<16, 2>(1),  vec_scales[r*2+1]);
    }

    dq_out[7].select<16, 2>(0) = cm_mul<half>(bk.select<16, 2>(0), vec_scales[14]);
    dq_out[7].select<16, 2>(1) = cm_mul<half>(bk.select<16, 2>(1),  vec_scales[15]);
    cm_svm_block_write(out, dq_out.format<half>());


}
'''

a=torch.randint(0, 3, [16,16]).to(dtype=torch.uint8)
scale=torch.rand(16).to(dtype=torch.half)
dqa=torch.zeros(16,16).to(dtype=torch.half)

for i in range (16):
    dqa[i,:] = a[i,:].to(dtype=torch.half) * scale[i]

print(a)
print(scale)
print(dqa)
b=torch.zeros(16,16).to(dtype=torch.uint8)
tA = cl.tensor(a.detach().numpy())
tB =cl.tensor(b.detach().numpy())
tscale = cl.tensor(scale.detach().numpy())
tC = cl.tensor(torch.zeros(8,32).to(dtype=torch.half).detach().numpy())
testcm.enqueue("cm_check", [1,1],[1,1], tA, tB, tscale, tC)

cl.finish()

expecta_int = a.transpose(0, 1).contiguous().view(dtype=torch.uint16).reshape(16,8).transpose(0,1).contiguous().view(dtype=torch.uint8)
expecta_half = dqa.transpose(0, 1).contiguous().view(dtype=torch.uint32).reshape(16,8).transpose(0,1).contiguous().view(dtype=torch.half)
print(expecta_half)
print(torch.from_numpy(tC.numpy()).reshape(8,32))

print(expecta_half == torch.from_numpy(tC.numpy()).reshape(8,32))

# print(torch.from_numpy(tB.numpy()).reshape(8,32))

# print(expecta_int == torch.from_numpy(tB.numpy()).reshape(8,32))
