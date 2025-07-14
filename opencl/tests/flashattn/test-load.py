import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)
torch.set_printoptions(linewidth=1024)

# prototyping CM kernels
from clops import cl
import numpy as np
import time


A = np.random.randint(-7, 8,[128, 128]).astype(np.float16)

print(A[:16, :16])

src = r'''

template<typename T, int M, int N>
void show(const matrix<T, M, N> mat) {
    printf("Matrix [%d, %d]:\n", M, N);
    for(int m = 0; m < M; m ++) {
        printf("\t[");
        for(int n = 0; n < N; n ++) {
            printf("%8.4f,", mat[m][n]);
        }
        printf("],\n");
    }
    printf("]\n");
}

template <typename T, int M, int N>
CM_INLINE void svm_read_2d(matrix_ref<T, M, N> out, svmptr_t base, uint pitch) {
    #pragma unroll
    for(int i = 0; i < out.n_rows(); i++) {
        cm_svm_block_read(base + i * pitch, out[i]);
    }
}

extern "C" _GENX_MAIN_ void cm_test(const half* A [[type("svmptr_t")]], int pitch) {
    matrix<half, 16, 16> in;
    svm_read_2d(in, (svmptr_t)A, pitch);
    show(in);
}

extern "C" _GENX_MAIN_ void cm_test2(SurfaceIndex A [[type("buffer_t")]], int pitch) {
    matrix<half, 16, 16> in;
    
    auto r = cm_load<uint, 8>(A, 0);
    show(r.format<half, 1, 16>());

    
    vector<unsigned, 16> Offsets;
    for(int i = 0; i < 16; i++) Offsets[i] = i*pitch;
    in.format<uint>() = cm_load<uint, VectorSize::N8>(A, Offsets);
    show(in);
}
'''

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

t_A = cl.tensor(A)

print(t_A.addr)

t_A.offset = t_A.strides[0] * 2
t_A.offset = 2*2

print(t_A.addr)
cm_kernels = cl.kernels(pyeval(src), f"-cmc -mdump_asm -g2 ")
cm_kernels.enqueue("cm_test", [1], [1], t_A, t_A.strides[0] * 2)
cm_kernels.enqueue("cm_test2", [1], [1], t_A, t_A.strides[0] * 2)
cl.finish()
