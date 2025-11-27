import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)
torch.set_printoptions(linewidth=1024)

# prototyping CM kernels
from clops import cl
import numpy as np
import time


A = np.random.randint(-7, 8,[11, 128]).astype(np.float16)

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

    // # stateless load
    // # compiler throws error if using half - "Transposed load can work only with U32 and U64 data sizes"
    // # but the result is still normal without transposed actually.
    matrix<half, 16, 32> in2;
    for (uint i = 0; i < 16; i++) {
        in2.select<1,1,32,1>(i,0).format<uint>() = cm_ptr_load<uint, VectorSize::N16>((const unsigned int *const)A, i*pitch);
    }
    show(in2);

    // # stateless gather
    // # 1. transposed load
    // # 2. VS no more than N8. Checked with LNL/DG2
    vector<unsigned, 16> Offsets;
    for(int i = 0; i < 16; i++) Offsets[i] = i*pitch;
    auto in3 = cm_ptr_load<uint, VectorSize::N8>((const unsigned int *const)A, Offsets);
    show(in3.format<half, 16, 16>());
}

extern "C" _GENX_MAIN_ void cm_test2(SurfaceIndex A [[type("buffer_t")]], int pitch) {
    matrix<half, 16, 16> in;
    // # stateful load
    auto r = cm_load<uint, 8>(A, 0);
    show(r.format<half, 1, 16>());

    // # stateful gather
    // # 1. transposed load
    // # 2. VS no more than N8. Checked with LNL/DG2
    vector<unsigned, 16> Offsets;
    for(int i = 0; i < 16; i++) Offsets[i] = i*pitch;
    in.format<uint>() = cm_load<uint, VectorSize::N8>(A, Offsets);
    show(in);
}

//# LSC load
#ifdef CM_HAS_LSC_UNTYPED_2D
extern "C" _GENX_MAIN_ void cm_test3(const half* A [[type("svmptr_t")]], int pitch) {
    matrix<half, 16, 16> in;

    auto r = cm_ptr_load<half, 16>(A, 127, 127, pitch-1, 0, 0);
    show(r.format<half, 1, 16>());


    in.format<half>() = cm_ptr_load<half, 16, 16>(A, 127, 127, pitch-1, 0, 0);
    show(in);
}
#endif

//# svm block read
extern "C" _GENX_MAIN_ void cm_test4(const half* A [[type("svmptr_t")]], int pitch_in_elem) {
    matrix<half, 16, 16> in;

    for (uint i = 0; i < 16; i++) {
        cm_svm_block_read<half, 16>((svmptr_t)((half*)A + (i * pitch_in_elem)), in.row(i));
    }
    show(in);

    vector<svmptr_t, 16> v_addrs;
    for (uint i = 0; i < 16; i++) {
        v_addrs[i] = (svmptr_t)((half*)A + (i * pitch_in_elem));
    }
    vector<half, 16> v_src;
    cm_svm_scatter_read<half, 16>(v_addrs, v_src);
    show(v_src.format<half, 1, 16>());
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
cm_kernels = cl.kernels(pyeval(src), f"-cmc -mCM_printregusage -mdump_asm -g2 ")
cm_kernels.enqueue("cm_test", [1], [1], t_A, t_A.strides[0] * 2)
cm_kernels.enqueue("cm_test2", [1], [1], t_A, t_A.strides[0] * 2)
# cm_kernels.enqueue("cm_test3", [1], [1], t_A, t_A.strides[0] * 2)
cm_kernels.enqueue("cm_test4", [1], [1], t_A, t_A.strides[0] * 2)
cl.finish()
