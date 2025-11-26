#!/usr/bin/python3
src=r'''

#include <cm/cm.h>
#include <cm/cmtl.h>

template <int M, int N>
CM_INLINE void cm_load_2d(matrix_ref<half, M, N> out, SurfaceIndex base, uint offset, uint pitch) {
    #pragma unroll
    for(int i = 0; i < out.n_rows(); i++) {
        out.row(i).format<uint>() = cm_load<uint, N/2>(base, offset + i * pitch);
    }
}

extern "C" _GENX_MAIN_ void test_stateful_load(SurfaceIndex A [[type("buffer_t")]],
                                             uint offset_in_byte, uint pitch_in_byte,
                                             half* B [[type("svmptr_t")]]) {
    half sum = 0;
    auto id_wg_m = cm_group_id(0);
    auto id_sg_m = cm_local_id(1);
    constexpr uint k_step = K_STEP;
    
    auto m = id_wg_m * BLOCK_WG_M + id_sg_m * BLOCK_SG_M;
    auto offset = offset_in_byte + m * pitch_in_byte;

    #pragma unroll
    for (uint k = 0; k < HEAD_SIZE*STRIDE; k += k_step) {
        matrix<half, 64, k_step> a0;

        cm_load_2d(a0, A, offset + k * 2, pitch_in_byte);

        #pragma unroll
        for(int i = 0; i < a0.n_rows(); i++) {
            // for(int j = 0; j < a0.n_cols(); j++)
            sum += a0.row(i)[0];
        }
    }                                      

    B[m/BLOCK_SG_M] = sum;
}

#ifdef CM_HAS_LSC_UNTYPED_2D
extern "C" _GENX_MAIN_ void test_lsc_2d_load(half* A [[type("svmptr_t")]],
                                           uint M, uint pitch_in_byte,
                                           half* B [[type("svmptr_t")]]) {
    half sum = 0;
    auto id_wg_m = cm_group_id(0);
    auto id_sg_m = cm_local_id(1);
    constexpr uint k_step = K_STEP;
    constexpr uint BLOCK_REG_A = 8*k_step;
    
    auto m = id_wg_m * BLOCK_WG_M + id_sg_m * BLOCK_SG_M;
        
    lsc::block_2d_desc<half, 1, 32, k_step> desc_a{ A, M - 1, (uint)((HEAD_SIZE*STRIDE) * sizeof(half) - 1), (uint)(pitch_in_byte - 1),
        0, (int)(m) };   // 32 = BLOCK_SG_M/2 ?

    #pragma unroll
    for (uint k = 0; k < HEAD_SIZE*STRIDE; k += k_step) {
        matrix<half, 64, k_step> a0;

        cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached, 0,  0>(a0.select<4, 1, BLOCK_REG_A, 1>(0).format<half>(), desc_a);
        cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached, 0, 32>(a0.select<4, 1, BLOCK_REG_A, 1>(4).format<half>(), desc_a);
        
        desc_a.set_block_x(desc_a.get_block_x() + k_step);

        #pragma unroll
        for(int i = 0; i < a0.n_rows(); i++) {
            // for(int j = 0; j < a0.n_cols(); j++)
            sum += a0.row(i)[0];
        }
    }                                      

    B[m/BLOCK_SG_M] = sum;
}
#endif

template <int M, int N>
CM_INLINE void cm_gather_2d(matrix_ref<half, M, N> out, SurfaceIndex base, uint offset, uint pitch_in_byte) {
    constexpr uint UNROLL = 8;
    #pragma unroll
    for (int j = 0; j < M; j += UNROLL) {
        vector<uint, UNROLL> offsets;
        #pragma unroll
        for(int i = 0; i < UNROLL; i++) {
            offsets[i] = offset + (j + i) * pitch_in_byte;
        }

        out.select<UNROLL, 1, N, 1>(j).format<uint>() = cm_load<uint, VectorSize::N8, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(base, offsets);
    }
}

extern "C" _GENX_MAIN_ void test_stateful_gather(SurfaceIndex A [[type("buffer_t")]],
                                             uint offset_in_byte, uint pitch_in_byte,
                                             half* B [[type("svmptr_t")]]) {
    half sum = 0;
    auto id_wg_m = cm_group_id(0);
    auto id_sg_m = cm_local_id(1);
    constexpr uint k_step = K_STEP;
    
    auto m = id_wg_m * BLOCK_WG_M + id_sg_m * BLOCK_SG_M;
    auto offset = offset_in_byte + m * pitch_in_byte;

    #pragma unroll
    for (uint k = 0; k < HEAD_SIZE*STRIDE; k += k_step) {
        matrix<half, BLOCK_SG_M, k_step> a0;

        cm_gather_2d(a0, A, offset + k * 2, pitch_in_byte);

        #pragma unroll
        for(int i = 0; i < a0.n_rows(); i += 1) {
            // # for(int j = 0; j < a0.n_cols(); j++)
                sum += a0.row(i)[0];
            //}
        }
    }

    B[m/BLOCK_SG_M] = sum;
}


extern "C" _GENX_MAIN_ void test_svm_block_read(const svmptr_t A [[type("svmptr_t")]],
                                             uint pitch_in_byte,
                                             half* B [[type("svmptr_t")]]) {
    half sum = 0;
    auto id_wg_m = cm_group_id(0);
    auto id_sg_m = cm_local_id(1);
    constexpr uint k_step = K_STEP;
    
    auto m = id_wg_m * BLOCK_WG_M + id_sg_m * BLOCK_SG_M;
    auto base_addr = A + m * pitch_in_byte;

    #pragma unroll
    for (uint k = 0; k < HEAD_SIZE*STRIDE; k += k_step) {
        matrix<half, BLOCK_SG_M, k_step> a0;

        #pragma unroll
        for(int i = 0; i < BLOCK_SG_M; i++){
            cm_svm_block_read<half, k_step>(base_addr + (k * 2 + i * pitch_in_byte), a0.row(i));
        }

        #pragma unroll
        for(int i = 0; i < a0.n_rows(); i += 1) {
            // # for(int j = 0; j < a0.n_cols(); j++)
                sum += a0.row(i)[0];
            //}
        }
    }

    B[m/BLOCK_SG_M] = sum;
}

'''

import os
from clops import cl
import numpy as np
import torch

torch.manual_seed(3)
torch.set_printoptions(linewidth=1024)

cl.profiling(True)

SG_M = 8
BLOCK_SG_M = 16   # each thread processes 64 lines
BLOCK_WG_M = BLOCK_SG_M * SG_M   # each workgroup processes 256 lines

K_STEP = 16

STRIDE = 16
HEAD_SIZE = 128

cwd = os.path.dirname(os.path.realpath(__file__))
jit_option = '-abortonspill -noschedule '
debug_option = '-mCM_printregusage -mdump_asm -g2'
k = cl.kernels(src, options=f'''-cmc -Qxcm_jit_option="{jit_option}"
               {debug_option}
               -Qxcm_register_file_size=256 -I{cwd}
               -DSG_M={SG_M} -DBLOCK_SG_M={BLOCK_SG_M} -DBLOCK_WG_M={BLOCK_WG_M}
               -DHEAD_SIZE={HEAD_SIZE} -DSTRIDE={STRIDE} -DK_STEP={K_STEP}
               ''')

H = 32
M, K = H*(32*1024)//STRIDE, HEAD_SIZE*STRIDE
A = torch.randint(-3, 3, size=[M, K], dtype=torch.int16).to(dtype=torch.float16).contiguous()

M_groups=M//BLOCK_WG_M
gws = [M_groups, SG_M]
lws = [1, SG_M]

B = torch.zeros([M//BLOCK_SG_M], dtype=torch.float16).contiguous()

total_bytes = M * K * 2 + (M//BLOCK_SG_M) * 2
flops = M * (HEAD_SIZE * STRIDE // K_STEP)
n_repeats = 10
for j in range(0, n_repeats):
    k.enqueue("test_stateful_load", gws, lws, cl.tensor(A.numpy()), 0, K*2, cl.tensor(B.numpy()))
    latency = cl.finish()
    for i, ns in enumerate(latency):
        print(f'(test_stateful_load [{j},{i}]) TPUT:{flops/ns:,.0f} GFLOPS, BW: {total_bytes / ns:,.2f} GB/s with [{total_bytes*1e-6:,} MB] during {ns*1e-3:,.0f} us')
        
# for j in range(0, n_repeats):
#     k.enqueue("test_lsc_2d_load", gws, lws, cl.tensor(A.numpy()), M, K*2, cl.tensor(B.numpy()))
#     latency = cl.finish()
#     for i, ns in enumerate(latency):
#         print(f'(test_lsc_2d_load [{j},{i}]) TPUT:{flops/ns:,.0f} GFLOPS, BW: {total_bytes / ns:,.2f} GB/s with [{total_bytes*1e-6:,} MB] during {ns*1e-3:,.0f} us')
        
for j in range(0, n_repeats):
    k.enqueue("test_stateful_gather", gws, lws, cl.tensor(A.numpy()), 0, K*2, cl.tensor(B.numpy()))
    latency = cl.finish()
    for i, ns in enumerate(latency):
        print(f'(test_stateful_gather [{j},{i}]) TPUT:{flops/ns:,.0f} GFLOPS, BW: {total_bytes / ns:,.2f} GB/s with [{total_bytes*1e-6:,} MB] during {ns*1e-3:,.0f} us')
        
for j in range(0, n_repeats):
    k.enqueue("test_svm_block_read", gws, lws, cl.tensor(A.numpy()), K*2, cl.tensor(B.numpy()))
    latency = cl.finish()
    for i, ns in enumerate(latency):
        print(f'(test_svm_block_read [{j},{i}]) TPUT:{flops/ns:,.0f} GFLOPS, BW: {total_bytes / ns:,.2f} GB/s with [{total_bytes*1e-6:,} MB] during {ns*1e-3:,.0f} us')