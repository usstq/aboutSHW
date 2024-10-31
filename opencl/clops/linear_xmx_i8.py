cl_kernel_sources = r'''ulong __attribute__((overloadable)) intel_get_cycle_counter( void );
uint __attribute__((overloadable)) intel_get_slice_id( void );
uint __attribute__((overloadable)) intel_get_subslice_id( void );
uint __attribute__((overloadable)) intel_get_dual_subslice_id( void );
uint __attribute__((overloadable)) intel_get_eu_id( void );
uint __attribute__((overloadable)) intel_get_eu_thread_id( void );

/*********************************************************************************
 matmul: D = A*B' + C
 in : A[M, K], B[N, K] and C[M,N] with 8-bit int
 out: [M, N] 32-bit int
*********************************************************************************/

#define unroll_for __attribute__((opencl_unroll_hint)) for

#define BLOCK_K 32
#define BLOCK_B 8
#define BLOCK_F 8
#define TILE_B 8

void print_input_int8(int8 result) {
    char4 temp[8];
    unroll_for(uint i = 0; i < 8; i++) {
        temp[i] = as_char4(result[i]);
    }
    uint sglid = (uint)get_sub_group_local_id();
    printf("input[%d] = [%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d]\n", sglid,
            temp[0][0], temp[0][1],temp[0][2],temp[0][3],
            temp[1][0], temp[1][1],temp[1][2],temp[1][3],
            temp[2][0], temp[2][1],temp[2][2],temp[2][3],
            temp[3][0], temp[3][1],temp[3][2],temp[3][3],
            temp[4][0], temp[4][1],temp[4][2],temp[4][3],
            temp[5][0], temp[5][1],temp[5][2],temp[5][3],
            temp[6][0], temp[6][1],temp[6][2],temp[6][3],
            temp[7][0], temp[7][1],temp[7][2],temp[7][3]);
}

void print_result_int8(int8 result) {
    uint sglid = (uint)get_sub_group_local_id();
    printf("result[%d] = [%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d]\n", sglid,
            result[0], result[1],result[2],result[3],
            result[4], result[5],result[6],result[7]);
}


void print_wi_info(uint out_b, uint out_f) {
    printf("global_id = (%d,%d,%d), local_id = (%d,%d,%d), group_id = (%d,%d,%d), subgroup_id=%d, subgroup_size=%d, subgroup_local_id=%d, out_b = %d, out_f = %d\n",
                get_global_id(0), get_global_id(1),get_global_id(2),
                get_local_id(0), get_local_id(1),get_local_id(2),
                get_group_id(0), get_group_id(1),get_group_id(2),
                get_sub_group_id(),get_sub_group_size(),
                get_sub_group_local_id(), out_b, out_f);
}

/****************************************************************************************************************************
XVE version:
   1. Tile 8x4 for wi, 8x32 for subgroup/SIMD
   2. M=1024, N=1024, K=2560, 1.2 ms
****************************************************************************************************************************/
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void matmul_bf_tile(__global const char* input,
                             __global const char* weights,
                             __global char* biases,
                             __global int* output,
                             int M, int N, int K) {
    uint gid = (uint)get_global_id(0);
    uint sglid = (uint)get_sub_group_local_id();
    uint SIMD = (uint)get_sub_group_size();

    uint sg_id = gid/SIMD;
    uint sg_block_f = sg_id % (N/SIMD);
    uint sg_block_b = sg_id / (N/SIMD);
    uint out_f = sg_block_f * BLOCK_F;
    uint out_b = sg_block_b * BLOCK_B;

    int acc[BLOCK_B] = { };
    char4  in_0[BLOCK_B] = { };
    char4  wei = { };

    // A x B
    uint iterations = K / BLOCK_K;
    __attribute__((opencl_unroll_hint(1)))
    for (uint ni = 0; ni < iterations; ++ni) {
        // Load input.
        uint input_offset = out_b * K + ni * BLOCK_K;
        unroll_for(uint bi = 0; bi < BLOCK_B; bi++) {
            in_0[bi] = as_char4(intel_sub_group_block_read((const __global uint*)(input + input_offset)));
            input_offset += K;
        }
        unroll_for(uint ki = 0; ki < BLOCK_K; ki += 4) {
            uint weights_offset = out_f * K + sglid * K + ki + ni * BLOCK_K;
            wei=as_char4(*(const __global int*)(weights + weights_offset));
            unroll_for (uint bi = 0; bi < BLOCK_B; ++bi) {
                char4 in_val = intel_sub_group_shuffle(in_0[bi], (ki/4) % SIMD);
                // printf("sgid = %d, in_val[%d] = [%d,%d,%d,%d]\n", sglid, ki, in_val[0], in_val[1],in_val[2],in_val[3]);
                acc[bi] += in_val[0] * wei[0] + in_val[1] * wei[1] + in_val[2] * wei[2] + in_val[3] * wei[3];
                // Require __opencl_c_integer_dot_product_input_4x8bit
                //int temp = dot(in_val, wei);
            }
        }
    }

    // BIAS
    char bias[TILE_B] = {};
    uint bias_offset = out_f  + out_b * N;
    unroll_for (uint bi = 0; bi < TILE_B; ++bi) {
        bias[bi] = biases[bias_offset + sglid];
        bias_offset += N;
    }

    int result[TILE_B] = { };
    unroll_for (uint bi = 0; bi < TILE_B; ++bi) {
        result[bi] = as_int(acc[bi] + bias[bi]);
    }

    // Write results
    uint output_offset = out_f  + out_b * N ;
    // printf("(gid=%d, SIMD=%d, sglid=%d) - (out_b=%d, out_f=%d): input_offset = %d\n", gid, SIMD, sglid, out_b, out_f, output_offset);
    unroll_for(uint bi = 0; bi < TILE_B; bi++) {
        intel_sub_group_block_write((__global uint *)(output + output_offset), result[bi]);
        output_offset += N;
    }
}

/****************************************************************************************************************************
XMX version 1: Base version
     1. Without memory share in subgroup
     2. M=1024, N=1024, K=2560, 0.644 ms
****************************************************************************************************************************/
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void matmul_bf_xmx_1(__global char* input,
                             __global const char* weights,
                             __global char* biases,
                             __global int* output,
                             int M, int N, int K) {
    uint gid = (uint)get_global_id(0);
    uint sglid = (uint)get_sub_group_local_id();
    uint SIMD = (uint)get_sub_group_size();

    uint sg_id = gid/SIMD;
    uint sg_block_f = sg_id % (N / SIMD);
    uint sg_block_b = sg_id / (N / SIMD);
    uint out_f = sg_block_f * BLOCK_F;
    uint out_b = sg_block_b * BLOCK_B;

    int8 result = 0;
    int8 a,b,c={};
    unroll_for(uint ki = 0; ki < K; ki += BLOCK_K) {
        uint input_offset = out_b * K + ki;
        unroll_for(uint bi =0; bi < BLOCK_B; bi++) {
            char4 temp;
            unroll_for(uint it = 0; it < 4; it++) {
                temp[it] = input[input_offset + sglid * 4 + it];
            }
            a[bi]= as_int(temp);
            input_offset += K;
        }
        uint weights_offset = out_f * K + sglid * K + ki;
        unroll_for(uint fi =0; fi < BLOCK_F; fi++) {
            char4 temp;
            unroll_for(uint it = 0; it < 4; it++) {
                temp[it]=weights[weights_offset + fi * 4 + it];
            }
            b[fi]=as_int(temp);
        }
        result += intel_sub_group_i8_i8_matrix_mad_k32(a,b,c);
    }

    uint bias_offset = out_b * N + out_f;
    unroll_for(uint bi =0; bi < BLOCK_B; bi++) {
        result[bi] += biases[bias_offset + sglid];
        bias_offset += N;
    }

    uint output_offset = out_b * N + out_f;
    unroll_for(uint bi =0; bi < BLOCK_B; bi++) {
        output[output_offset + sglid] = result[bi];
        output_offset += N;
    }
}
/****************************************************************************************************************************
XMX version 2
     1. Input data shared in subgroup
     2. WI: 1x8 MADs each iter, subgroup: 1x8xSIMD=64 MADs each iter
     3. M=1024, N=1024, K=2560, 0.219 ms
****************************************************************************************************************************/
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void matmul_bf_xmx_2(__global const char* input,
                             __global const char* weights,
                             __global const char* biases,
                             __global int* output,
                             int M, int N, int K) {
    uint gid = (uint)get_global_id(0);
    uint sglid = (uint)get_sub_group_local_id();
    uint SIMD = (uint)get_sub_group_size();
    uint sg_id = gid/SIMD;
    uint sg_block_f = sg_id % (N / SIMD);
    uint sg_block_b = sg_id / (N / SIMD);
    uint out_f = sg_block_f * BLOCK_F;
    uint out_b = sg_block_b * BLOCK_B;
    // print_wi_info(out_f, out_b);

    int8 result = {}, c = {};
    unroll_for(uint ki = 0; ki < K; ki += BLOCK_K) {
        uint input_offset = out_b * K + ki;
        int8 a, b;
        unroll_for(uint bi = 0;bi < BLOCK_B; bi++) {
            a[bi] = as_int(intel_sub_group_block_read((const __global uint*)(input + input_offset)));
            input_offset += K;
        }
        // print_input_int8(a);
        b=*(const __global int8*)(weights + out_f * K + sglid * K + ki );
        // print_input_int8(b);
        result += intel_sub_group_i8_i8_matrix_mad_k32(a,b,c);
        // print_result_int8(result);
    }
    // print_result_int8(result);
    uint bias_offset = out_b * N + out_f;
    unroll_for(uint bi = 0; bi < BLOCK_B; bi++) {
        result[bi] += biases[bias_offset + sglid];
        bias_offset += N;
    }

    uint output_offset = out_b * N + out_f;
    unroll_for(uint bi = 0; bi < BLOCK_B; bi++) {
        intel_sub_group_block_write((__global uint *)(output + output_offset), result[bi]);
        output_offset += N;
    }
}

/****************************************************************************************************************************
XMX version 3:
     1. Input + weights data shared in subgroup
     2. WI: 1x8 MADs each iter, subgroup: 1x8xSIMD=64 MADs each iter
     3. M=1024, N=1024, K=2560, 0.220 ms
****************************************************************************************************************************/
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void matmul_bf_xmx_3(__global char* input,
                             __global const char* weights,
                             __global char* biases,
                             __global int* output,
                             int M, int N, int K) {
    uint gid = (uint)get_global_id(0);
    uint sglid = (uint)get_sub_group_local_id();
    uint SIMD = (uint)get_sub_group_size();

    uint sg_id = gid/SIMD;
    uint sg_block_f = sg_id % (N / SIMD);
    uint sg_block_b = sg_id / (N / SIMD);
    uint out_f = sg_block_f * BLOCK_F;
    uint out_b = sg_block_b * BLOCK_B;
    int8 a, b, c, wei, result = {};

    unroll_for(uint ki = 0; ki < K; ki += BLOCK_K) {
        uint input_offset = out_b * K + ki;
        uint weights_offset = out_f * K + ki;
        unroll_for(uint fi = 0; fi < BLOCK_F; fi++) {
            wei[fi] = as_int(intel_sub_group_block_read((const __global uint*)(weights + weights_offset)));
            weights_offset += K;
        }
        unroll_for(uint bi = 0; bi < BLOCK_B; bi++) {
            a[bi] = as_int(intel_sub_group_block_read((const __global uint*)(input + input_offset)));
            input_offset += K;
        }

        #if 1
        int8 data[8];
        unroll_for(uint bi = 0; bi < BLOCK_B; bi++) {
            unroll_for(uint i = 0; i < BLOCK_F; i ++) {
                data[i][bi] = intel_sub_group_shuffle(wei[i], bi);
            }
        }
        b = data[sglid];
        #else
        unroll_for(uint bi = 0; bi < BLOCK_B; bi++) {
            // b[bi] = intel_sub_group_shuffle(wei[sglid], bi);
            b[bi] = sub_group_broadcast(wei[sglid], bi);
        }
        #endif
        result += intel_sub_group_i8_i8_matrix_mad_k32(a,b,c);
    }

    uint bias_offset = out_b * N + out_f;
    unroll_for(uint bi = 0; bi < BLOCK_B; bi++) {
        result[bi] += biases[bias_offset + sglid];
        bias_offset += N;
    }

    uint output_offset = out_b * N + out_f;
    unroll_for(uint bi = 0; bi < BLOCK_B; bi++) {
        intel_sub_group_block_write((__global uint *)(output + output_offset), result[bi]);
        output_offset += N;
    }
}

/****************************************************************************************************************************
XMX version 4:
     1. Input data shared in subgroup, weight shared in 4 x dM in M dimension 
     2. WI: 4x8 MADs each iter, subgroup: 4x8xSIMD=256 MADs each iter
     3. M=1024, N=1024, K=2560, 0.143 ms
****************************************************************************************************************************/
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void matmul_bf_xmx_4(__global char* input,
                             __global const char* weights,
                             __global char* biases,
                             __global int* output,
                             int M, int N, int K) {
    uint gid = (uint)get_global_id(0);
    uint sglid = (uint)get_sub_group_local_id();
    uint SIMD = (uint)get_sub_group_size();
    #define BLOCK_NUM 4

    uint sg_id = gid/SIMD;
    uint sg_block_f = sg_id % (N / SIMD);
    uint sg_block_b = sg_id / (N / SIMD);
    uint out_f = sg_block_f * BLOCK_F;
    uint out_b = sg_block_b * BLOCK_B * BLOCK_NUM;

    int8 result[BLOCK_NUM] = {}, c;
    unroll_for(uint ki = 0; ki < K; ki += BLOCK_K) {
        uint input_offset = out_b * K + ki;
        int8 a[BLOCK_NUM], b;

        unroll_for(uint num = 0; num < BLOCK_NUM; num++) {
            unroll_for(uint bi = 0;bi < BLOCK_B; bi++) {
                a[num][bi] = as_int(intel_sub_group_block_read((const __global uint*)(input + input_offset)));
                input_offset += K;
            }
        }
        b=*(const __global int8*)(weights + out_f * K + sglid * K + ki );

        unroll_for(uint num = 0; num < BLOCK_NUM; num++) {
            result[num] += intel_sub_group_i8_i8_matrix_mad_k32(a[num],b,c);
        }
    }

    uint bias_offset = out_b * N + out_f;
    unroll_for(uint num = 0; num < BLOCK_NUM; num++) {
        unroll_for(uint bi = 0; bi < BLOCK_B; bi++) {
            result[num][bi] += biases[bias_offset + sglid];
            bias_offset += N;
        }
    }

    uint output_offset = out_b * N + out_f;
    unroll_for(uint num = 0; num < BLOCK_NUM; num++) {
        unroll_for(uint bi = 0; bi < BLOCK_B; bi++) {
            intel_sub_group_block_write((__global uint *)(output + output_offset), result[num][bi]);
            output_offset += N;
        }
    }
}

/****************************************************************************************************************************
XMX version 5:
     1. Input data shared in subgroup
     2. Weights data in SLM, shared in one workgroup with 2 subgroups 
     3. WI: 1x8 MADs each iter, subgroup: 1x8xSIMD=64 MADs each iter
     4. M=1024, N=1024, K=2560, 0.221 ms ms
****************************************************************************************************************************/
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void matmul_bf_xmx_5(__global char* input,
                             __global const char* weights,
                             __global char* biases,
                             __global int* output,
                             int M, int N, int K) {
    uint gid = (uint)get_global_id(0);
    uint sglid = (uint)get_sub_group_local_id();
    uint SIMD = (uint)get_sub_group_size();
    uint subgroup_id = get_sub_group_id();

    #define SUBGROUP_TILE_B 2
    uint sg_id = gid/SIMD;
    uint sg_block_f = (sg_id / SUBGROUP_TILE_B) % (N / SIMD);
    uint sg_block_b = (sg_id / (N / SIMD) / SUBGROUP_TILE_B ) * SUBGROUP_TILE_B + sg_id % SUBGROUP_TILE_B;
    uint out_f = sg_block_f * BLOCK_F;
    uint out_b = sg_block_b * BLOCK_B;
    int8 result = {},c = {};
    __local int8 wei_slm[BLOCK_F];

    unroll_for(uint ki = 0; ki < K; ki += BLOCK_K) {
        uint input_offset = out_b * K + ki;
        int8 a,b;
        unroll_for(uint bi = 0;bi < BLOCK_B; bi++) {
            a[bi] = as_int(intel_sub_group_block_read((const __global uint*)(input + input_offset)));
            input_offset += K;
        }
        if(subgroup_id == 0)
            wei_slm[sglid] = *(const __global int8*)(weights + out_f * K + sglid * K + ki );

        barrier(CLK_LOCAL_MEM_FENCE);
        // b=*(const __global int8*)(weights + out_f * K + sglid * K + ki );
        b = wei_slm[sglid];
        result += intel_sub_group_i8_i8_matrix_mad_k32(a,b,c);
    }

    uint bias_offset = out_b * N + out_f;
    unroll_for(uint bi = 0; bi < BLOCK_B; bi++) {
        result[bi] += biases[bias_offset + sglid];
        bias_offset += N;
    }

    uint output_offset = out_b * N + out_f;
    unroll_for(uint bi = 0; bi < BLOCK_B; bi++) {
        intel_sub_group_block_write((__global uint *)(output + output_offset), result[bi]);
        output_offset += N;
    }
}
/****************************************************************************************************************************
XMX version 6:
     1. Input data shared in subgroup
     2. 2x subgroup pair to do one XMX - DPASW
     3. WI: 1x8 MADs, subgroup: 4x8=32 MADs
     4. M=1024, N=1024, K=2560, 0.295393 ms
****************************************************************************************************************************/
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void matmul_bf_xmx_6(__global char* input,
                             __global const char* weights,
                             __global char* biases,
                             __global int* output,
                             int M, int N, int K) {
    uint gid = (uint)get_global_id(0);
    uint sglid = (uint)get_sub_group_local_id();
    uint SIMD = (uint)get_sub_group_size();
    uint sg_id = gid/SIMD;
    uint sg_block_f = (sg_id / 2) % (N / SIMD);
    uint sg_block_b = (sg_id / (N / SIMD) / 2 ) * 2 + sg_id % 2;
    uint out_f = sg_block_f * BLOCK_F;
    uint out_b = sg_block_b * BLOCK_B / 2;
    int8 result = {}, c = {};

    unroll_for(uint ki = 0; ki < K; ki += BLOCK_K) {
        uint input_offset = out_b * K + ki;
        int4 a;
        int8 b;
        unroll_for(uint bi = 0;bi < BLOCK_B / 2; bi++) {
            a[bi] = as_int(intel_sub_group_block_read((const __global uint*)(input + input_offset)));
            input_offset += K;
        }

        uint weights_offset = out_f * K + sglid * K + ki;
        unroll_for(uint fi = 0; fi < BLOCK_F; fi++) {
            b[fi]=*(const __global int*)(weights + weights_offset + fi * 4 );
        }
        result += intel_sub_group_i8_i8_split_matrix_mad_k32(a,b,c);
    }

    uint bias_offset = out_b * N + out_f;
    unroll_for(uint bi = 0; bi < BLOCK_B; bi++) {
        result[bi] += biases[bias_offset + sglid];
        bias_offset += N;
    }

    uint output_offset = out_b * N + out_f;
    unroll_for(uint bi = 0; bi < BLOCK_B; bi++) {
        intel_sub_group_block_write((__global uint *)(output + output_offset), result[bi]);
        output_offset += N;
    }
}

/****************************************************************************************************************************
XMX version peak:
     1. Do nothing except XMX + accumulator
     2. Only used for peak performance reference
     3. WI: 1x8x(K/32) MADs, subgroup: 1x8xSIMDxK/32=2K MADs
     4. M=1024, N=1024, K=2560, 0.0359509 ms
****************************************************************************************************************************/
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void matmul_bf_xmx_peak(__global char* input,
                             __global const char* weights,
                             __global char* biases,
                             __global int* output,
                             int M, int N, int K) {
    uint gid = (uint)get_global_id(0);
    uint SIMD = (uint)get_sub_group_size();
    uint sg_id = gid/SIMD;
    uint out_f = (sg_id % (N / SIMD)) * BLOCK_F;
    uint out_b = (sg_id / (N / SIMD)) * BLOCK_B;
    int8 result;
    
    int8 a = as_int8(intel_sub_group_block_read8((__global uint*) input));
    int8 b = as_int8(intel_sub_group_block_read8((__global uint*) weights));
    int8 c= as_int8(intel_sub_group_block_read8((__global uint*) biases));

#if 0
    // 0.049689 ms
    unroll_for(uint it = 0; it < K; it += BLOCK_K) {
        result = intel_sub_group_i8_i8_matrix_mad_k32(a,b,result);
    }
#else
    #define ACC_CNT 10
    int8 temp[ACC_CNT];
    __attribute__((opencl_unroll_hint(1)))
    for(uint it = 0; it < K/BLOCK_K; it += ACC_CNT) {
        __attribute__((opencl_unroll_hint(ACC_CNT)))
        for(uint i = 0; i < ACC_CNT; i++){
            temp[i] = intel_sub_group_i8_i8_matrix_mad_k32(a,b,temp[i]);
        }
        unroll_for(uint i = 0; i < ACC_CNT; i++){
            result += temp[i];
        }
    }
#endif
    intel_sub_group_block_write8((__global uint*)output, result[0]);
}

__attribute__((intel_reqd_sub_group_size(8)))
__kernel void matmul_bf_xmx_peak_fake(__global char* input,
                             __global const char* weights,
                             __global char* biases,
                             __global int* output,
                             int M, int N, int K) {
    uint gid = (uint)get_global_id(0);
    uint SIMD = (uint)get_sub_group_size();
    uint sg_id = gid/SIMD;
    uint out_f = (sg_id % (N / SIMD)) * BLOCK_F;
    uint out_b = (sg_id / (N / SIMD)) * BLOCK_B;
    int8 result;
    
    int8 a = 0, b = 0, c= 0;

#if 0
    // 0.049689 ms
    unroll_for(uint it = 0; it < K; it += BLOCK_K) {
        result = intel_sub_group_i8_i8_matrix_mad_k32(a,b,result);
    }
#else
    #define ACC_CNT 10
    int8 temp[ACC_CNT];
    __attribute__((opencl_unroll_hint(1)))
    for(uint it = 0; it < K/BLOCK_K; it += ACC_CNT) {
        __attribute__((opencl_unroll_hint(ACC_CNT)))
        for(uint i = 0; i < ACC_CNT; i++){
            temp[i] = intel_sub_group_i8_i8_matrix_mad_k32(a,b,temp[i]);
        }
        unroll_for(uint i = 0; i < ACC_CNT; i++){
            result += temp[i];
        }
    }
#endif
    intel_sub_group_block_write8((__global uint*)output, result[0]);
}

'''

from . import cl
import numpy as np
from .utils import *

TILE_B=8
BLOCK_NUM=4
SUBGROUP_TILE_B=2
#K_groups = 32
#cl_kernels = kernel_cache(cl_kernel_sources, f"-D K_groups={K_groups}")
cl_kernels = kernel_cache(cl_kernel_sources)

class Linear_XMX_I8:
    def __init__(self, weight, bias, M, N, K):
        self.N = N
        self.M = M
        self.K = K

        assert(self.K % 32 == 0)
        assert(self.M % 8 == 0)
        assert(self.N % 8 == 0)
        
        self.weight = to_cl(weight)
        self.bias = to_cl(bias)
        
        self.kernel_list = [["matmul_bf_tile",  [self.M*self.N//TILE_B, 1, 1],[8,1,1]],
                            ["matmul_bf_xmx_1", [self.M*self.N//TILE_B, 1, 1],[8,1,1]],
                            ["matmul_bf_xmx_2", [self.M*self.N//TILE_B, 1, 1],[8,1,1]],
                            ["matmul_bf_xmx_3", [self.M*self.N//TILE_B, 1, 1],[8,1,1]],
                            ["matmul_bf_xmx_4", [self.M*self.N//TILE_B//BLOCK_NUM, 1, 1],[8,1,1]],
                            ["matmul_bf_xmx_5", [self.M*self.N//TILE_B, 1, 1],[8*SUBGROUP_TILE_B,1,1]],
                            ["matmul_bf_xmx_6", [self.M*self.N*2//TILE_B, 1, 1],[16,1,1]],
                            ["matmul_bf_xmx_peak", [self.M*self.N//TILE_B, 1, 1],[16*8,1,1]],
                            ["matmul_bf_xmx_peak_fake", [self.M*self.N//TILE_B, 1, 1],[16*8,1,1]]]

    def __call__(self, input, output, kernel_id = 4):
        # # shape inference
        # i_shape = input.shape
        # o_shape = list(i_shape)
        # o_shape[-1] = self.N
        # output = cl.tensor(o_shape, input.dtype)
        self.kernel_id = kernel_id

        cl_kernels.enqueue(self.kernel_list[kernel_id][0],
                           self.kernel_list[kernel_id][1],
                           self.kernel_list[kernel_id][2], 
                           input,
                           self.weight,
                           self.bias,
                           output,
                           self.M,
                           self.N,
                           self.K)
        return output
    
    def get_kernel_name(self):
        return self.kernel_list[self.kernel_id][0]


if __name__ == "__main__":
    import sys
    cl.profiling(True)
    def test_matmul(shape, Bcnt = 0, kernel_id = 7):
        M, N, K = shape
        A = torch.randint(-2, 2, [M, K], dtype=torch.int8)
        B = torch.randint(-2, 2, [N, K], dtype=torch.int8)
        C = torch.randint(-2, 2, [M, N], dtype=torch.int8)
        ref = torch.matmul(A.int(), B.int().transpose(1,0))
        ref = torch.add(ref, C).int().numpy()
        Bsize = K*N*1 + M*K*1 + M*N*1
        if (Bcnt <= 0): Bcnt = int(500e6)//(Bsize)
        linears = [Linear_XMX_I8(B,C,M,N,K) for _ in range(Bcnt)]

        input = to_cl(A)
        D = torch.randint(-1, 1, [M, N], dtype=torch.int)
        output = to_cl(D)
        for l in linears:
            output = l(input, output, kernel_id)

        durs = cl.finish()
        gops = shape[0] * shape[1] * shape[2] * 2  + shape[0] * shape[1]
        for ns in durs:
            print(f"{linears[0].get_kernel_name()} : {shape} {Bsize*1e-6:.3f} MB {ns*1e-6:.3f} ms, BW: { Bsize/ns : .2f} GB/s,  Compute: {gops/ns/1000 : .2f} TOPS/s")
        if (len(durs)):
            mean_ns = sum(durs)/len(durs)
            print(f"[total]  {mean_ns*1e-6: .3f} ms, BW: { Bsize/mean_ns : .2f} GB/s, Compute: {gops/mean_ns/1000 : .2f} TOPS/s")

        res = output.numpy()
        if kernel_id < 7:
            compare(ref, res, 0, 0)

    test_matmul([1024, 1024, 2560], 20, 7)
    sys.exit(0)

    for id in range(8):
        test_matmul([1024, 1024, 2560], 20, id)

    


