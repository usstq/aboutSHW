cl_kernel_sources = r'''ulong __attribute__((overloadable)) intel_get_cycle_counter( void );
uint __attribute__((overloadable)) intel_get_slice_id( void );
uint __attribute__((overloadable)) intel_get_subslice_id( void );
uint __attribute__((overloadable)) intel_get_dual_subslice_id( void );
uint __attribute__((overloadable)) intel_get_eu_id( void );
uint __attribute__((overloadable)) intel_get_eu_thread_id( void );

#define unroll_for __attribute__((opencl_unroll_hint)) for

#if DEBUG
void print_as_half16(int8 result, uint offset) {
     uint sglid = (uint)get_sub_group_local_id();
     uint sg_id = (uint)get_sub_group_id();
     int sg_c = get_local_id(0);
     int sg_n = get_local_id(1);
     int sg_m = get_local_id(2);

     half16 temp = as_half16(result);
     printf("half16[%2d-%2d] =[%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f], offset = %d\n",sg_id,sglid,
             temp[0], temp[1],temp[2],temp[3],temp[4], temp[5],temp[6],temp[7],
             temp[8], temp[9],temp[10],temp[11],temp[12], temp[13],temp[14],temp[15], offset);
}

void print_as_half8(half8 temp, uint offset) {
     uint sglid = (uint)get_sub_group_local_id();
     uint sg_id = (uint)get_sub_group_id();
     int sg_c = get_local_id(0);
     int sg_n = get_local_id(1);
     int sg_m = get_local_id(2);

     printf("half8[%2d-%2d] = [%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f], offset = %d\n",sg_id,sglid,
             temp[0], temp[1],temp[2],temp[3],temp[4], temp[5],temp[6],temp[7], offset);
}

void print_as_half4(half4 temp, uint offset) {
     uint sglid = (uint)get_sub_group_local_id();
     uint sg_id = (uint)get_sub_group_id();
     int sg_c = get_local_id(0);
     int sg_n = get_local_id(1);
     int sg_m = get_local_id(2);

     printf("half8[%2d-%2d] = [%.3f,%.3f,%.3f,%.3f], offset = %d\n",sg_id,sglid,
             temp[0], temp[1],temp[2],temp[3], offset);
}

void print_as_half16_local(__local uint* ptr, int offset) {
     uint sglid = (uint)get_sub_group_local_id();
     uint sg_id = (uint)get_sub_group_id();
     int sg_c = get_local_id(0);
     int sg_n = get_local_id(1);
     int sg_m = get_local_id(2);

     half16 temp = as_half16(*(__local uint8 *)(ptr));
     printf("local_half16[%2d-%2d] = [%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f], offset = %d\n",
             sg_id,sglid,
             temp[0], temp[1],temp[2],temp[3],temp[4], temp[5],temp[6],temp[7],
             temp[8], temp[9],temp[10],temp[11],temp[12], temp[13],temp[14],temp[15], offset);
}

void print_as_half8_local(__local uint* ptr, int offset) {
     uint sglid = (uint)get_sub_group_local_id();
     uint sg_id = (uint)get_sub_group_id();
     int sg_c = get_local_id(0);
     int sg_n = get_local_id(1);
     int sg_m = get_local_id(2);

     half8 temp = as_half8(*(__local uint4 *)(ptr));
     printf("local_half16[%2d-%2d] = [%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f], offset = %d\n",
             sg_id,sglid,
             temp[0], temp[1],temp[2],temp[3],temp[4], temp[5],temp[6],temp[7], offset);
}

void print_wi_info() {
    printf("global_id = (%d,%d,%d), local_id = (%d,%d,%d), group_id = (%d,%d,%d), subgroup_id=%d, subgroup_size=%d, subgroup_local_id=%d\n",
                get_global_id(0), get_global_id(1),get_global_id(2),
                get_local_id(0), get_local_id(1),get_local_id(2),
                get_group_id(0), get_group_id(1),get_group_id(2),
                get_sub_group_id(),get_sub_group_size(),
                get_sub_group_local_id());
}
#endif
/****************************************************************************************************************************
MLP reference version:
     1.  gws = [M,N,1]
     2.  lws = [8,1,1]
****************************************************************************************************************************/
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void mlp_ref(__global const half* input,
                       __global const half* weight_gate,
                       __global const half* weight_up,
                       __global const half* bias,
                       __global half* out,
                       int M, int K, int N) {
    int m = get_global_id(0);
    int n = get_global_id(1);
    
    half sum_gate = (half)0.0, sum_up = (half)0.0;
    unroll_for(int ki = 0; ki < K; ki++) {
        sum_gate += input[m*K+ki] * weight_gate[n*K+ki];
        sum_up   += input[m*K+ki] * weight_up[n*K+ki];
    }
    half sum = sum_up * sum_gate / (1.0f + native_exp(-sum_gate));
    // half sum = sum_gate;
    *(out + m*N + n) = sum;
    //printf("out[%d,%d] = %.4f\n",m,n,sum);
}


/****************************************************************************************************************************
MLP base version: dpas - 4x1 xmx  
    gws=[SG_SIZE, N//SG_OUTPUT_N, M//SG_OUTPUT_M] = [8, N//(1*8), M//(4*8)]
    lws=[SG_SIZE, WG_SG_N, WG_SG_M] = [8, 8, 2]
    workgroup_num = [1, (N//SG_OUTPUT_N)/WG_SG_N, M//SG_OUTPUT_M/WG_SG_M
                = [1, N//(1*8)//8, M//(4*8)//2] = N//64 * M//64
    subgroup_num_in_workgroup
                = [WG_SG_N, WG_SG_M]
                = [8, 2] = 16
****************************************************************************************************************************/
/*
#if 0
#define SIMD 8  // fixed value
// Each SG has 4*1 XMX for gate and 4*1 XMX for up
#define SG_XMX_M 4
#define SG_XMX_N 1
// subgroup number in each workgroup
// max: XVE count in each XeCore
#define WG_SG_N 8  
// max: threads number in each XVE
#define WG_SG_M 2
#endif
*/

#define SG_SIZE SIMD
#define XMX_OUT_BLOCK_M 8   // Can be: 1,2,4,8
#define XMX_OUT_BLOCK_N 8   // fixed value
#define XMX_IN_BLOCK_M XMX_OUT_BLOCK_M
#define XMX_IN_BLOCK_K 16    // fixed: half:16, char:32
#define XMX_WEI_BLOCK_K XMX_IN_BLOCK_K
#define XMX_WEI_BLOCK_N 8    // fixed value

#define XMX_OUT_BLOCK_SIZE (XMX_OUT_BLOCK_M*XMX_OUT_BLOCK_N) // 8*8=64
#define XMX_INPUT_BLOCK_SIZE (XMX_IN_BLOCK_M*XMX_IN_BLOCK_K) // 8*16=128

// output sub-matrix shape in each subgroup
#define SG_OUTPUT_M (SG_XMX_M*XMX_OUT_BLOCK_M)  // 4*8
#define SG_OUTPUT_N (SG_XMX_N*XMX_OUT_BLOCK_N)  // 1*8 for each gate and up output

// output sub-matrix size in each workgroup
#define WG_OUTPUT_M  (SG_OUTPUT_M * WG_SG_M)
#define WG_OUTPUT_N  (SG_OUTPUT_N * WG_SG_N)

// step of K in each iteration, it is limited by SLM 
#define BK 64
#define dK XMX_IN_BLOCK_K      // 16: contains 16 half/32 bytes
#define INPUT_BLOCK_K  (BK/XMX_IN_BLOCK_K)    // 64/16=4
#define INPUT_BLOCK_M  (SG_XMX_M * WG_SG_M)    // 4*2=8 block

//4
#define INPUT_K_N  (BK/dK)
//2
#define INPUT_K_M  (WG_SG_N/INPUT_K_N)

// 4*8*64/8=256
#define SG_COPY_INPUT_BLOCK_SIZE  (SG_OUTPUT_M * BK/WG_SG_N)
// 4*64/8/16=2    256/8/16=2
#define ROW_PER_CHANNEL (SG_COPY_INPUT_BLOCK_SIZE/SIMD/XMX_IN_BLOCK_K)

// SG_OUTPUT_N * WG_SG_N* BK/(WG_SG_N * WG_SG_M) =  SG_OUTPUT_N * (BK/WG_SG_M)
// (1*8)*16=1*8*16 half = 1*8 regs
// #define SG_COPY_WEIGHT_BLOCK_SIZE  (SG_OUTPUT_N * dK)
#define SG_COPY_WEIGHT_BLOCK_SIZE  (SG_OUTPUT_N * BK / WG_SG_M )
// 1*8*8*64/(8*2)/(8*16)=2
#define BLOCK_K_PER_SG  ((WG_OUTPUT_N * BK)/(WG_SG_N*WG_SG_M)/SG_COPY_WEIGHT_BLOCK_SIZE)

#define INPUT_COPY_BLOCK_SG_M_PER_SG_ROW (WG_SG_N/INPUT_BLOCK_K)  // 8/4=2
//#define INPUT_COPY_BLOCK_M_PER_SG  (INPUT_COPY_BLOCK_SG_M_PER_SG_ROW * XMX_IN_BLOCK_M)  // 2*8=16
#define INPUT_COPY_BLOCK_M_PER_SG  (SG_COPY_INPUT_BLOCK_SIZE/dK)  // 128/16=8
        
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void mlp_base_4x1(__global const half* input,
                       __global const half* weight_gate,
                       __global const half* weight_up,
                       __global const half* bias,
                       __global half* out,
                       int M, int K, int N) {
                           
    __local half local_input[WG_OUTPUT_M * BK]; // (4*8)*2*64 = 4 KB
    __local half local_gate[WG_OUTPUT_N * BK];  // (1*8)*8*64 = 4 KB
    __local half local_up[WG_OUTPUT_N * BK];    // (1*8)*8*64 = 4 KB
    
    int wg_n = get_group_id(1);
    int wg_m = get_group_id(2);
    int sg_id = get_sub_group_id();
    
    //print_wi_info();
    int sg_c = get_local_id(0);
    int sg_n = get_local_id(1);
    int sg_m = get_local_id(2);

    float8 gate_output[4] = {0.0f};
    float8 up_output[4] = {0.0f};
    
    unroll_for(uint ki = 0; ki < K; ki += BK) {
        // Copy input data into local memory
        // Each subgroup copy (SG_OUTPUT_M * BK)/WG_SG_N = (4*8)*64/8 = 2*8*16 half
        // Each channel copy 16*16/SIMD=16*16/8 = 2*16 half data, 2 rows
        // (4*8)*64/8=4*64= 2*8*16 half = 2*8 regs
        {
            uint local_input_offset = sg_id * SG_COPY_INPUT_BLOCK_SIZE + sg_c * ROW_PER_CHANNEL * dK;
            uint input_offset = wg_m * WG_OUTPUT_M * K + (sg_m % WG_SG_M) * SG_OUTPUT_M * K + (sg_n%INPUT_COPY_BLOCK_SG_M_PER_SG_ROW)*INPUT_COPY_BLOCK_M_PER_SG*K + (sg_n / INPUT_COPY_BLOCK_SG_M_PER_SG_ROW) * dK +  sg_c * ROW_PER_CHANNEL * K + ki;
            unroll_for(uint i = 0; i < ROW_PER_CHANNEL; i++) {
                #if 0
                int8 row_data = *(__global int8 *)(input + input_offset + i * K);
                *(int8 *)(local_input + local_input_offset + i * dK) = row_data;
                #else
                *(int8 *)(local_input + local_input_offset + i * dK) = *(__global int8 *)(input + input_offset + i * K);
                #endif
            }       
        }

        // Copy weight data into local memory
        // Each subgroup copy (WG_OUTPUT_N * BK)/(WG_SG_N*WG_SG_M) = (8*8)*64/(8*2) = 2*8*16 half
        // Each channel copy 16*16/SIMD=16*16/8 = 2*16 half data, 2 cols
        {
            uint gate_offset =  (wg_n * WG_OUTPUT_N + sg_n * SG_OUTPUT_N + sg_c) * K +  ki + sg_m * BLOCK_K_PER_SG * dK;
            uint local_gate_offset = (sg_m * BLOCK_K_PER_SG * WG_SG_N + sg_n) * SG_COPY_WEIGHT_BLOCK_SIZE; // block_write
            unroll_for(uint i = 0; i < BLOCK_K_PER_SG; i++) {
                int8 gate_cols_data = *(__global int8 *)(weight_gate + gate_offset + dK * i);
                int8 up_cols_data   = *(__global int8 *)(weight_up   + gate_offset + dK * i);
                intel_sub_group_block_write8((__local uint*)(local_gate + local_gate_offset + SG_COPY_WEIGHT_BLOCK_SIZE * WG_SG_N * i), as_uint8(gate_cols_data));
                intel_sub_group_block_write8((__local uint*)(local_up   + local_gate_offset + SG_COPY_WEIGHT_BLOCK_SIZE * WG_SG_N * i), as_uint8(up_cols_data));
                // print_as_half16(gate_cols_data, gate_offset + dK * i);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        //int iii = 8;
        //print_as_half16_local((__local uint *)(local_gate + SG_COPY_WEIGHT_BLOCK_SIZE * iii), SG_COPY_WEIGHT_BLOCK_SIZE * iii);

        const __local half* local_input_ptr = (const __local half *)(local_input + sg_m * WG_SG_N * SG_COPY_INPUT_BLOCK_SIZE);
        const __local half* local_gate_ptr  = (const __local half *)(local_gate  + sg_n * SG_COPY_WEIGHT_BLOCK_SIZE);
        const __local half* local_up_ptr    = (const __local half *)(local_up    + sg_n * SG_COPY_WEIGHT_BLOCK_SIZE);
        __attribute__((opencl_unroll_hint(1)))
        for(int kii = 0; kii < BK/dK; kii ++) {
            int8 in0 = as_int8(intel_sub_group_block_read8((__local uint*)(local_input_ptr + kii * 2 * SG_COPY_INPUT_BLOCK_SIZE)));
            int8 in1 = as_int8(intel_sub_group_block_read8((__local uint*)(local_input_ptr + kii * 2 * SG_COPY_INPUT_BLOCK_SIZE + 1 * SG_COPY_INPUT_BLOCK_SIZE/2)));
            int8 in2 = as_int8(intel_sub_group_block_read8((__local uint*)(local_input_ptr + kii * 2 * SG_COPY_INPUT_BLOCK_SIZE + 2 * SG_COPY_INPUT_BLOCK_SIZE/2)));
            int8 in3 = as_int8(intel_sub_group_block_read8((__local uint*)(local_input_ptr + kii * 2 * SG_COPY_INPUT_BLOCK_SIZE + 3 * SG_COPY_INPUT_BLOCK_SIZE/2)));
            
            // print_as_half16(in0, kii);
            
            int8 gate = as_int8(intel_sub_group_block_read8((__local uint*)(local_gate_ptr + kii * WG_SG_N * SG_COPY_WEIGHT_BLOCK_SIZE)));
            int8 up   = as_int8(intel_sub_group_block_read8((__local uint*)(local_up_ptr   + kii * WG_SG_N * SG_COPY_WEIGHT_BLOCK_SIZE)));
            
            gate_output[0] = intel_sub_group_f16_f16_matrix_mad_k16(in0, gate, gate_output[0]);
            gate_output[1] = intel_sub_group_f16_f16_matrix_mad_k16(in1, gate, gate_output[1]);
            gate_output[2] = intel_sub_group_f16_f16_matrix_mad_k16(in2, gate, gate_output[2]);
            gate_output[3] = intel_sub_group_f16_f16_matrix_mad_k16(in3, gate, gate_output[3]);
            
            up_output[0] = intel_sub_group_f16_f16_matrix_mad_k16(in0, up, up_output[0]);
            up_output[1] = intel_sub_group_f16_f16_matrix_mad_k16(in1, up, up_output[1]);
            up_output[2] = intel_sub_group_f16_f16_matrix_mad_k16(in2, up, up_output[2]);
            up_output[3] = intel_sub_group_f16_f16_matrix_mad_k16(in3, up, up_output[3]);
        }
        
        // need to sync again at the end, to avoid faster threads
        // fetching the next block into the cache before slower threads are done
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    __local half * scratch = local_input + get_sub_group_id() * XMX_OUT_BLOCK_SIZE * SG_XMX_M;
    unroll_for(int i = 0; i < SG_XMX_M; i ++) {
        half8 output = convert_half8(gate_output[i] / (1.0 + native_exp(-gate_output[i])) * up_output[i]);
        intel_sub_group_block_write_us8(scratch + i*XMX_OUT_BLOCK_SIZE, as_ushort8(output));
    }
    
    __global half* dst_ptr = out + (wg_m * WG_OUTPUT_M + sg_m * SG_OUTPUT_M + sg_c) * N + (wg_n * WG_OUTPUT_N + sg_n * SG_OUTPUT_N);
    // int channel = get_sub_group_local_id();
    __local half * src = (__local half *)(scratch + sg_c * 8);
    unroll_for(int i = 0; i < SG_XMX_M; i ++) {
        half8 value = *(__local half8 *)(src);
        *(__global half8 *)dst_ptr = value;
        dst_ptr += N * 8;
        src += XMX_OUT_BLOCK_SIZE;
    }
}

/****************************************************************************************************************************
MLP: XMX dpas - mxn
    SG_OUTPUT_N = 2*8 = 16
    WG_SG_N = 8
   
    define SG_XMX_M 4
    define SG_XMX_N 2

    gws=[SG_SIZE, N//SG_OUTPUT_N, M//SG_OUTPUT_M] = [8, N//(2*8), M//(4*8)]
    lws=[SG_SIZE, WG_SG_N, WG_SG_M] = [8, 8, 2]

    workgroup_num = [1, (N//SG_OUTPUT_N)/WG_SG_N, M//SG_OUTPUT_M/WG_SG_M
                = [1, N//(2*8)//8, M//(4*8)//2] = N//128 * M//64
    subgroup_num_in_workgroup
                = [WG_SG_N, WG_SG_M]
                = [8, 2] = 16
****************************************************************************************************************************/
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void mlp_base_mxn(__global const half* input,
                       __global const half* weight_gate,
                       __global const half* weight_up,
                       __global const half* bias,
                       __global half* out,
                       int M, int K, int N) {
                           
    __local half local_input[WG_OUTPUT_M * BK]; // (4*8)*2*64 = 4 KB
    __local half local_gate[WG_OUTPUT_N * BK];  // (2*8)*8*64 = 8 KB
    __local half local_up[WG_OUTPUT_N * BK];    // (2*8)*8*64 = 8 KB
    
    int wg_n = get_group_id(1);
    int wg_m = get_group_id(2);
    int sg_id = get_sub_group_id();
    
    //print_wi_info();
    int sg_c = get_local_id(0); // sub-group channel id
    int sg_n = get_local_id(1); // sub-group id in 'N' dim
    int sg_m = get_local_id(2); // sub-group id in 'M' dim
        
    float8 gate_output[SG_XMX_M][SG_XMX_N] = {0.0f};
    float8 up_output[SG_XMX_M][SG_XMX_N] = {0.0f};
    
    unroll_for(uint ki = 0; ki < K; ki += BK) {
        // Copy input data into local memory
        // Each subgroup copy (SG_OUTPUT_M * BK)/WG_SG_N = (4*8)*64/8 = 2*8*16 half
        // Each channel copy 16*16/SIMD=16*16/8 = 2*16 half data, 2 rows
        // (4*8)*64/8=4*64= 2*8*16 half = 2*8 regs
        {
            uint local_input_offset = sg_id * SG_COPY_INPUT_BLOCK_SIZE + sg_c * ROW_PER_CHANNEL * dK;
            uint input_offset = wg_m * WG_OUTPUT_M * K + (sg_m % WG_SG_M) * SG_OUTPUT_M * K + (sg_n%INPUT_COPY_BLOCK_SG_M_PER_SG_ROW)*INPUT_COPY_BLOCK_M_PER_SG*K + (sg_n / INPUT_COPY_BLOCK_SG_M_PER_SG_ROW) * dK +  sg_c * ROW_PER_CHANNEL * K + ki;
            unroll_for(uint i = 0; i < ROW_PER_CHANNEL; i++) {
                #if 0
                int8 row_data = *(__global int8 *)(input + input_offset + i * K);
                *(int8 *)(local_input + local_input_offset + i * dK) = row_data;
                #else
                *(int8 *)(local_input + local_input_offset + i * dK) = *(__global int8 *)(input + input_offset + i * K);
                #endif
            }
        }

        // Copy weight data into local memory
        // Each subgroup copy (WG_OUTPUT_N * BK)/(WG_SG_N*WG_SG_M) = (8*8)*64/(8*2) = 2*8*16 half
        // Each channel copy 16*16/SIMD=16*16/8 = 2*16 half data, 2 cols
        // BLOCK_K_PER_SG=2*8*8*64/(8*2)/(16*2*8)=2
        {
            uint gate_offset =  (wg_n * WG_OUTPUT_N + sg_n * SG_OUTPUT_N + sg_c) * K +  ki + sg_m * BLOCK_K_PER_SG * dK;
            uint local_gate_offset = (sg_m * BLOCK_K_PER_SG * WG_SG_N + sg_n) * SG_COPY_WEIGHT_BLOCK_SIZE; // block_write
            unroll_for(uint i = 0; i < BLOCK_K_PER_SG; i++) {
                unroll_for(uint n = 0; n < SG_XMX_N; n++) {
                    int8 gate_cols_data = *(__global int8 *)(weight_gate + gate_offset + dK * i + XMX_OUT_BLOCK_N * n * K);
                    int8 up_cols_data   = *(__global int8 *)(weight_up   + gate_offset + dK * i + XMX_OUT_BLOCK_N * n * K);
                    intel_sub_group_block_write8((__local uint*)(local_gate + local_gate_offset + SG_COPY_WEIGHT_BLOCK_SIZE * WG_SG_N * i + dK * XMX_OUT_BLOCK_N * n), as_uint8(gate_cols_data));
                    intel_sub_group_block_write8((__local uint*)(local_up   + local_gate_offset + SG_COPY_WEIGHT_BLOCK_SIZE * WG_SG_N * i + dK * XMX_OUT_BLOCK_N * n), as_uint8(up_cols_data));
                    if(sg_n==0 && sg_m==0 && sg_c==0 && wg_n==1) {
                        //print_as_half16(as_int8(gate_cols_data), wg_n);
                        //print_as_half16(as_int8(up_cols_data), local_gate_offset);
                    }
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        const __local half* local_input_ptr = (const __local half *)(local_input + sg_m * WG_SG_N * SG_COPY_INPUT_BLOCK_SIZE);
        const __local half* local_gate_ptr  = (const __local half *)(local_gate  + sg_n * SG_COPY_WEIGHT_BLOCK_SIZE);
        const __local half* local_up_ptr    = (const __local half *)(local_up    + sg_n * SG_COPY_WEIGHT_BLOCK_SIZE);
        __attribute__((opencl_unroll_hint(1)))
        for(int kii = 0; kii < BK/dK; kii ++) {
            int8 in_vec[SG_XMX_M];
            int8 gate_vec[SG_XMX_M], up_vec[SG_XMX_M];
            
            unroll_for(int m = 0; m < SG_XMX_M; m++) {
                in_vec[m] = as_int8(intel_sub_group_block_read8((__local uint*)(local_input_ptr + kii * 2 * SG_COPY_INPUT_BLOCK_SIZE + m * XMX_INPUT_BLOCK_SIZE)));
            }
            // print_as_half16(in_vec[0], kii);
    
            unroll_for(int n = 0; n < SG_XMX_N; n++) {
                gate_vec[n] = as_int8(intel_sub_group_block_read8((__local uint*)(local_gate_ptr + kii * WG_SG_N * SG_COPY_WEIGHT_BLOCK_SIZE + n * dK * XMX_OUT_BLOCK_N)));
                up_vec[n]   = as_int8(intel_sub_group_block_read8((__local uint*)(local_up_ptr   + kii * WG_SG_N * SG_COPY_WEIGHT_BLOCK_SIZE + n * dK * XMX_OUT_BLOCK_N)));
            }
            
            unroll_for(uint m = 0; m < SG_XMX_M; m++) {
                unroll_for(uint n =0; n < SG_XMX_N; n++) {
                    gate_output[m][n] = intel_sub_group_f16_f16_matrix_mad_k16(in_vec[m], gate_vec[n], gate_output[m][n]);
                    up_output[m][n] = intel_sub_group_f16_f16_matrix_mad_k16(in_vec[m], up_vec[n], up_output[m][n]);
                }
            }
        }
        // need to sync again at the end, to avoid faster threads
        // fetching the next block into the cache before slower threads are done
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    __local half * scratch = local_input + get_sub_group_id() * XMX_OUT_BLOCK_SIZE * SG_XMX_M * SG_XMX_N;
    unroll_for(int m = 0; m < SG_XMX_M; m++) {
        unroll_for(int n = 0; n < SG_XMX_N; n++) {
            half8 output = convert_half8(gate_output[m][n] / (1.0 + native_exp(-gate_output[m][n])) * up_output[m][n]);
            // half8 output = convert_half8(up_output[m][n]);
            // print_as_half8(output,i);
            intel_sub_group_block_write_us8(scratch + (m*SG_XMX_N+n)*XMX_OUT_BLOCK_SIZE, as_ushort8(output));
        }
    }
    
    __global half* dst_ptr = out + (wg_m * WG_OUTPUT_M + sg_m * SG_OUTPUT_M + sg_c) * N + (wg_n * WG_OUTPUT_N + sg_n * SG_OUTPUT_N);
    // int channel = get_sub_group_local_id();
    __local half * src = (__local half *)(scratch + sg_c * XMX_OUT_BLOCK_N);
    unroll_for(int m = 0; m < SG_XMX_M; m++) {
        unroll_for(int n = 0; n < SG_XMX_N; n++) {
            half8 value = *(__local half8 *)(src);
             src += XMX_OUT_BLOCK_SIZE;
            *(__global half8 *)(dst_ptr + n * XMX_OUT_BLOCK_N) = value;
        }
        dst_ptr += N * 8;
    }
}

/****************************************************************************************************************************
MLP: XMX dpasw - 4x1 for each weights
    SG_OUTPUT_N = 1*8 = 8
    WG_SG_N = 8
   
    define SG_XMX_M 4   - XMX block number in M
    define SG_XMX_N 1   - XMX block number in N
    
    define WG_SG_M 4   - SG number in M
    define WG_SG_N 8   - SG number in N

    gws=[SG_SIZE, N//SG_OUTPUT_N, M//SG_OUTPUT_M] = [8, N//(2*8), M//(4*8)]
    lws=[SG_SIZE, WG_SG_N, WG_SG_M] = [8, 8, 4]

    workgroup_num = [1, (N//SG_OUTPUT_N)/WG_SG_N, M//SG_OUTPUT_M/WG_SG_M
                = [1, N//(2*8)//8, M//(4*8)//2] = N//128 * M//64
    subgroup_num_in_workgroup
                = [WG_SG_N, WG_SG_M]
                = [8, 4] = 16
****************************************************************************************************************************/
#ifdef XMX_OUT_BLOCK_M
#undef XMX_OUT_BLOCK_M
#undef XMX_IN_BLOCK_M
#undef XMX_OUT_BLOCK_SIZE
#undef SG_OUTPUT_M
#undef WG_OUTPUT_M
#undef SG_COPY_INPUT_BLOCK_SIZE

#define XMX_OUT_BLOCK_M 8
#define XMX_IN_BLOCK_M  4
#define XMX_OUT_BLOCK_SIZE (XMX_OUT_BLOCK_M*XMX_OUT_BLOCK_N) // 8*8=64
#define SG_OUTPUT_M (SG_XMX_M*XMX_OUT_BLOCK_M)  // 4*8=32
#define WG_OUTPUT_M  (SG_OUTPUT_M*WG_SG_M/2)   // 16*4=64

// Size for XMX compute in each SG
#define SG_INPUT_M  (XMX_IN_BLOCK_M*SG_XMX_M)  // 4*4=16
#define SG_INPUT_N  dK
#define SG_INPUT_BLOCK_SIZE  (SG_INPUT_M * SG_INPUT_N) //16*16=256

// Size for data copy in each SG
#define SG_COPY_INPUT_BLOCK_N  dK
#define SG_COPY_INPUT_BLOCK_SIZE  (SG_INPUT_M * BK/WG_SG_N) // 16*64/8=128
#define SG_COPY_INPUT_ROW_PER_CHANNEL (SG_COPY_INPUT_BLOCK_SIZE/SIMD/XMX_IN_BLOCK_K)  // 128/8/16=1
#endif


/****************************************************************************************************************************
MLP: XMX dpasw - 4x1 for each weights
    SG_OUTPUT_N = 1*8 = 8
    WG_SG_N = 8
   
    define SG_XMX_M 4   - XMX block number in M
    define SG_XMX_N 1   - XMX block number in N
    
    define WG_SG_M 4   - SG number in M
    define WG_SG_N 8   - SG number in N

    gws=[SG_SIZE, N//SG_OUTPUT_N, M//SG_OUTPUT_M] = [8, N//(2*8), M//(4*8)]
    lws=[SG_SIZE, WG_SG_N, WG_SG_M] = [8, 8, 4]

    workgroup_num = [1, (N//SG_OUTPUT_N)/WG_SG_N, M//SG_OUTPUT_M/WG_SG_M
                = [1, N//(2*8)//8, M//(4*8)//2] = N//128 * M//64
    subgroup_num_in_workgroup
                = [WG_SG_N, WG_SG_M]
                = [8, 4] = 16
****************************************************************************************************************************/
#ifdef XMX_OUT_BLOCK_M
#undef XMX_OUT_BLOCK_M
#undef XMX_IN_BLOCK_M
#undef XMX_OUT_BLOCK_SIZE
#undef SG_OUTPUT_M
#undef WG_OUTPUT_M
#undef SG_COPY_INPUT_BLOCK_SIZE
#undef SG_INPUT_N
#undef SG_COPY_INPUT_ROW_PER_CHANNEL
#undef SG_COPY_WEIGHT_BLOCK_SIZE
#undef SG_INPUT_M

#define XMX_OUT_BLOCK_M 8
#define XMX_OUT_BLOCK_SIZE 64 // 8*8=64

#define XMX_IN_BLOCK_M  4
#define XMX_IN_BLOCK_N  8

#define SG_OUTPUT_M (SG_XMX_M*XMX_OUT_BLOCK_M)  // 4*8=32
#define WG_OUTPUT_M  (SG_OUTPUT_M*WG_SG_M)   // 32*4=128

// Size for XMX compute in each SG
#define SG_INPUT_M  (4*SG_XMX_M)  // 4*4=16
#define SG_INPUT_N  dK
#define SG_INPUT_BLOCK_SIZE  (SG_INPUT_M * SG_INPUT_N) //16*16=256

// Size for data copy in each SG
#define SG_COPY_INPUT_BLOCK_N  dK
#define SG_COPY_INPUT_BLOCK_SIZE  (4*8*BK/WG_SG_N) // 32*64/8=256
#define SG_COPY_INPUT_ROW_PER_CHANNEL (SG_COPY_INPUT_BLOCK_SIZE/SIMD/dK)  // 256/8/16=2

#define SG_COPY_WEIGHT_BLOCK_SIZE  (8*BK/WG_SG_M)  // 8 * 64 / 4 = 128
#define SG_COPY_WEIGHT_COL_PER_CHANNEL (SG_COPY_WEIGHT_BLOCK_SIZE/dK/SIMD)  // 128/16/8=1

#endif

__attribute__((intel_reqd_sub_group_size(8)))
__kernel void mlp_dpasw_4x1(__global half* input,
                       __global half* weight_gate,
                       __global half* weight_up,
                       __global half* bias,
                       __global half* out,
                       int M, int K, int N) {
                           
    __local half local_input[WG_OUTPUT_M * BK]; // (4*16)*64*2 = 8 KB
    __local half local_gate[WG_OUTPUT_N * BK];  // (2*8)*8*64*2 = 16 KB
    __local half local_up[WG_OUTPUT_N * BK];    // (2*8)*8*64*2 = 16 KB
    
    int wg_n = get_group_id(1);
    int wg_m = get_group_id(2);
    int sg_id = get_sub_group_id();

    int sg_c = get_local_id(0); // sub-group channel id
    int sg_n = get_local_id(1); // sub-group id in 'N' dim
    int sg_m = get_local_id(2); // sub-group id in 'M' dim
    
    float8 gate_output[SG_XMX_M][SG_XMX_N] = {0.0f};
    float8 up_output[SG_XMX_M][SG_XMX_N] = {0.0f};

    // The beginning address of this WG
    input += wg_m * WG_OUTPUT_M * K;
    weight_gate += wg_n * WG_OUTPUT_N * K;
    weight_up += wg_n * WG_OUTPUT_N * K;

    // The input beginning address of this SG
    {
        uint block_rows = SG_COPY_INPUT_BLOCK_SIZE / dK;  // 256/16=16 rows
        uint block_num_m = (4*8)/ block_rows;  // 32/16=2
        uint offset_m = (sg_m * 32) + (sg_n % 2) * block_rows;
        uint offset_n = (sg_n / 2) * dK;
        input += offset_m * K + offset_n;
    }
    
    // The weight beginning address of this SG
    {
        uint block_size_n = SG_COPY_WEIGHT_BLOCK_SIZE / dK;   //8
        uint block_num_n = WG_OUTPUT_N / block_size_n;   //64/8=8
        uint offset_n = sg_n * block_size_n;
        uint offset_m = sg_m * dK;
        weight_gate += offset_n * K + offset_m;
        weight_up += offset_n * K + offset_m;
    }

    unroll_for(uint ki = 0; ki < K; ki += BK) {
        // Copy input data into local memory
        {
            uint local_input_offset = (sg_m * 8 + sg_n) * SG_COPY_INPUT_BLOCK_SIZE;
            unroll_for(uint i = 0; i < SG_COPY_INPUT_ROW_PER_CHANNEL; i++) {
                int8 row_data = *(__global int8 *)(input + ki + i * K + sg_c * SG_COPY_INPUT_ROW_PER_CHANNEL * K);
                *(int8 *)(local_input + local_input_offset + i * dK + sg_c * SG_COPY_INPUT_ROW_PER_CHANNEL * dK) = row_data;
            }
        }

        // Copy weight data into local memory
        {
            uint local_gate_offset = sg_id * SG_COPY_WEIGHT_BLOCK_SIZE;
            unroll_for(uint i = 0; i < SG_COPY_WEIGHT_COL_PER_CHANNEL; i++) {
                int8 gate_cols_data = *(__global int8 *)(weight_gate + ki + (sg_c + i) * K);
                intel_sub_group_block_write8((__local uint*)(local_gate + local_gate_offset + i * SIMD * dK), as_uint8(gate_cols_data));
            }
            
            unroll_for(uint i = 0; i < SG_COPY_WEIGHT_COL_PER_CHANNEL; i++) {
                int8 up_cols_data   = *(__global int8 *)(weight_up   + ki + (sg_c + i) * K);
                intel_sub_group_block_write8((__local uint*)(local_up   + local_gate_offset + i * SIMD * dK), as_uint8(up_cols_data));
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        const __local half* local_input_ptr = (const __local half *)(local_input + sg_m * (32 * BK) + (sg_n % 2) * 16 * 4);
        const __local half* local_gate_ptr  = (const __local half *)(local_gate  + sg_n * dK * SG_OUTPUT_N);
        const __local half* local_up_ptr    = (const __local half *)(local_up    + sg_n * dK * SG_OUTPUT_N);
        __attribute__((opencl_unroll_hint(1)))
        for(int kii = 0; kii < BK/dK; kii++) {
            int4 in_vec[SG_XMX_M];
            unroll_for(int m = 0; m < SG_XMX_M; m++) {
                in_vec[m] = as_int4(intel_sub_group_block_read4((__local uint*)(local_input_ptr + kii * (32 * dK)  + m * 8 * dK)));
            }

            int8 gate_vec[SG_XMX_N], up_vec[SG_XMX_N];
            unroll_for(int n = 0; n < SG_XMX_N; n++) {
                gate_vec[n] = as_int8(intel_sub_group_block_read8((__local uint*)(local_gate_ptr + kii * WG_OUTPUT_N * dK +  n * 8 * dK)));
                up_vec[n]   = as_int8(intel_sub_group_block_read8((__local uint*)(local_up_ptr   + kii * WG_OUTPUT_N * dK +  n * 8 * dK)));
            }

            unroll_for(uint n = 0; n < SG_XMX_N; n++) {
                unroll_for(uint m =0; m < SG_XMX_M; m++) {
                    gate_output[m][n] = intel_sub_group_f16_f16_split_matrix_mad_k16(in_vec[m], gate_vec[n], gate_output[m][n]);
                    up_output[m][n] = intel_sub_group_f16_f16_split_matrix_mad_k16(in_vec[m], up_vec[n], up_output[m][n]);
                }
            }
        }
        // need to sync again at the end, to avoid faster threads
        // fetching the next block into the cache before slower threads are done
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    __local half * scratch = local_input + sg_id * SG_COPY_INPUT_BLOCK_SIZE;
    unroll_for(int m = 0; m < SG_XMX_M; m++) {
        unroll_for(int n = 0; n < SG_XMX_N; n++) {
            half8 output = convert_half8(gate_output[m][n] / (1.0 + native_exp(-gate_output[m][n])) * up_output[m][n]);
            intel_sub_group_block_write_us8(scratch + (m*SG_XMX_N+n) * 8*8, as_ushort8(output));
        }
    }
    
    __global half* dst_base_ptr = out;
    // The output beginning address of this SG
    {
        uint m =  wg_m * WG_OUTPUT_M + sg_m * SG_OUTPUT_M + sg_c;
        uint n =  wg_n * WG_OUTPUT_N + sg_n * SG_OUTPUT_N;
        out += m * N + n;
    }
    
    __global half* dst_ptr = out;    
    __local half * src = (__local half *)(scratch + sg_c * 8);
    unroll_for(int m = 0; m < SG_XMX_M; m++) {
        unroll_for(int n = 0; n < SG_XMX_N; n++) {
            half8 value = *(__local half8 *)(src);
            *(__global half8 *)(dst_ptr + n * 8) = value;
            src += 8 * 8;
        }
        dst_ptr += N * 8;
    }
}

'''

from . import cl
import numpy as np
from .utils import *

# XMX 4x1:  44.92 TOPS/s  -> 56.82 TOPS/s -> 98.68 TOPS/s
# XMX 4x2:  10.00 TOPS/s
# XMX 2x2:  34.46 TOPS/s
# XMX 2x1:  30.34 TOPS/s
SG_XMX_M = 4
SG_XMX_N = 1
SIMD = 8

# subgroup number in each workgroup
WG_SG_N = 8  # max: XVE count in each XeCore
WG_SG_M = 4  # max: threads number in each XVE

# output sub-matrix shape in each subgroup
SG_OUTPUT_M = (SG_XMX_M*8)
SG_OUTPUT_N = (SG_XMX_N*8)  # For each gate and up output

#cl_kernels = kernel_cache(cl_kernel_sources, f"-D K_groups={K_groups}")
cl_kernels = kernel_cache(cl_kernel_sources, f"-DSG_XMX_M={SG_XMX_M} -DSG_XMX_N={SG_XMX_N} -DWG_SG_N={WG_SG_N} -DWG_SG_M={WG_SG_M} -DSIMD={SIMD}")

class MLP_f16xmx:
    def __init__(self, gate, up, bias):
        #self.N = N
        #self.M = M
        #self.K = K
        
        self.N, self.K = gate.shape # weight: [N, K]
        tN, tK = up.shape # weight: [N, K]
        assert(self.N==tN)
        assert(self.K==tK)

        assert(self.K % 64 == 0)
        assert(self.N % 64 == 0)
        
        self.gate = to_cl(gate)
        self.up = to_cl(up)
        if bias is not None:
            self.bias = to_cl(bias)
        else:
            self.bias = None
    
    
    def get_kernel(self,id):    
        #gws=[SIMD, N//SG_OUTPUT_N, M//SG_OUTPUT_M] = [8, N//(1*8), M//(4*8)]
        #lws=[SIMD, WG_SG_N, WG_SG_M] = [8, 8, 2]
        
        kernel_list = [["mlp_ref",        [self.M, self.N, 1],[SIMD, 1, 1]],
                       ["mlp_base_4x1",   [SIMD, self.N//SG_OUTPUT_N, self.M//SG_OUTPUT_M],[SIMD, WG_SG_N, WG_SG_M]],
                       ["mlp_base_mxn",   [SIMD, self.N//SG_OUTPUT_N, self.M//SG_OUTPUT_M],[SIMD, WG_SG_N, WG_SG_M]], # best dpas
                       ["mlp_dpasw_4x1",  [SIMD, self.N//SG_OUTPUT_N, self.M//SG_OUTPUT_M],[SIMD, WG_SG_N, WG_SG_M]]] # best dpasw
        
        return kernel_list[id]
        

    def __call__(self, input, kernel_id = 4):
        i_shape = input.shape
        o_shape = list(i_shape)
        o_shape[-1] = self.N
        output = cl.tensor(o_shape, input.dtype)
        self.M = input.numel // self.K
        #print('MLP:', i_shape, self.M, self.N, self.K)
        #assert(self.M % 128 == 0)
        
        if self.M >= 128:
            assert(self.M % 128 == 0)     
            self.kernel_id = kernel_id
            kernel_info = self.get_kernel(kernel_id)
        else:
            kernel_info = self.get_kernel(kernel_id)
                
        cl_kernels.enqueue(kernel_info[0],
                            kernel_info[1],
                            kernel_info[2], 
                            input,
                            self.gate,
                            self.up,
                            self.bias,
                            output,
                            self.M,
                            self.K,
                            self.N)
            
        return output
    
    def get_kernel_name(self):
        return self.get_kernel(self.kernel_id)[0]

    def get_gws(self):
        return self.get_kernel(self.kernel_id)[1]
    
    def get_lws(self):
        return self.get_kernel(self.kernel_id)[2]
    

if __name__ == "__main__":
    import sys
    cl.profiling(True)
    def test_matmul(shape, Bcnt = 0, kernel_id = 0):
        M, K, N = shape
        start=-1
        end=2
        A = torch.randint(start, end, [M, K], dtype=torch.float16)
        B1 = torch.randint(start, end, [N, K], dtype=torch.float16)
        B2 = torch.randint(start, end, [N, K], dtype=torch.float16)
        C = torch.randint(start, end, [M, N], dtype=torch.float16)
        #D = torch.randint(start, end, [M, N], dtype=torch.float16)
        #output = to_cl(D)

        if False:
            for j in range(M):
                for i in range(K):
                    A[j,i] = (j * K + i)*0.001;
            for j in range(N):
                for i in range(K):
                    B1[j,i] = (j * K + i)*0.001;
                    B2[j,i] = (j * K + i)*0.001; 
                                
        input = to_cl(A)
        if False:
            ref1 = torch.matmul(A, B1.transpose(1,0))
            np.set_printoptions(suppress=True)
            ref2 = torch.matmul(A, B2.transpose(1,0))
            ref1 = ref1/(1.0+torch.exp(-ref1))
            ref = (ref1*ref2).float().numpy()
        else:
            ref_imp = MLP_f16xmx(B1,B2,C)
            ref = ref_imp(input, 0).numpy()
            ref_dur = cl.finish()

        Bsize = (K*N*2 + M*K)*2
        if (Bcnt <= 0): Bcnt = 1
        linears = [MLP_f16xmx(B1,B2,C) for _ in range(Bcnt)]

        for l in linears:
            output = l(input, kernel_id)

        durs = cl.finish()
        gops = M * K * (N*2) * 2  + M * N * (3 + 1)
        actual_size = (N//WG_SG_N//SG_OUTPUT_N) * (M//WG_SG_M//SG_OUTPUT_M) * (WG_SG_M*SG_OUTPUT_M*K*2 + WG_SG_N*SG_OUTPUT_N*K*2 + WG_SG_N*SG_OUTPUT_N*K*2)
        
        print(f"GWS={linears[0].get_gws()}, LWS={linears[0].get_lws()}")
        for ns in durs:
            print(f"{linears[0].get_kernel_name()} : {shape} {Bsize*1e-6:.3f}->{actual_size*1e-6:.3f} MB {ns*1e-6:.3f} ms, BW: { actual_size/ns : .2f} GB/s,  Compute: {gops/ns/1000 : .2f} TOPS/s")
        if (len(durs)):
            mean_ns = sum(durs)/len(durs)
            print(f"[total]  {mean_ns*1e-6: .3f} ms, BW: { Bsize/mean_ns : .2f} GB/s, Compute: {gops/mean_ns/1000 : .2f} TOPS/s")

        res = output.numpy()
        print("ref =",ref[0,:8])
        print("res =",res[0,:8])
        if kernel_id < 7:
            compare(ref, res, 0.5, 0.5)

    #test_matmul([16*1024, 896*2, 2432], 20, 2)
    #sys.exit(0)

    for id in range(4):
        test_matmul([16*1024, 896*2, 2432], 20, id)
    sys.exit(0)
    
    test_matmul([16*1024, 896*2, 2432], 20, 2)
    test_matmul([16*1024, 896*2, 2432], 20, 3)
    
    #test_matmul([128, 128, 128], 1, 2)
    #test_matmul([1024, 1024, 2560], 20, 0)
    #test_matmul([1024, 1024, 2560], 20, 2)
    #test_matmul([1024, 1024, 2560], 20, 4)
    #test_matmul([256, 256, 2560], 1, 3)
    #test_matmul([1024, 1024, 2560], 20, 1)
    sys.exit(0)


    


