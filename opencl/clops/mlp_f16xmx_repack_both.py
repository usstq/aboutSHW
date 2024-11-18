cl_kernel_sources = r'''ulong __attribute__((overloadable)) intel_get_cycle_counter( void );
uint __attribute__((overloadable)) intel_get_slice_id( void );
uint __attribute__((overloadable)) intel_get_subslice_id( void );
uint __attribute__((overloadable)) intel_get_dual_subslice_id( void );
uint __attribute__((overloadable)) intel_get_eu_id( void );
uint __attribute__((overloadable)) intel_get_eu_thread_id( void );

#define unroll_for __attribute__((opencl_unroll_hint)) for

void print_as_half16(int8 result, uint offset) {
     uint sglid = (uint)get_sub_group_local_id();
     uint sg_id = (uint)get_sub_group_id();
     int sg_c = get_local_id(0); //# sub-group channel id
     int sg_n = get_local_id(1); //# sub-group id in 'N' dim
     int sg_m = get_local_id(2); //# sub-group id in 'M' dim

     half16 temp = as_half16(result);
     printf("half16[%2d-%2d] =[%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f], offset = %d\n",sg_id,sglid,
             temp[0], temp[1],temp[2],temp[3],temp[4], temp[5],temp[6],temp[7],
             temp[8], temp[9],temp[10],temp[11],temp[12], temp[13],temp[14],temp[15], offset);
}

void print_as_half8(half8 temp, uint offset) {
     uint sglid = (uint)get_sub_group_local_id();
     uint sg_id = (uint)get_sub_group_id();
     int sg_c = get_local_id(0); //# sub-group channel id
     int sg_n = get_local_id(1); //# sub-group id in 'N' dim
     int sg_m = get_local_id(2); //# sub-group id in 'M' dim

     printf("half8[%2d-%2d] = [%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f], offset = %d\n",sg_id,sglid,
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

/****************************************************************************************************************************
repack weight layout from bfyx linear to tile layout, as well as interlace for 2 weight matrix
    Don't enable hyper threads, each WG will create 16 SG, each SG run in one EU.
    Each SG will copy 2 x (16 * 8) for gate and up weights, which is cols-first layout.
    Each WG will copy (4*4) x (2x(16*8)) data
           WG_N = 4*8 = 32
           WG_M = 4*16 = 64
     1.  gws = [16,K//16,N//8]
     2.  lws = [16,4,4]
     
     WG_N = N//8//4
     WG_M = K//16//4
****************************************************************************************************************************/
#define REPACK_SGS 8        // it equals to N of XMX block  
#define REPACK_TILE_K 16    // it equals to dK
#define REPACK_TILE_N 8  // 8, it equals to N of XMX block
#define REPACK_TILE_SIZE (REPACK_TILE_K * REPACK_TILE_N)  // 16*8=128
__attribute__((intel_reqd_sub_group_size(REPACK_SGS)))
__kernel void mlp_repack(__global half* weight_gate,
                         __global half* weight_up,
                         __global half* out,
                         int N, int K) {

    int sg_c = get_local_id(0); // sub-group channel id
    int wg_k = get_group_id(1);
    int wg_n = get_group_id(2);
    
    // Start address
    weight_gate += wg_n * REPACK_TILE_N * K + wg_k * REPACK_TILE_K + sg_c * K;
    weight_up   += wg_n * REPACK_TILE_N * K + wg_k * REPACK_TILE_K + sg_c * K;
    out         += (wg_k * get_global_size(2) + wg_n) * REPACK_TILE_SIZE * 2 + sg_c * REPACK_TILE_K;
    
    uint8 gate_col = *(__global uint8 *)(weight_gate);
    uint8 up_col = *(__global uint8 *)(weight_up);
    
    #if 0
    intel_sub_group_block_write8((__global uint*)(out), gate_col);
    intel_sub_group_block_write8((__global uint*)(out + REPACK_TILE_SIZE), up_col);    
    #else
        *(__global uint8*)(out) = gate_col;
        *(__global uint8*)(out + REPACK_TILE_SIZE) = up_col;
    #endif
    
    //if(wg_k==1 && wg_n==0)
    //    print_as_half16(as_int8(gate_col), sg_c);                   
}

/****************************************************************************************************************************
repack weight layout from bfyx linear to tile layout, as well as interlace for 2 weight matrix
    Each WG will copy 2 x (64*64) for gate and up weights, which is cols-first layout.
    SG is 4*8 in WG, so each SG will copy 2*(64*64)/(8*4)=2*128
    Each channel will copy 128/8=16, only 1 col 
           WG_N = 4*8 = 32
           WG_M = 4*16 = 64
     1.  gws = [8,N//8,K//16]
     2.  lws = [8,8,4]
****************************************************************************************************************************/
#define REPACK_WG_TILE_K 64
#define REPACK_WG_TILE_N 64
#define REPACK_WG_TILE_SIZE  (REPACK_WG_TILE_K * REPACK_WG_TILE_N)   //64*64=32*128
#define REPACK_SG_TILE_K 16
#define REPACK_SG_TILE_N 8
#define REPACK_SG_TILE_SIZE  (REPACK_SG_TILE_K * REPACK_SG_TILE_N)  //128
__attribute__((intel_reqd_sub_group_size(REPACK_SGS)))
__kernel void mlp_repack_sg_tile(__global half* weight_gate,
                         __global half* weight_up,
                         __global half* out,
                         int N, int K) {

    int wg_n = get_group_id(1);
    int wg_k = get_group_id(2);
    
    int sg_c = get_local_id(0); // sub-group channel id
    int sg_n = get_local_id(1); // sub-group id in 'N' dim
    int sg_k = get_local_id(2); // sub-group id in 'K' dim
    
    __global half* weight_gate_base = weight_gate;
    __global half* out_base = out;
    
    // Start address of WG
    uint weight_offset = wg_n * REPACK_WG_TILE_N * K + wg_k * REPACK_WG_TILE_K;
    // Start address of SG
    weight_offset += sg_n * REPACK_SG_TILE_N * K + sg_k * REPACK_SG_TILE_K;
    
    weight_gate += weight_offset + sg_c * K;
    weight_up   += weight_offset + sg_c * K;
    
    // Start address of WG
    int wg_size_n = get_global_size(1)/get_local_size(1);
    int wg_size_k = get_global_size(2)/get_local_size(2);
    out  += (wg_k * wg_size_n + wg_n) * REPACK_WG_TILE_SIZE * 2;
    // Start address of SG
    out  += (sg_k * 8 + sg_n) * REPACK_SG_TILE_SIZE * 2 + sg_c * REPACK_SG_TILE_K;
        
    uint8 gate_col = *(__global uint8 *)(weight_gate);
    uint8 up_col = *(__global uint8 *)(weight_up);
    
    #if 0
        intel_sub_group_block_write8((__global uint*)(out), gate_col);
        intel_sub_group_block_write8((__global uint*)(out + REPACK_SG_TILE_SIZE), up_col);    
    #else
        *(__global uint8*)(out) = gate_col;
        *(__global uint8*)(out + REPACK_SG_TILE_SIZE) = up_col;
    #endif
    
    if(wg_k==1 && wg_n==1 && sg_c==0) {
        int wg_id = wg_k * wg_size_n + wg_n;
        int sg_size = get_local_size(1) * get_local_size(2);
        int g_sg_id = wg_id * sg_size + get_sub_group_id();
        //print_as_half16(as_int8(gate_col), g_sg_id);
        //print_as_half16(as_int8(gate_col), (out - out_base)/REPACK_SG_TILE_SIZE/2);
    }                
}

/****************************************************************************************************************************
repack input matrix from bfyx linear to slice tile layout for each WG
    Easch SG will copy 8x16=128 data for f16 format
       WG_SG_N=4
       WG_SG_M=B//8, supposed B=128, then WG_SG_M=128//8=16
       Each WG will have (B//8)*4 = 16*4=64 SGs
    Each WG will copy 8*16*64=8092, which transpose from rows-first layout to cols_first.
    Each channel will copy 128/8=16, only 1 rows 
           WG_SG_N = 4
           WG_SG_M = 16
     1.  gws = [8,K//16,M//8]
     2.  lws = [8,4,B//8] = [8,4,16]
     
input: linear byfx layout
output:  Meta tile block [B=128,64]
         tile block [32, 64]
         sublock [8,16]     
     
         sb0 sb16 sb32 sb48
         sb1 sb17 sb33 sb49
         sb2 sb18 sb34 sb50
         sb3 sb19 sb35 sb51
         sb4 sb20 sb36 sb52
         sb5 sb21 sb37 sb53
         ...
         sb14 sb30 sb46 sb62
         sb15 sb31 sb47 sb63
 ****************************************************************************************************************************/
#define REPACK_INPUT_WG_TILE_K 64
#define REPACK_INPUT_WG_TILE_M B   //128
#define REPACK_INPUT_WG_TILE_SIZE  (REPACK_INPUT_WG_TILE_K * REPACK_INPUT_WG_TILE_M)   //64*128
#define REPACK_INPUT_SG_TILE_K 16
#define REPACK_INPUT_SG_TILE_M 8
#define REPACK_INPUT_SG_TILE_SIZE  (REPACK_SG_TILE_K * REPACK_SG_TILE_M)  //128
#define REPACK_INPUT_SG_CHANNEL_COPY_ROWS  (REPACK_INPUT_SG_TILE_SIZE/REPACK_INPUT_SG_TILE_K/REPACK_SGS)  // 128/16/8=1
#define REPACK_INPUT_WG_SG_N 4
#define REPACK_INPUT_WG_SG_M (B>>3)  //B/8=128/8=16
#define REPACK_INPUT_WG_N (K/REPACK_INPUT_WG_TILE_K)
#define REPACK_INPUT_WG_M (M/B)      //M/B
__attribute__((intel_reqd_sub_group_size(REPACK_SGS)))
__kernel void mlp_repack_input(__global half* input,
                         __global half* out,
                         int M, int K) {
    int B=128;
    int wg_n = get_group_id(1);
    int wg_m = get_group_id(2);
    
    int sg_c = get_local_id(0); // sub-group channel id
    int sg_n = get_local_id(1); // sub-group id in 'N' dim
    int sg_m = get_local_id(2); // sub-group id in 'M' dim
    
    __global half* input_base = input;
    __global half* out_base = out;

    // Start address of WG
    uint input_offset = wg_m * REPACK_INPUT_WG_TILE_M * K + wg_n * REPACK_INPUT_WG_TILE_K;
    // Start address of SG
    input_offset += sg_m * REPACK_INPUT_SG_TILE_M * K + sg_n * REPACK_INPUT_SG_TILE_K;
    // Start address of Channel
    input += input_offset + sg_c * K;
    
    // Start address of WG
    out  += (wg_m * REPACK_INPUT_WG_N + wg_n) * REPACK_INPUT_WG_TILE_SIZE;
    // Start address of SG
    out  += (sg_m + sg_n * REPACK_INPUT_WG_SG_M) * REPACK_SG_TILE_SIZE + sg_c * REPACK_SG_TILE_K;
        
    uint8 value = *(__global uint8 *)(input);
    *(uint8 *)(out) = value;              
}

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
#define WG_SG_M 4
#endif
*/

#define SG_SIZE SIMD
#define XMX_OUT_BLOCK_M 8   // Can be: 1,2,4,8
#define XMX_OUT_BLOCK_N 8   // fixed value

#define XMX_IN_BLOCK_M  XMX_OUT_BLOCK_M
#define XMX_IN_BLOCK_K 16    // fixed: half:16, char:32
#define XMX_WEI_BLOCK_K XMX_IN_BLOCK_K
#define XMX_WEI_BLOCK_N 8    // fixed value

#define XMX_OUT_BLOCK_SIZE (XMX_OUT_BLOCK_M*XMX_OUT_BLOCK_N) // 8*8=64
#define XMX_INPUT_BLOCK_SIZE (XMX_IN_BLOCK_M*XMX_IN_BLOCK_K) // 8*16=128
#define XMX_WEI_BLOCK_SIZE (XMX_WEI_BLOCK_N*XMX_WEI_BLOCK_K) // 8*16=128

// output sub-matrix shape in each subgroup
#define SG_OUTPUT_M (SG_XMX_M*XMX_OUT_BLOCK_M)  // 4*8
#define SG_OUTPUT_N (SG_XMX_N*XMX_OUT_BLOCK_N)  // 1*8 for each gate and up output
#define SG_INPUT_M  SG_OUTPUT_M  // 4*8
#define SG_INPUT_BLOCK_SIZE  (SG_INPUT_M*dK)    // 32*16=512
#define SG_WEI_N SG_OUTPUT_N // 8
#define SG_WEI_BLOCK_SIZE   (SG_OUTPUT_N*dK) // 8 *16=128

// output sub-matrix size in each workgroup
#define WG_OUTPUT_M  (SG_OUTPUT_M * WG_SG_M)  // 32*4=128
#define WG_OUTPUT_N  (SG_OUTPUT_N * WG_SG_N)  // 8*8=64
#define WG_INPUT_M   WG_OUTPUT_M  // 32*4=128

// step of K in each iteration, it is limited by SLM 
#define BK 64
#define dK XMX_IN_BLOCK_K      // 16: contains 16 half/32 bytes
#define SG_INPUT_BLOCK_K  (BK/dK)    // 64/16=4

// 4*8*64/8=256
#define SG_COPY_INPUT_BLOCK_SIZE  (SG_OUTPUT_M * BK/WG_SG_N)
#define SG_COPY_INPUT_SUB_BLOCK_SIZE (8*dK)
// 4*64/8/16=2    256/8/16=2
#define SG_COPY_INPUT_ROW_PER_CHANNEL (SG_COPY_INPUT_BLOCK_SIZE/SIMD/XMX_IN_BLOCK_K)
#define SG_COPY_INPUT_ROW_NUM         (SG_COPY_INPUT_BLOCK_SIZE/dK)  // 256/16=16

// 8*64/4=128
#define SG_COPY_WEIGHT_BLOCK_SIZE  (SG_OUTPUT_N * BK / WG_SG_M )  
// 128/16/8=1
#define SG_COPY_WEI_COL_PER_CHANNEL  (SG_COPY_WEIGHT_BLOCK_SIZE/XMX_IN_BLOCK_K/SIMD)
#define WG_COPY_WEI_SIZE  (SG_COPY_WEIGHT_BLOCK_SIZE * WG_SG_N * WG_SG_M)   // 128*8*4= 4096

__attribute__((intel_reqd_sub_group_size(SG_SIZE)))
__kernel void mlp_dpas_repack_both(__global const half* input,
                       __global const half* weight_gate_up, // 16*8 tile interlace layout
                       __global const half* bias,
                       __global half* out,
                       int M, int K, int N) {
                           
    __local half local_input[WG_OUTPUT_M * BK]; // 128*64*2= 16 KB
    __local half local_gate[WG_OUTPUT_N * BK];  // 64*64*2 = 8 KB
    __local half local_up[WG_OUTPUT_N * BK];    // 64*64*2 = 8 KB
    
    int wg_n = get_group_id(1);
    int wg_m = get_group_id(2);
    int sg_id = get_sub_group_id();
    
    //print_wi_info();
    int sg_c = get_local_id(0); //# sub-group channel id
    int sg_n = get_local_id(1); //# sub-group id in 'N' dim
    int sg_m = get_local_id(2); //# sub-group id in 'M' dim
    
    float8 gate_output[SG_XMX_M][SG_XMX_N] = {0.0f};
    float8 up_output[SG_XMX_M][SG_XMX_N] = {0.0f};
    
    unroll_for(uint ki = 0; ki < K; ki += BK) {
        // Copy input data into local memory
        {
            uint local_input_offset = (sg_m*16 + sg_n*2) * SG_COPY_INPUT_SUB_BLOCK_SIZE;
            uint input_offset = wg_m * WG_OUTPUT_M * K + ki * WG_OUTPUT_M
                              + ((sg_n/2)*16 + (sg_n%2)*2 + sg_m*4) * SG_COPY_INPUT_SUB_BLOCK_SIZE;
            unroll_for(uint i = 0; i < SG_COPY_INPUT_ROW_PER_CHANNEL; i++) {
                #if 1
                int8 row_data = *(__global int8 *)(input + input_offset + i * SG_COPY_INPUT_SUB_BLOCK_SIZE + sg_c * dK);
                *(int8 *)(local_input + local_input_offset + i * SG_COPY_INPUT_SUB_BLOCK_SIZE + sg_c * dK) = row_data;
                #else
                int8 row_data = as_int8(intel_sub_group_block_read8((__global uint*)(input + input_offset + i * SG_COPY_INPUT_SUB_BLOCK_SIZE)));
                intel_sub_group_block_write8((__local uint*)(local_input + local_input_offset + i * SG_COPY_INPUT_SUB_BLOCK_SIZE), as_uint8(row_data));
                #endif
                //if(i==0 && sg_c==0 && sg_m==0 && sg_n==1) {
                    //print_as_half16(row_data, input_offset/SG_COPY_INPUT_SUB_BLOCK_SIZE);
                //}
            }
        }

        // Copy weight data into local memory
        {
            int wg_size_n = get_global_size(1)/get_local_size(1);
            uint wei_offset = wg_n * WG_COPY_WEI_SIZE * 2 + (sg_m * WG_SG_N  + sg_n)* SG_COPY_WEIGHT_BLOCK_SIZE * 2 + ki/BK * wg_size_n * WG_COPY_WEI_SIZE * 2;
            
            uint gate_offset = wei_offset + sg_c * dK;
            uint up_offset = gate_offset + SG_COPY_WEIGHT_BLOCK_SIZE;
            uint local_gate_offset = (sg_m * SG_COPY_WEI_COL_PER_CHANNEL * WG_SG_N + sg_n) * SG_COPY_WEIGHT_BLOCK_SIZE; // block_write
            
            int8 gate_cols_data = *(__global int8 *)(weight_gate_up + gate_offset);
            int8 up_cols_data   = *(__global int8 *)(weight_gate_up + up_offset);
            intel_sub_group_block_write8((__local uint*)(local_gate + local_gate_offset), as_uint8(gate_cols_data));
            intel_sub_group_block_write8((__local uint*)(local_up + local_gate_offset), as_uint8(up_cols_data));
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        const __local half* local_input_ptr = (const __local half *)(local_input + sg_m * WG_SG_N * SG_COPY_INPUT_BLOCK_SIZE);
        const __local half* local_gate_ptr  = (const __local half *)(local_gate  + sg_n * SG_WEI_BLOCK_SIZE);
        const __local half* local_up_ptr    = (const __local half *)(local_up    + sg_n * SG_WEI_BLOCK_SIZE);
        __attribute__((opencl_unroll_hint(1)))
        for(int kii = 0; kii < BK/dK; kii ++) {
            int8 in_vec[SG_XMX_M];
            int8 gate_vec[SG_XMX_M], up_vec[SG_XMX_M];
            
            unroll_for(int m = 0; m < SG_XMX_M; m++) {
                in_vec[m] = as_int8(intel_sub_group_block_read8((__local uint*)(local_input_ptr + kii * SG_INPUT_BLOCK_SIZE + m * XMX_INPUT_BLOCK_SIZE)));
            }
            //if(sg_n==0 && sg_m==0 && sg_c==0){
            //    int  m = 0;
            //    print_as_half16(as_int8(in_vec[m]), local_input_ptr + kii * SG_INPUT_BLOCK_SIZE + m * XMX_INPUT_BLOCK_SIZE - local_input);
            //}
    
            unroll_for(int n = 0; n < SG_XMX_N; n++) {
                gate_vec[n] = as_int8(intel_sub_group_block_read8((__local uint*)(local_gate_ptr + kii * WG_SG_N * SG_WEI_BLOCK_SIZE + n * dK * XMX_OUT_BLOCK_N)));
                up_vec[n]   = as_int8(intel_sub_group_block_read8((__local uint*)(local_up_ptr   + kii * WG_SG_N * SG_WEI_BLOCK_SIZE + n * dK * XMX_OUT_BLOCK_N)));
            }

            if(sg_n==0 && sg_m==0 && sg_c==0 && kii==0 && ki==0){
                int  n = 0;
                //print_as_half16(as_int8(gate_vec[n]), wg_n);
                //print_as_half16(as_int8(up_vec[n]), local_gate_ptr + kii * WG_SG_N * SG_WEI_BLOCK_SIZE + n * dK * XMX_OUT_BLOCK_N - local_gate);
                //int m = 2;
                //print_as_half16(as_int8(in_vec[m]), local_input_ptr + kii * SG_INPUT_BLOCK_SIZE + m * XMX_INPUT_BLOCK_SIZE - local_input);
            }
            
            unroll_for(uint m = 0; m < SG_XMX_M; m++) {
                unroll_for(uint n =0; n < SG_XMX_N; n++) {
                    gate_output[m][n] = intel_sub_group_f16_f16_matrix_mad_k16(in_vec[m], gate_vec[n], gate_output[m][n]);
                    up_output[m][n] = intel_sub_group_f16_f16_matrix_mad_k16(in_vec[m], up_vec[n], up_output[m][n]);
                }
            }

        }
        //# need to sync again at the end, to avoid faster threads
        //# fetching the next block into the cache before slower threads are done
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    __local half * scratch = local_input + sg_id * XMX_OUT_BLOCK_SIZE * SG_XMX_M * SG_XMX_N;
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
__attribute__((intel_reqd_sub_group_size(SG_SIZE)))
__kernel void mlp_dpasw_repack_both(__global const half* input,
                       __global const half* weight_gate_up, // 16*8 tile interlace layout
                       __global const half* bias,
                       __global half* out,
                       int M, int K, int N) {
                           
    __local half local_input[WG_OUTPUT_M * BK]; // 128*64*2= 16 KB
    __local half local_gate[WG_OUTPUT_N * BK];  // 64*64*2 = 8 KB
    __local half local_up[WG_OUTPUT_N * BK];    // 64*64*2 = 8 KB
    
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
        {
            uint local_input_offset = (sg_m*16 + sg_n*2) * SG_COPY_INPUT_SUB_BLOCK_SIZE;
            uint input_offset = wg_m * WG_OUTPUT_M * K + ki * WG_OUTPUT_M
                              + ((sg_n/2)*16 + (sg_n%2)*2 + sg_m*4) * SG_COPY_INPUT_SUB_BLOCK_SIZE;
            unroll_for(uint i = 0; i < SG_COPY_INPUT_ROW_PER_CHANNEL; i++) {
                #if 1
                int8 row_data = *(__global int8 *)(input + input_offset + i * SG_COPY_INPUT_SUB_BLOCK_SIZE + sg_c * dK);
                *(int8 *)(local_input + local_input_offset + i * SG_COPY_INPUT_SUB_BLOCK_SIZE + sg_c * dK) = row_data;
                #else
                int8 row_data = as_int8(intel_sub_group_block_read8((__global uint*)(input + input_offset + i * SG_COPY_INPUT_SUB_BLOCK_SIZE)));
                intel_sub_group_block_write8((__local uint*)(local_input + local_input_offset + i * SG_COPY_INPUT_SUB_BLOCK_SIZE), as_uint8(row_data));
                #endif
            }
        }

        // Copy weight data into local memory
        {
            int wg_size_n = get_global_size(1)/get_local_size(1);
            uint wei_offset = wg_n * WG_COPY_WEI_SIZE * 2 + (sg_m * WG_SG_N  + sg_n)* SG_COPY_WEIGHT_BLOCK_SIZE * 2 + ki/BK * wg_size_n * WG_COPY_WEI_SIZE * 2;
            
            uint gate_offset = wei_offset + sg_c * dK;
            uint up_offset = gate_offset + SG_COPY_WEIGHT_BLOCK_SIZE;
            uint local_gate_offset = (sg_m * SG_COPY_WEI_COL_PER_CHANNEL * WG_SG_N + sg_n) * SG_COPY_WEIGHT_BLOCK_SIZE; // block_write
            
            int8 gate_cols_data = *(__global int8 *)(weight_gate_up + gate_offset);
            int8 up_cols_data   = *(__global int8 *)(weight_gate_up + up_offset);
            intel_sub_group_block_write8((__local uint*)(local_gate + local_gate_offset), as_uint8(gate_cols_data));
            intel_sub_group_block_write8((__local uint*)(local_up + local_gate_offset), as_uint8(up_cols_data));
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        const __local half* local_input_ptr = (const __local half *)(local_input + sg_m * WG_SG_N * SG_COPY_INPUT_BLOCK_SIZE);
        const __local half* local_gate_ptr  = (const __local half *)(local_gate  + sg_n * SG_WEI_BLOCK_SIZE);
        const __local half* local_up_ptr    = (const __local half *)(local_up    + sg_n * SG_WEI_BLOCK_SIZE);
        __attribute__((opencl_unroll_hint(1)))
        for(int kii = 0; kii < BK/dK; kii ++) {
            int4 in_vec[SG_XMX_M];
            int8 gate_vec[SG_XMX_M], up_vec[SG_XMX_M];
            
            unroll_for(int m = 0; m < SG_XMX_M; m++) {
                in_vec[m] = as_int4(intel_sub_group_block_read4((__local uint*)(local_input_ptr + kii * SG_INPUT_BLOCK_SIZE + m * XMX_INPUT_BLOCK_SIZE + (sg_id % 2)* (4*16))));
            }
    
            unroll_for(int n = 0; n < SG_XMX_N; n++) {
                gate_vec[n] = as_int8(intel_sub_group_block_read8((__local uint*)(local_gate_ptr + kii * WG_SG_N * SG_WEI_BLOCK_SIZE + n * dK * XMX_OUT_BLOCK_N)));
                up_vec[n]   = as_int8(intel_sub_group_block_read8((__local uint*)(local_up_ptr   + kii * WG_SG_N * SG_WEI_BLOCK_SIZE + n * dK * XMX_OUT_BLOCK_N)));
            }

            unroll_for(uint m = 0; m < SG_XMX_M; m++) {
                unroll_for(uint n =0; n < SG_XMX_N; n++) {
                    gate_output[m][n] = intel_sub_group_f16_f16_split_matrix_mad_k16(in_vec[m], gate_vec[n], gate_output[m][n]);
                    up_output[m][n] = intel_sub_group_f16_f16_split_matrix_mad_k16(in_vec[m], up_vec[n], up_output[m][n]);
                }
            }

        }
        // need to sync again at the end, to avoid faster threads
        // fetching the next block into the cache before slower threads are done
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    __local half * scratch = local_input + sg_id * XMX_OUT_BLOCK_SIZE * SG_XMX_M * SG_XMX_N;
    unroll_for(int m = 0; m < SG_XMX_M; m++) {
        unroll_for(int n = 0; n < SG_XMX_N; n++) {
            half8 output = convert_half8(gate_output[m][n] / (1.0 + native_exp(-gate_output[m][n])) * up_output[m][n]);
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

'''

from . import cl
import numpy as np
from .utils import *

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

class MLP_f16xmx_repack_both:
    def __init__(self, gate, up, bias):
        
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

        self.input_repack = None
        gate_up_shape=list(gate.shape)
        gate_up_shape[0] *= 2
        #print('MLP: gate_shape = ', gate_up_shape, self.N, self.K) 
        self.gate_up = cl.tensor(gate_up_shape, self.gate.dtype)
        if False:
            cl_kernels.enqueue("mlp_repack",
                            [8, tK//16, tN//8],
                            [8, 1, 1], 
                            self.gate,
                            self.up,
                            self.gate_up,
                            tN,
                            tK)
        else:
            cl_kernels.enqueue("mlp_repack_sg_tile",
                            [8, tN//8, tK//16],
                            [8, 8, 4], 
                            self.gate,
                            self.up,
                            self.gate_up,
                            tN,
                            tK)
            
            
        durs = cl.finish()
        Bsize = tN * tK * 2 * 2
        for ns in durs:
            print(f"mlp_repack : {gate.shape} {Bsize*1e-6:.3f} MB costs {ns*1e-6:.3f} ms, R/W BW: { 2*Bsize/ns : .2f} GB/s")

    def get_kernel(self,id):        
        kernel_list = [["mlp_ref",               [self.M, self.N, 1],[SIMD, 1, 1]],
                       ["mlp_dpas_repack_both",       [SIMD, self.N//SG_OUTPUT_N, self.M//SG_OUTPUT_M],[SIMD, WG_SG_N, WG_SG_M]],
                       ["mlp_dpasw_repack_both",      [SIMD, self.N//SG_OUTPUT_N, self.M//SG_OUTPUT_M],[SIMD, WG_SG_N, WG_SG_M]]]
        
        return kernel_list[id]
        

    def __call__(self, input, kernel_id = 2):
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
          
        if kernel_id == 0:
            # ref implement      
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
        else:
            if self.input_repack is None or self.input_repack.shape != input.shape:
                self.input_repack = cl.tensor(list(input.shape), input.dtype)

            cl_kernels.enqueue("mlp_repack_input",
                               [8,self.K//16,self.M//8],
                               [8,4,16],
                               input,
                               self.input_repack,
                               self.M,
                               self.K)
            #print(self.input_repack.numpy()[2*2, :])
            #print(self.input_repack.numpy().shape)
            #durs = cl.finish()

            cl_kernels.enqueue(kernel_info[0],
                            kernel_info[1],
                            kernel_info[2], 
                            self.input_repack,
                            self.gate_up,
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
        
        if False:
            np.random.seed(0)
            A = np.random.randint(start, end, [M, K]).astype(np.float16)
            B1 = np.random.randint(start, end, [N, K]).astype(np.float16)
            B2 = np.random.randint(start, end, [N, K]).astype(np.float16)
            C = np.random.randint(start, end, [M,N]).astype(np.float16)
        else:
            A = torch.randint(start, end, [M, K], dtype=torch.float16)
            B1 = torch.randint(start, end, [N, K], dtype=torch.float16)
            B2 = torch.randint(start, end, [N, K], dtype=torch.float16)
            C = torch.randint(start, end, [M, N], dtype=torch.float16)
        print("prepare data ...done") 

        if False:
            for j in range(M):
                for i in range(K):
                    A[j,i] = (j * K + i)*0.001;
            for j in range(N):
                for i in range(K):
                    B1[j,i] = (j * K + i)*0.001;
                    B2[j,i] = (j * K + i)*0.001; 
               
            
        #print("prepare A ...")    
        input = to_cl(A)
        
        np.set_printoptions(suppress=True)
        if False:
            ref1 = torch.matmul(A, B1.transpose(1,0))
            np.set_printoptions(suppress=True)
            print("A = ",A.numpy())
            #print("B1 = ",B1.numpy())
            #print("B2 = ",B2.numpy())
            #print("ref1 = ",ref1.numpy()[:8,:8])
            ref2 = torch.matmul(A, B2.transpose(1,0))
            #print("ref2 = ",ref2.numpy())
            #print("ref1*ref2 = ",ref1*ref2)
            ref1 = ref1/(1.0+torch.exp(-ref1))
            #print("swiglu(ref1) =", ref1.numpy()[:8,:8])
            ref = (ref1*ref2).float().numpy()
            #print("ref =", ref[:8,:8])
        else:
            ref_imp = MLP_f16xmx_repack_both(B1,B2,C)
            ref = ref_imp(input, 0).numpy()
            #ref_dur = cl.finish()

        Bsize = (K*N*2 + M*K)*2
        if (Bcnt <= 0): Bcnt = 1
        linears = [MLP_f16xmx_repack_both(B1,B2,C) for _ in range(Bcnt)]

        for l in linears:
            output = l(input, kernel_id)

        durs = cl.finish()
        gops = M * K * (N*2) * 2  + M * N * (3 + 1)
        actual_size = (N//WG_SG_N//SG_OUTPUT_N) * (M//WG_SG_M//SG_OUTPUT_M) * (WG_SG_M*SG_OUTPUT_M*K*2 + WG_SG_N*SG_OUTPUT_N*K*2 + WG_SG_N*SG_OUTPUT_N*K*2)
        
        print(f"GWS={linears[0].get_gws()}, LWS={linears[0].get_lws()}")
        is_input_repack = True
        total_ns = 0.0
        for ns in durs:
            if is_input_repack:
                print(f"mlp_input_repack : {shape} {(M*K*2)*1e-6:.3f} MB costs {ns*1e-6:.3f} ms, R/W BW: { 2*(M*K*2)/ns : .2f} GB/s")
            else:
                total_ns += ns
                print(f"{linears[0].get_kernel_name()} : {shape} {Bsize*1e-6:.3f}->{actual_size*1e-6:.3f} MB {ns*1e-6:.3f} ms, BW: { actual_size/ns : .2f} GB/s,  Compute: {gops/ns/1000 : .2f} TOPS/s")
            is_input_repack = not is_input_repack
        if (len(durs)):
            #mean_ns = sum(durs)/len(durs)
            mean_ns = total_ns/(len(durs)/2)
            print(f"[total]  {mean_ns*1e-6: .3f} ms, Compute: {gops/mean_ns/1000 : .2f} TOPS/s")

        res = output.numpy()
        print("ref =",ref[0,:8])
        print("res =",res[0,:8])
        compare(ref, res, 0.5, 0.5)


    test_matmul([128, 64, 64], 1, 1)
    test_matmul([16*1024, 896*2, 2432], 20, 2)
    #test_matmul([16*1024, 896*2, 2432], 20, 2)
    sys.exit(0)

    #test_matmul([128, 256, 256], 1, 1)
    #test_matmul([128, 64, 64], 1, 1)
    test_matmul([1024, 2560, 1024], 20, 1)
    test_matmul([1024, 2560, 1024], 20, 2)
    test_matmul([1024, 2560, 1024], 20, 3)
    #test_matmul([256, 256, 2560], 1, 3)
    #test_matmul([1024, 1024, 2560], 20, 1)
    sys.exit(0)

    for id in range(8):
        test_matmul([1024, 1024, 2560], 20, id)

    


