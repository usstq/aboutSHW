from . import cl
import numpy as np
from .utils import *
from . import compare
from . import utils

cl_kernel_sources_ref = r'''
    __kernel void gemmA(__global half * A, __global half *B,  __global half *CC, __global half * alpha, int N, int K, int k_blk) {
#if 1
        // Seems kblocks accumulation would introduce error. So ref kernel accumulation should behave same with target.
        int n_idx = get_global_id(0);
        size_t blk_num = (K + k_blk -1 )/ k_blk;
        half sum = 0.f;
        for (int i = 0; i < blk_num; i++) {
            half sub_sum = 0.f;
            int k_idx = i*k_blk;
            // incase K%k_blk != 0;
            if ((k_idx + k_blk) > K)
                k_blk = K - k_idx;
            for (size_t sub_k = 0; sub_k < k_blk; sub_k++) {
                sub_sum = fma(A[k_idx], B[k_idx*N+n_idx], sub_sum);
                k_idx++;
            }
            sum += sub_sum;
        }
        CC[n_idx] = sum * alpha[0];
#else
        int n_idx = get_global_id(0);
        half sum = 0.f;
        for (int i = 0; i < K; i++)
            sum = fma(A[i], B[i*N+n_idx], sum);
        CC[n_idx] = sum;
#endif
    }
    __kernel void reduce(__global half * temp_C, __global half *C, int N, int cnt) {
        int n_idx = get_global_id(0);
        half sum = 0.f;
        for (int i = 0; i < cnt; i++)
            sum += temp_C[i*N + n_idx];
        C[n_idx] = sum;
    }
    
    __kernel void gemmB(__global half * A, __global half *B,  __global half *CC, int N, int K) {
        int n_idx = get_global_id(0);
        half sum = 0.f;
        for (int i = 0; i < K; i++)
            sum = fma(A[i], B[i*N+n_idx], sum);
        CC[n_idx] = sum;
    }
'''

cl_kernel_sources = r'''
    __attribute__((intel_reqd_sub_group_size(SG_SZ)))
    __kernel void gemmA(__global half * A, __global half *B,  __global half *temp_res, int K) {
        int gid =  get_group_id(0);
        int sgid = get_sub_group_id();
        int lid =  get_sub_group_local_id();
        //How many K is accumulated in the WG.
        int k_start_grp = gid * K_PER_WG;
        int k_end_grp = (k_start_grp + K_PER_WG) > K ?  (K): (k_start_grp+K_PER_WG);

        // store each sg accumulation result into SLM. Will sum sg result into wg result. 
        __local half sgs_sum_buff[SG_NUM * RANK];
        __local half *sg_buffer_ptr = sgs_sum_buff + sgid * RANK;
        __global half *C_ptr = temp_res + RANK * gid;

#if GEMMA_USE_SLM
        //Workgroup share same input activation and each subgroup would access all the elements of the shared intput.Put it into SLM.
        __local half local_input[K_PER_WG];
        //sg could diverge here. not all the sgs can satisfy 'offset < k_len'.
        __attribute__((opencl_unroll_hint))
        for (int offset = sgid * SG_SZ; offset < (k_end_grp - k_start_grp); offset += SG_SZ * SG_NUM ) {
            __global half *input_ptr = A + k_start_grp + offset;
            half copy_val = as_half(intel_sub_group_block_read_us((const __global ushort*)input_ptr));
            local_input[offset + lid] = copy_val;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
#endif
        int k_start_sg = k_start_grp + sgid * K_PER_SG;
        __global half *A_ptr_sg = A + k_start_sg;
        __global half *B_ptr_sg = B + k_start_sg * RANK;

        //The sg is needs to accumulate. Otherwise, sg not needed. sg diverge here.
        if (k_start_sg <  k_end_grp) {
            int klen_sg = ((k_start_sg + K_PER_SG) > k_end_grp) ? (k_end_grp-k_start_sg): K_PER_SG;

            for (int nblk = 0; nblk < RANK / SG_SZ; nblk++) {
                __global half *A_ptr = A_ptr_sg;
                __global half *B_ptr = B_ptr_sg + nblk * SG_SZ;
                half sum = 0.f;
                for (int kk = 0;  kk < klen_sg; kk += SG_SZ) {
#if GEMMA_USE_SLM
                    ushort input = intel_sub_group_block_read_us((const __local ushort*)(local_input + sgid * (int)(K_PER_SG) + kk));
#else
                    ushort input = intel_sub_group_block_read_us((const __global ushort*)(A_ptr + kk));
#endif
                    __attribute__((opencl_unroll_hint))
                    for (int j = 0; j < SG_SZ; j++) {
                        half bb = as_half(intel_sub_group_block_read_us((const __global ushort*)(B_ptr)));
                        half aa = as_half(intel_sub_group_broadcast(input, j));
                        sum = fma(aa, bb, sum);
                        B_ptr += RANK;
                    }
                }
                *(sg_buffer_ptr + nblk * SG_SZ + lid) = sum;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        int sg_offset = sgid * SG_SZ;
        //Each SG would try updating data.
         __attribute__((opencl_unroll_hint))
        for (int base = 0; (sg_offset  + base ) < RANK; base += SG_SZ * SG_NUM ) {
            half sum = 0.f;
            __attribute__((opencl_unroll_hint))
            for (int i = 0;  i < SG_NUM; i++) {
                sum += as_half(intel_sub_group_block_read_us((const __local ushort*)(sgs_sum_buff + i * RANK + sg_offset)));
            }
            intel_sub_group_block_write_us((const __global ushort*)(C_ptr + sg_offset), as_short(sum));
        }
    }


    __attribute__((intel_reqd_sub_group_size(SG_SZ)))
    __kernel void gemmB(__global half * A,  __global half *B,  __global half *CC, __global half * alpha, int N) {
        int wg_id = get_group_id(0);
        int sg_id = get_sub_group_id();
        int sg_num = get_num_sub_groups();
        int sg_idx = wg_id * sg_num  +  sg_id;
        int id_sg_local = get_sub_group_local_id();
#if GEMMB_USE_SLM
        __local half reduce[RANK];
#else
        half reduce[RANK] = {0.f};
#endif

        __global half *C_ptr = CC + sg_idx * SG_SZ;
        __global half *A_ptr = A;
        __global half *B_ptr = B + sg_idx * SG_SZ;

        //1. Reduce
#if GEMMB_USE_SLM
        //EACH WG would reduce input activation and save into local memory `reduce[RANK]`.
        int local_sz = get_local_size(0);
        //sg would diverge here. not all the sgs can satisfy 'offset < RANK'.
        __attribute__((opencl_unroll_hint))
        for (int offset = sg_id * SG_SZ; offset < RANK; offset += local_sz) {
            __global half *subA_ptr = A_ptr + offset;
            __attribute__((opencl_unroll_hint))
            for (int part_idx = 0; part_idx < PART_NUM; part_idx++) {
                half partial_val = as_half(intel_sub_group_block_read_us((const __global ushort*)subA_ptr));
                reduce[offset + id_sg_local] += partial_val;
                subA_ptr += RANK;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
#else
        //Each work item would reduce the input activation.
        for (int i = 0; i < PART_NUM; i++) {
            __attribute__((opencl_unroll_hint))
            for (int j = 0; j < RANK; j++) {
                reduce[j] += A_ptr[i * RANK + j];
            }
        }
#endif
        //2.  GEMMB
        half sum = 0;
        for (int kk = 0; kk < RANK; kk += SG_SZ) {
#if GEMMB_USE_SLM
            half input = as_half(intel_sub_group_block_read_us((const __local ushort*)(reduce + kk)));
#else
            half input = reduce[id_sg_local + kk];
#endif
            input *= alpha[0];
            __attribute__((opencl_unroll_hint))
            for (int j = 0; j < SG_SZ; j++) {
                half bb = as_half(intel_sub_group_block_read_us((const __global ushort*)B_ptr));
                half aa = as_half(intel_sub_group_broadcast(as_ushort(input), j));
                sum = fma(aa, bb, sum);
                B_ptr += N;
            }
        }
        intel_sub_group_block_write_us((const __global ushort*)C_ptr, as_short(sum));
    }
    '''
class LORA:
    def __init__(self, rank, input_state, output_state, gemma_wgs, gemma_sgs_per_wg, gemma_slm = True, gemmb_slm = True, use_ref = False):
        self.rank = rank
        self.input_state = input_state
        self.output_state = output_state
        self.gemmb_local_sz = 128
        self.sg_sz = 16
        self.gemma_slm = gemma_slm
        self.gemmb_slm = gemmb_slm
        self.use_ref = use_ref
        self.gemma_wgs = gemma_wgs
        self.gemma_sgs_per_wg = gemma_sgs_per_wg

        assert self.input_state % self.sg_sz == 0, f"'input state' {self.input_state} is not multiple of SG_SZ {self.sg_sz}"
        assert self.rank % self.sg_sz == 0, f"'RANK' {self.rank} is not multiple of SG_SZ {self.sg_sz}"
        assert self.output_state % self.sg_sz == 0, f"'output state' {self.output_state} is not multiple of SG_SZ {self.sg_sz}"
        assert (self.output_state) % self.gemmb_local_sz == 0, f"GEMMB: 'global size' {self.output_state} is not multiple of 'local size' {self.gemmb_local_sz}"
        
        INPUT_STATE_SG_NUM  = self.input_state // self.sg_sz
        self.input_state_per_wg = ((INPUT_STATE_SG_NUM + gemma_wgs - 1) // gemma_wgs) * self.sg_sz
        # one block is self.sg_sz
        INPUT_STATE_BLKS_PER_WG = self.input_state_per_wg / self.sg_sz
        INPUT_STATE_BLKS_PER_SG = (INPUT_STATE_BLKS_PER_WG + gemma_sgs_per_wg - 1) // gemma_sgs_per_wg
        input_state_per_sg = INPUT_STATE_BLKS_PER_SG * self.sg_sz
        
        options = f'-DSG_SZ={self.sg_sz} -DK_PER_WG={self.input_state_per_wg} -DSG_NUM={gemma_sgs_per_wg} -DRANK={rank} \
                    -DK_PER_SG={input_state_per_sg} -DGEMMA_USE_SLM={gemma_slm} -DGEMMB_USE_SLM={gemmb_slm} -DPART_NUM={gemma_wgs}'
        self.cl_kernels_opt = kernel_cache(cl_kernel_sources, options)
        self.cl_kernels_ref = kernel_cache(cl_kernel_sources_ref)

    def __call__(self):

        vRANGE = 2
        np.random.seed(0)
        A = np.random.randint(-vRANGE, vRANGE+1, [1, self.input_state]).astype(np.float16)
        weiA = np.random.randint(-vRANGE, vRANGE, [self.input_state, self.rank]).astype(np.float16)
        weiB = np.random.randint(-vRANGE, vRANGE, [self.rank, self.output_state]).astype(np.float16)
        alpha = np.random.rand(1).astype(np.float16)
        tAlpha = cl.tensor(alpha)
        tA = cl.tensor(A)
        tweiA = cl.tensor(weiA)
        tweiB = cl.tensor(weiB)

        tA_output = cl.tensor([self.gemma_wgs, self.rank], np.dtype(np.float16))
        # tA_reduce = cl.tensor([1, self.rank], np.dtype(np.float16))
        tA_output_ref = cl.tensor([1, self.rank], np.dtype(np.float16))
        tB_output =  cl.tensor([1, self.output_state], np.dtype(np.float16))

        if self.use_ref:
            self.cl_kernels_ref.enqueue("gemmA", [self.rank],[self.rank], tA, tweiA, tA_output_ref, tAlpha, self.rank, self.input_state, self.input_state_per_wg)
            REF_LOCAL_SIZE = 128
            assert self.output_state % REF_LOCAL_SIZE == 0, f"'OUTPUT_STATE' {self.output_state} is not multiple of LOCAL_SIZE {REF_LOCAL_SIZE}"
            self.cl_kernels_ref.enqueue("gemmB", [self.output_state],[REF_LOCAL_SIZE], tA_output_ref, tweiB, tB_output, self.output_state, self.rank)
        else:
            cl_kernels = self.cl_kernels_opt
            cl_kernels.enqueue("gemmA", [self.sg_sz * self.gemma_wgs * self.gemma_sgs_per_wg],[self.sg_sz * self.gemma_sgs_per_wg], tA, tweiA, tA_output, self.input_state)
            cl_kernels.enqueue("gemmB", [self.output_state],[self.gemmb_local_sz], tA_output, tweiB, tB_output, tAlpha, self.output_state)
        cl.finish()
        return tB_output
if __name__ == "__main__":
    RANK = 64
    INPUT_STATE = 1536
    OUTPUT_STATE = INPUT_STATE
    WG_NUM = 8
    SG_NUM = 5
    SG_SZ = 16
    ref = LORA(RANK, INPUT_STATE, OUTPUT_STATE, WG_NUM, SG_NUM, False, False, True)
    opt = LORA(RANK, INPUT_STATE, OUTPUT_STATE, WG_NUM, SG_NUM, False, False, False)
    res_ref = ref()
    res_opt = opt()
    compare(res_ref.numpy(), res_opt.numpy())
    print("---------------")

    
