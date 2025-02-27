from . import cl
import numpy as np
from .utils import *
from . import compare
from . import utils

cl_kernel_sources_ref = r'''
    __kernel void gemmA(__global half * A, __global half *B,  __global half *CC, __global half * alpha, int N, int K, int k_blk) {
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
        CC[n_idx] = sum * alpha[n_idx];
    }
    __kernel void reduce(__global half * temp_C, __global half *C, int N, int cnt) {
        int n_idx = get_global_id(0);
        half sum = 0.f;
        for (int i = 0; i < cnt; i++)
            sum += temp_C[i*N + n_idx];
        C[n_idx] = sum;
    }
    
    __kernel void gemmB(__global half * main_input, __global half * A, __global half *B,  __global half *CC, int N, int K) {
        int n_idx = get_global_id(0);
        half sum = 0.f;
        for (int i = 0; i < K; i++)
            sum = fma(A[i], B[i*N+n_idx], sum);
        CC[n_idx] = sum + main_input[n_idx];
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
    __kernel void gemmB(__global half * main_input, __global half * A,  __global half *B,  __global half *CC, __global half * alpha, int N) {
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
            half sum = 0.f;
            __attribute__((opencl_unroll_hint))
            for (int part_idx = 0; part_idx < PART_NUM; part_idx++) {
                half partial_val = as_half(intel_sub_group_block_read_us((const __global ushort*)subA_ptr));
                sum += partial_val;
                subA_ptr += RANK;
            }
            reduce[offset + id_sg_local] += sum;
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
        __global half *C_ptr = CC + sg_idx * SG_SZ;
        __global half *main_input_ptr = main_input + sg_idx * SG_SZ;
        half m_input = as_half(intel_sub_group_block_read_us((const __global ushort*)main_input_ptr));

        half sum = 0;
        for (int kk = 0; kk < RANK; kk += SG_SZ) {
            half scale = as_half(intel_sub_group_block_read_us((const __global ushort*)(alpha + kk)));
#if GEMMB_USE_SLM
            half input = as_half(intel_sub_group_block_read_us((const __local ushort*)(reduce + kk)));
#else
            half input = reduce[id_sg_local + kk];
#endif
            input *= scale;
            __attribute__((opencl_unroll_hint))
            for (int j = 0; j < SG_SZ; j++) {
                half bb = as_half(intel_sub_group_block_read_us((const __global ushort*)B_ptr));
                half aa = as_half(intel_sub_group_broadcast(as_ushort(input), j));
                sum = fma(aa, bb, sum);
                B_ptr += N;
            }
        }
        intel_sub_group_block_write_us((const __global ushort*)C_ptr, as_short(sum+m_input));
    }
    '''


GEMMB_USE_SLM = 0
GEMMA_USE_SLM = 0
GEMMA_WGS = 0
GEMMA_SGS_PER_WG = 0
VERBOSE=False

class LORA:
    def __init__(self, rank, input_state, output_state, gemma_wgs, gemma_sgs_per_wg, use_ref = False):
        self.rank = rank
        self.input_state = input_state
        self.output_state = output_state
        self.gemmb_local_sz = 128
        self.sg_sz = 16
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
                    -DK_PER_SG={input_state_per_sg} -DGEMMA_USE_SLM={GEMMA_USE_SLM} -DGEMMB_USE_SLM={GEMMB_USE_SLM} -DPART_NUM={gemma_wgs}'
        self.cl_kernels_opt = kernel_cache(cl_kernel_sources, options)
        self.cl_kernels_ref = kernel_cache(cl_kernel_sources_ref)

    def __call__(self, mainInput, loraInput, stataA, stateAlpha, stateB):

        tAlpha = cl.tensor(stateAlpha)
        tmainInput = cl.tensor(mainInput)
        tloraInput = cl.tensor(loraInput)
        t_stateA = cl.tensor(stataA)
        t_stateB = cl.tensor(stateB)
        tA_output = cl.tensor([self.gemma_wgs, self.rank], np.dtype(np.float16))
        tRes =  cl.tensor([1, self.output_state], np.dtype(np.float16))

        if self.use_ref:
            tA_output_ref = cl.tensor([1, self.rank], np.dtype(np.float16))
            self.cl_kernels_ref.enqueue("gemmA", [self.rank],[self.rank], tloraInput, t_stateA,
                                        tA_output_ref, tAlpha, self.rank, self.input_state, self.input_state_per_wg)
            REF_LOCAL_SIZE = 128
            assert self.output_state % REF_LOCAL_SIZE == 0, f"'OUTPUT_STATE' {self.output_state} is not multiple of LOCAL_SIZE {REF_LOCAL_SIZE}"
            self.cl_kernels_ref.enqueue("gemmB", [self.output_state],[REF_LOCAL_SIZE],
                                        tmainInput, tA_output_ref, t_stateB, tRes, self.output_state, self.rank)
        else:
            cl_kernels = self.cl_kernels_opt
            cl_kernels.enqueue("gemmA", [self.sg_sz * self.gemma_wgs * self.gemma_sgs_per_wg],[self.sg_sz * self.gemma_sgs_per_wg], 
                                        tloraInput, t_stateA, tA_output, self.input_state)
            cl_kernels.enqueue("gemmB", [self.output_state],[self.gemmb_local_sz], 
                                        tmainInput, tA_output, t_stateB, tRes, tAlpha, self.output_state)
        # cl.finish()
        return tRes

def test(input_state, rank, output_state, check_acc = False, gemma_wgs=GEMMA_WGS, gemma_sgs_per_wg=GEMMA_SGS_PER_WG):
    cl.profiling(True)
    SG_SZ = 16
    vRANGE = 2
    REPEAT = 4
    stateA_list= [np.random.randint(-vRANGE, vRANGE, [input_state, rank]).astype(np.float16)for _ in range(REPEAT)]
    alpha_list = [np.random.rand(rank).astype(np.float16)for _ in range(REPEAT)]
    stateB_list = [np.random.randint(-vRANGE, vRANGE, [rank, output_state]).astype(np.float16)for _ in range(REPEAT)]
    loraInput_list = [np.random.randint(-vRANGE, vRANGE+1, [1, input_state]).astype(np.float16)for _ in range(REPEAT)]
    mainInput_list = [np.random.randint(-vRANGE, vRANGE+1, [1, output_state]).astype(np.float16)for _ in range(REPEAT)]
    ref = LORA(rank, input_state, output_state, GEMMA_WGS, GEMMA_SGS_PER_WG, True)
    opt = LORA(rank, input_state, output_state, GEMMA_WGS, GEMMA_SGS_PER_WG, False)
    if check_acc:
        res_ref = ref(mainInput_list[0], loraInput_list[0], stateA_list[0], alpha_list[0], stateB_list[0])
        res_opt = opt(mainInput_list[0], loraInput_list[0], stateA_list[0], alpha_list[0], stateB_list[0])
        cl.finish()
        compare(res_ref.numpy(), res_opt.numpy())
        print(f'INPUT_STATE:{input_state}, RANK:{rank}, OUPUT_STATE:{output_state} ACC PASS!')
    else:
        for i in range(0, REPEAT):
            opt(mainInput_list[i], loraInput_list[i], stateA_list[i], alpha_list[i], stateB_list[i])
        profiling_data  = cl.finish()
        flops_a = rank*input_state*2
        flops_b = rank*output_state*2
        # how many CPU memory loaded into GPU memory, stateA + duplicate input activation. duplicate would be in GPU global cache?
        rd_bytes_a = (rank*input_state+input_state*GEMMA_WGS)*2
        # gemma output intermedia result should reside in GPU memory in assumption,stateB + alpha + main_input
        rd_bytes_b = (rank*output_state+output_state+rank)*2
        print(f'----------------------------------------------------------------------------------------------------------------------------------')
        print(f'| INPUT_STATE:{input_state}, RANK:{rank}, OUPUT_STATE:{output_state} perf:')
        print(f'----------------------------------------------------------------------------------------------------------------------------------')
        for i in range(1, REPEAT):
            ns_a = profiling_data[i*2]
            ns_b = profiling_data[i*2 + 1]
            if VERBOSE:
                print(f"[GEMMA]: {flops_a*1e-6:.1f} Mflop/{ns_a*1e-3:.1f} us = {flops_a/ns_a*1e-3:.3f} TFlops/s  {rd_bytes_a/1e6:.1f} MB/{ns_a*1e-3:.1f} us = {rd_bytes_a/ns_a:.1f} GB/s ,\
 [GEMMB]: {flops_b*1e-6:.1f} Mflop/{ns_b*1e-3:.1f} us = {flops_b/ns_b*1e-3:.3f} TFlops/s  {rd_bytes_b/1e6:.1f} MB/{ns_b*1e-3:.1f} us = {rd_bytes_b/ns_b:.1f} GB/s ,\
 [total]: {(ns_a+ns_b)*1e-3:.1f}us")
            else:
                print(f'[total]: {(ns_a+ns_b)*1e-3:.1f}us')
        
if __name__ == "__main__":
    GEMMB_USE_SLM = 1
    GEMMA_USE_SLM = 1
    GEMMA_WGS = 8
    GEMMA_SGS_PER_WG = 16
    # VERBOSE=True
    # for rank in [16, 32, 64]:
    for rank in [64]:
        for out_state in [512, 1536, 3840]:
            # test(1536, rank, out_state, True)
            test(1536, rank, out_state)


