from . import cl
import numpy as np
from .utils import *
from . import compare
from . import utils

cl_kernel_sources_ref = r'''
    __kernel void gemmA(__global half * A, __global half *B,  __global half *CC, int N, int K, int k_blk, int M) {
        // Seems kblocks accumulation would introduce error. So ref kernel accumulation should behave same with target.
        int m_idx = get_global_id(0);
        int n_idx = get_global_id(1);
        size_t blk_num = (K + k_blk -1 )/ k_blk;
        half sum = 0.f;
        for (int i = 0; i < blk_num; i++) {
            half sub_sum = 0.f;
            int k_idx = i*k_blk;
            // incase K%k_blk != 0;
            if ((k_idx + k_blk) > K)
                k_blk = K - k_idx;
            for (size_t sub_k = 0; sub_k < k_blk; sub_k++) {
                sub_sum = fma(A[m_idx * K + k_idx], B[k_idx*N+n_idx], sub_sum);
                k_idx++;
            }
            sum += sub_sum;
        }
        //CC[m_idx * N + n_idx] = sum * alpha[n_idx];
        CC[m_idx * N + n_idx] = sum;
    }
    __kernel void reduce(__global half * temp_C, __global half *C, int N, int cnt) {
        int m_idx = get_global_id(0);
        int n_idx = get_global_id(1);
        half sum = 0.f;
        for (int i = 0; i < cnt; i++)
            sum += temp_C[m_idx*cnt*N + i*N + n_idx];
        C[m_idx * N + n_idx] = sum;
    }
    
    __kernel void gemmB(__global half * main_input, __global half * A, __global half *B,  __global half *CC, __global half * alpha, int N, int K, int M) {
        int m_idx = get_global_id(0);
        int n_idx = get_global_id(1);
        half sum = 0.f;
        for (int i = 0; i < K; i++)
            sum = fma(A[m_idx * K + i] * alpha[i], B[i*N+n_idx], sum);
        CC[m_idx * N + n_idx] = sum + main_input[m_idx * N + n_idx];
    }
'''

cl_kernel_sources = r'''
    __attribute__((intel_reqd_sub_group_size(SG_SZ)))
    __kernel void gemmA(__global half * A, __global half *B,  __global half *temp_res, int K, int M) {
        int gid_k =  get_group_id(1);
        int sgid = get_sub_group_id();
        // For 2nd token, rank is small. sg in one wg would be divided by 2 dimensions to increase threads number in wg.
        int sg_num_N = RANK / SG_SZ;
        int sgid_n = sgid % (sg_num_N);
        int sgid_k = sgid / (sg_num_N);

        int lid =  get_sub_group_local_id();
        //How many K is accumulated in the WG.
        int k_start_grp = gid_k * K_PER_WG;
        int k_end_grp = (k_start_grp + K_PER_WG) > K ?  (K): (k_start_grp+K_PER_WG);

        // store each sg accumulation result into SLM. Will sum sg result into wg result. 
        __local half sgs_sum_buff[SG_NUM_K * RANK];
        __local half *sg_buffer_ptr = sgs_sum_buff + sgid_k * RANK;
        __global half *C_ptr = temp_res + RANK * gid_k;

#if GEMMA_USE_SLM
        //Workgroup share same input activation and each subgroup would access all the elements of the shared intput.Put it into SLM.
        __local half local_input[K_PER_WG];
        int local_sz = get_local_size(2);
        //sg could diverge here. not all the sgs can satisfy 'offset < k_len'.
        __attribute__((opencl_unroll_hint))
        for (int offset = sgid * SG_SZ; offset < (k_end_grp - k_start_grp); offset += local_sz) {
            __global half *input_ptr = A + k_start_grp + offset;
            half copy_val = as_half(intel_sub_group_block_read_us((const __global ushort*)input_ptr));
            local_input[offset + lid] = copy_val;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
#endif
        int k_start_sg = k_start_grp + sgid_k * K_PER_K_SG;
        __global half *A_ptr_base = A + k_start_sg;
        __global half *B_ptr_base = B + k_start_sg * RANK;

        //The sg is needs to accumulate. Otherwise, sg not needed. sg_k diverge here.
        if (k_start_sg <  k_end_grp) {
            int klen_sg = ((k_start_sg + K_PER_K_SG) > k_end_grp) ? (k_end_grp-k_start_sg): K_PER_K_SG;
            __global half *A_ptr = A_ptr_base;
            __global half *B_ptr = B_ptr_base + sgid_n * SG_SZ;
            half sum = 0.f;
            for (int kk = 0;  kk < klen_sg; kk += SG_SZ) {
#if GEMMA_USE_SLM
                ushort input = intel_sub_group_block_read_us((const __local ushort*)(local_input + sgid_k * (int)(K_PER_K_SG) + kk));
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
            *(sg_buffer_ptr + sgid_n * SG_SZ + lid) = sum;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        int sg_offset = sgid * SG_SZ;
        //Each SG would try updating data. 
         __attribute__((opencl_unroll_hint))
        for (int base = 0; (sg_offset  + base ) < RANK; base += SG_SZ * SG_NUM_K ) {
            half sum = 0.f;
            __attribute__((opencl_unroll_hint))
            for (int i = 0;  i < SG_NUM_K; i++) {
                sum += as_half(intel_sub_group_block_read_us((const __local ushort*)(sgs_sum_buff + i * RANK + sg_offset)));
            }
            intel_sub_group_block_write_us((const __global ushort*)(C_ptr + sg_offset), as_short(sum));
        }
    }


    __attribute__((intel_reqd_sub_group_size(SG_SZ)))
    __kernel void gemmB(__global half * main_input, __global half * A,  __global half *B,  __global half *CC, __global half * alpha, int N, int M) {
        int wg_id = get_group_id(1);
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
        int local_sz = get_local_size(1);
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
            reduce[offset + id_sg_local] = sum;
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
        if (sg_idx * SG_SZ >= N)
            return;
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
# Global for tests
GEMMB_USE_SLM = 1
GEMMA_USE_SLM = 1
# distribute GEMMA in K dimension into different GEMMA_WGS.
GEMMA_WGS = 8
# distribute GEMMB in N dimension into different GEMMB_WGS.
GEMMB_WGS = 8
VERBOSE=True

class LORA:
    def __init__(self, batch, rank, input_state, output_state, gemma_wgs, gemma_k_sg_num_per_wg, use_ref = False):
        self.rank = rank
        self.batch = batch

        self.input_state = input_state
        self.output_state = output_state
        self.sg_sz = 16
        # GEMMB divided N across WGs.Try to increase the thread occupancy. Min(1024, output_state/GEMMB_WGS)
        # It would be better to ensure wg number is multiple of Xecore number.
        self.gemmb_wg_sz = min((output_state//self.sg_sz)//GEMMB_WGS, 512//self.sg_sz) * self.sg_sz
        self.use_ref = use_ref
        self.gemma_wgs = gemma_wgs
        self.gemma_k_sg_num_per_wg = gemma_k_sg_num_per_wg

        assert self.input_state % self.sg_sz == 0, f"'input state' {self.input_state} is not multiple of SG_SZ {self.sg_sz}"
        assert self.rank % self.sg_sz == 0, f"'RANK' {self.rank} is not multiple of SG_SZ {self.sg_sz}"
        assert self.gemmb_wg_sz % self.sg_sz == 0, f"'gemmb_wg_sz' {self.gemmb_wg_sz} is not multiple of SG_SZ {self.sg_sz}"
        # for the N dimension:
        #                     {rank/sg_size} subgroups are needed.
        # For the K dimension:
        #                   one block is self.sg_sz. Based on gemma_wgs, gemma_k_sg_num_per_wg and input_states, deduce how many input state(K) accumulated per k_sg.
        input_state_blk_num  = self.input_state // self.sg_sz
        self.input_state_per_wg = ((input_state_blk_num + gemma_wgs - 1) // gemma_wgs) * self.sg_sz
        input_state_blks_per_wg = self.input_state_per_wg / self.sg_sz
        input_state_blks_per_k_sg = (input_state_blks_per_wg + gemma_k_sg_num_per_wg - 1) // gemma_k_sg_num_per_wg
        input_state_per_k_sg = input_state_blks_per_k_sg * self.sg_sz
        
        self.m_blk=1
        
        options = f'-DSG_SZ={self.sg_sz} -DK_PER_WG={self.input_state_per_wg} -DSG_NUM_K={gemma_k_sg_num_per_wg} -DRANK={rank} -DM_BLK={self.m_blk}\
                    -DK_PER_K_SG={input_state_per_k_sg} -DGEMMA_USE_SLM={GEMMA_USE_SLM} -DGEMMB_USE_SLM={GEMMB_USE_SLM} -DPART_NUM={gemma_wgs}'
        self.cl_kernels_opt = kernel_cache(cl_kernel_sources, options)
        self.cl_kernels_ref = kernel_cache(cl_kernel_sources_ref)
        if use_ref == False:
            print(f'----------------------------------------------------------------------------------------------------------------------------------')
            print(f'| INPUT_STATE:{input_state}, RANK:{rank}, OUPUT_STATE:{output_state} perf:')
            print(f'| [GEMMA]: GWS:[{batch}, {gemma_wgs}, {self.rank * gemma_k_sg_num_per_wg}], LWS:[1, {self.rank * gemma_k_sg_num_per_wg}] ')
            print(f'| [GEMMB]: GWS:[{batch}, {output_state}], LWS:[1, {self.gemmb_wg_sz}] ')
            print(f'----------------------------------------------------------------------------------------------------------------------------------')

    def __call__(self, mainInput, loraInput, stataA, stateAlpha, stateB, Aoutput, result):
        tA_reduce_ref = cl.tensor([self.batch, self.rank], np.dtype(np.float16))
        if self.use_ref:
            tA_output_ref = cl.tensor([self.batch, self.rank], np.dtype(np.float16))
            self.cl_kernels_ref.enqueue("gemmA", [self.batch, self.rank],[1, self.rank], loraInput, stataA,
                                        tA_output_ref, self.rank, self.input_state, self.input_state_per_wg, self.batch)
            REF_LOCAL_SIZE = 16
            assert self.output_state % REF_LOCAL_SIZE == 0, f"'OUTPUT_STATE' {self.output_state} is not multiple of LOCAL_SIZE {REF_LOCAL_SIZE}"
            self.cl_kernels_ref.enqueue("gemmB", [self.batch, self.output_state],[1, REF_LOCAL_SIZE],
                                        mainInput, tA_output_ref, stateB, result, stateAlpha, self.output_state, self.rank, self.batch)
            return [result, tA_output_ref]
        else:
            cl_kernels = self.cl_kernels_opt
            # GEMMA: ONE WG would has {self.gemma_k_sg_num_per_wg*self.rank/SG_SZ}subgroups. self.rank/SG_SZ subgroups on N dimension, self.gemma_k_sg_num_per_wg on K dimension
            # Total {self.gemma_wg} WGS
            cl_kernels.enqueue("gemmA", [self.batch, self.gemma_wgs, self.gemma_k_sg_num_per_wg * self.rank],[1, 1, self.gemma_k_sg_num_per_wg * self.rank], 
                                        loraInput, stataA, Aoutput, self.input_state, self.batch)
            self.cl_kernels_ref.enqueue("reduce", [self.batch, self.rank],[1, self.rank], Aoutput, tA_reduce_ref, self.rank, self.gemma_wgs)

            # GEMMB: ONE WG would has {gemmb_wg_sz/SG_SZ}subgroups.
            # Total {output_state/gemmb_wg_sz} WGS
            cl_kernels.enqueue("gemmB", [self.batch, (self.output_state + self.gemmb_wg_sz - 1)//self.gemmb_wg_sz * self.gemmb_wg_sz],[1, self.gemmb_wg_sz], 
                                        mainInput, Aoutput, stateB, result, stateAlpha, self.output_state, self.batch)
        # cl.finish()
            return [result, tA_reduce_ref]

def test(batch, input_state, rank, output_state, check_acc = False):
    cl.profiling(True)
    SG_SZ = 16
    vRANGE = 1
    REPEAT = 20
    gemma_k_sg_num_per_wg = min(16, 1024//rank)
    gemma_wgs = GEMMA_WGS
    stateA_list= [cl.tensor(np.random.randint(-vRANGE, vRANGE+1, [input_state, rank]).astype(np.float16))for _ in range(REPEAT)]
    alpha_list_np = [np.random.randint(-1, 2, [rank]).astype(np.float16) for _ in range(REPEAT)]
    alpha_list = [cl.tensor(np.multiply(alpha, 0.5)) for alpha in alpha_list_np]
    stateB_list = [cl.tensor(np.random.randint(-vRANGE, vRANGE+1, [rank, output_state]).astype(np.float16))for _ in range(REPEAT)]
    loraInput_list = [cl.tensor(np.random.randint(-vRANGE, vRANGE+1, [batch, input_state]).astype(np.float16))for _ in range(REPEAT)]
    mainInput_list = [cl.tensor(np.random.randint(-vRANGE, vRANGE+1, [batch, output_state]).astype(np.float16))for _ in range(REPEAT)]
    #Must set the output to be zeros to avoid not all the data is updated in GEMMA. 
    A_output_list = [cl.tensor(np.zeros([batch, gemma_wgs, rank]).astype(np.float16))for _ in range(REPEAT)]
    res_list = [cl.tensor([batch, output_state], np.dtype(np.float16))for _ in range(REPEAT)]
    # Invalid GPU cache.Ensure data is not in cache. Cache size 4MB.
    tempbuffer =  cl.tensor(np.zeros([8*1024*1024]).astype(np.float16))
    cl.finish()
    ref = LORA(batch, rank, input_state, output_state, gemma_wgs, gemma_k_sg_num_per_wg, True)
    opt = LORA(batch, rank, input_state, output_state, gemma_wgs, gemma_k_sg_num_per_wg, False)

    if check_acc:
        res_opt = opt(mainInput_list[0], loraInput_list[0], stateA_list[0], alpha_list[0], stateB_list[0], A_output_list[0], res_list[0])
        res_ref = ref(mainInput_list[0], loraInput_list[0], stateA_list[0], alpha_list[0], stateB_list[0], A_output_list[0], res_list[0])
        cl.finish()
        if 0:
            ouputA_N = np.matmul(loraInput_list[0].numpy(), stateA_list[0].numpy())
            compare(ouputA_N, res_ref[1].numpy())
            ouputA_N = np.multiply(ouputA_N, alpha_list[0].numpy())
            outputB_N = np.matmul(ouputA_N, stateB_list[0].numpy())
            outputB_N = np.add(outputB_N, mainInput_list[0].numpy())
            compare(res_ref[0].numpy(), outputB_N)
        # compare(res_ref[0].numpy(), res_opt[0].numpy())
        # compare(res_ref[1].numpy(), res_opt[1].numpy())

        print(f'INPUT_STATE:{input_state}, RANK:{rank}, OUPUT_STATE:{output_state} ACC PASS!')
    else:
        for i in range(0, REPEAT):
            opt(mainInput_list[i], loraInput_list[i], stateA_list[i], alpha_list[i], stateB_list[i],  A_output_list[i], res_list[i])
        profiling_data  = cl.finish()
        flops_a = rank*input_state*2
        flops_b = rank*output_state*2
        # how many CPU memory loaded into GPU memory, stateA + input activation. duplicate input activation would be in GPU global cache?
        rd_bytes_a = (rank*input_state+input_state)*2
        # gemma output intermedia result should reside in GPU memory in assumption,stateB + alpha + main_input + output
        rd_bytes_b = (rank*output_state+output_state*2+rank)*2

        for i in range(1, REPEAT):
            # ns_a = profiling_data[i*2]
            # ns_b = profiling_data[i*2 + 1]
            ns_a = profiling_data[i*3]
            ns_b = profiling_data[i*3 + 2]
            if VERBOSE:
                print(f"[GEMMA]: {flops_a*1e-6:.1f} Mflop/{ns_a*1e-3:.1f} us = {flops_a/ns_a:.3f} GFlops/s  {rd_bytes_a/1e6:.1f} MB/{ns_a*1e-3:.1f} us = {rd_bytes_a/ns_a:.1f} GB/s ,\
 [GEMMB]: {flops_b*1e-6:.1f} Mflop/{ns_b*1e-3:.1f} us = {flops_b/ns_b:.3f} GFlops/s  {rd_bytes_b/1e6:.1f} MB/{ns_b*1e-3:.1f} us = {rd_bytes_b/ns_b:.1f} GB/s ,\
 [total]: {(ns_a+ns_b)*1e-3:.1f}us")
            else:
                print(f'[latency]: {ns_a*1e-3:.1f} + {ns_b*1e-3:.1f} = {(ns_a+ns_b)*1e-3:.1f}us')
        
if __name__ == "__main__":
    if 1:
        for batch in (5, 19, 255):
            test(batch, 1536, 64, 128, True)
        # for rank in range(16//16, 256//16):
        #     for out_state in range(512//16, 3840//16):
        #         for input_state in (1024, 1536, 2048, 2560, 3072):
        #             test(4, input_state, rank*16, out_state*16, True)
    else:
        for rank in (16, 32, 64):
            for out_state in (1536, 512):
                for input_state in (1536, 1024):
                    test(20, input_state, rank, out_state, True)
        # for rank in [64]:
        #     for out_state in [512, 1536, 3840]:
        #         test(1, 1536, rank, out_state)
    # for rank in [1024]:
    #     for out_state in [4096*2]:
    #     # for out_state in [512]:
    #         test(1536, rank, out_state)
