from . import cl
import numpy as np
from .utils import *
from . import compare
from . import utils


MAX_GEMMA_SGK = 32
VERBOSE=True

"""
------------------------------------------------------------------------------------------------------------------
ref kernel
------------------------------------------------------------------------------------------------------------------
"""

cl_kernel_sources_ref = r'''
    __kernel void gemmA(__global float * A, __global float *B,  __global float *CC, int N, int K, int k_blk, int M) {
        // Seems kblocks accumulation would introduce error. So ref kernel accumulation should behave same with target.
        int m_idx = get_global_id(0);
        int n_idx = get_global_id(1);
        size_t blk_num = (K + k_blk -1 )/ k_blk;
        float sum = 0.f;
        for (int i = 0; i < blk_num; i++) {
            float sub_sum = 0.f;
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
    __kernel void reduce(__global float * temp_C, __global float *C, int N, int cnt) {
        int m_idx = get_global_id(0);
        int n_idx = get_global_id(1);
        float sum = 0.f;
        for (int i = 0; i < cnt; i++)
            sum += temp_C[m_idx*cnt*N + i*N + n_idx];
        C[m_idx * N + n_idx] = sum;
    }
    
    __kernel void gemmB(__global float * main_input, __global float * A, __global float *B,  __global float *CC, __global float * alpha, int N, int K, int M) {
        int m_idx = get_global_id(0);
        int n_idx = get_global_id(1);
        float sum = 0.f;
        for (int i = 0; i < K; i++)
            sum = fma(A[m_idx * K + i] * alpha[i], B[i*N+n_idx], sum);
        CC[m_idx * N + n_idx] = sum + main_input[m_idx * N + n_idx];
    }
'''

"""
------------------------------------------------------------------------------------------------------------------
opt lora 2nd token
------------------------------------------------------------------------------------------------------------------
"""
opt_lora_kernel_2nd = r'''
#define MAX_GEMMA_SGK (''' + f'{MAX_GEMMA_SGK}' + r''') 
#define MAX_LORA_RANK 256
#define MAX_GEMMA_SG_BK 64
    __attribute__((intel_reqd_sub_group_size(SG_SZ)))
    __kernel void gemmA(__global float * A, __global float *B,  __global float *C, int K) {
        int gid =  get_group_id(0);
        int sgid = get_sub_group_id();
        // For 2nd token, rank is small. sg in one wg would be divided by 2 dimensions to increase threads number in wg.
        int sgN = RANK / SG_SZ;
        int sgid_k = sgid / sgN;
        int n_idx = sgid % sgN * SG_SZ;
        int lid =  get_sub_group_local_id();
        //How many K is accumulated in the WG.
        int bk_wg = GEMMA_SGK * GEMMA_SG_BK;
        int k_start_wg = gid * bk_wg;
        int wg_k_len = (k_start_wg + bk_wg) > K ?  (K - k_start_wg) : bk_wg;
        int sgK = (wg_k_len + GEMMA_SG_BK - 1) / GEMMA_SG_BK;

        // store each sg accumulation result into SLM. Will reduce sg result into wg result. 
        __local float fma_buff[MAX_GEMMA_SGK * MAX_LORA_RANK];
        __local float *sg_fma_buff = fma_buff + sgid_k * MAX_LORA_RANK;

        //put all need input activation into SLM. 'sgN' sgs would share same input.
        __local float local_input[MAX_GEMMA_SGK * MAX_GEMMA_SG_BK];
        int local_sz = get_num_sub_groups() * SG_SZ;
        //sg could diverge here. not all the sgs can satisfy 'offset < k_len'.
        for (int offset = sgid * SG_SZ; offset < wg_k_len; offset += local_sz) {
            __global float *input_ptr = A + k_start_wg + offset;
            float copy_val = as_float(intel_sub_group_block_read((const __global uint*)input_ptr));
            local_input[offset + lid] = copy_val;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        int k_offset = sgid_k * GEMMA_SG_BK;
        int k_idx = k_start_wg + k_offset;

        //The sg is needs to accumulate. Otherwise, sg not needed. sg_k diverge here.
        if (sgid_k * GEMMA_SG_BK  <  wg_k_len) {
            int klen_sg =  (k_offset + GEMMA_SG_BK) > wg_k_len ? (wg_k_len - k_offset) : GEMMA_SG_BK;
            __global float *B_ptr = B + k_idx * RANK + n_idx;
            __local float *A_ptr = local_input + k_offset;
            float sum = 0.f;
            for (int kk = 0;  kk < klen_sg; kk += SG_SZ) {
                uint input = intel_sub_group_block_read((const __local uint*)(A_ptr + kk));
                __attribute__((opencl_unroll_hint))
                for (int j = 0; j < SG_SZ; j++) {
                    float bb = as_float(intel_sub_group_block_read((const __global uint*)(B_ptr)));
                    float aa = as_float(sub_group_broadcast(input, j));
                    sum = fma(aa, bb, sum);
                    B_ptr += RANK;
                }
            }
            *(sg_fma_buff + n_idx + lid) = sum;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        __global float *C_ptr = C + RANK * gid;
        //only need sg on N dimenion to update data.
        if (sgid_k != 0)
            return;
        float sum = 0.f;
        if (sgK == GEMMA_SGK) {
            //unroll
            __attribute__((opencl_unroll_hint))
            for (int i = 0;  i < GEMMA_SGK; i++) {
                sum += as_float(intel_sub_group_block_read((const __local uint*)(fma_buff + i * MAX_LORA_RANK + n_idx)));
            }
        } else {
            //can't unroll, tail handling
            for (int i = 0;  i < sgK; i++)
            sum += as_float(intel_sub_group_block_read((const __local uint*)(fma_buff + i * MAX_LORA_RANK + n_idx)));
        }
        intel_sub_group_block_write((const __global uint*)(C_ptr + n_idx), as_int(sum));
    }

    __attribute__((intel_reqd_sub_group_size(SG_SZ)))
    __kernel void gemmB(__global float * main_input, __global float * A,  __global float *B,  __global float *C, __global float * alpha, int N) {
        int wg_id = get_group_id(0);
        int sg_id = get_sub_group_id();
        int sg_num = get_num_sub_groups();
        int n_idx = (wg_id * sg_num  +  sg_id) * SG_SZ;
        int id_sg_local = get_sub_group_local_id();
        __local float reduce[MAX_LORA_RANK];

        __global float *B_ptr = B + n_idx;

        //1. Reduce
        //EACH WG would reduce input activation and save into local memory `reduce[RANK]`.
        int local_sz = get_local_size(0);
        for (int offset = sg_id * SG_SZ; offset < RANK; offset += local_sz) {
            __global float *A_ptr = A + offset;
            float sum = 0.f;
            __attribute__((opencl_unroll_hint))
            for (int part_idx = 0; part_idx < GEMMB_PART_NUM; part_idx++) {
                float partial_val = as_float(intel_sub_group_block_read((const __global uint*)A_ptr));
                sum += partial_val;
                A_ptr += RANK;
            }
            reduce[offset + id_sg_local] = sum;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (n_idx >= N)
            return;
        //2.  GEMMB
        __global float *C_ptr = C + n_idx;

        float sum = 0;
        __attribute__((opencl_unroll_hint))
        for (int kk = 0; kk < RANK; kk += SG_SZ) {
            float scale = as_float(intel_sub_group_block_read((const __global uint*)(alpha + kk)));
            float input = as_float(intel_sub_group_block_read((const __local uint*)(reduce + kk)));

            input *= scale;
            __attribute__((opencl_unroll_hint))
            for (int j = 0; j < SG_SZ; j++) {
                float bb = as_float(intel_sub_group_block_read((const __global uint*)B_ptr));
                float aa = as_float(sub_group_broadcast(as_uint(input), j));
                sum = fma(aa, bb, sum);
                B_ptr += N;
            }
        }
        __global float *main_input_ptr = main_input + n_idx;
        float m_input = as_float(intel_sub_group_block_read((const __global uint*)main_input_ptr));
        intel_sub_group_block_write((const __global uint*)C_ptr, as_int(sum+m_input));
    }
    '''


def ALIGN_UP(a, b):
    return ((a + (b -1)) // b *b)
def DIV_UP(a, b):
    return ((a + (b -1)) // b)

# GEMMA has a big K but small N(rank), GEMMA kernel  would divide K by WGs and K diemsnion SGs.
# GEMMB only divide N by WGs and SGs because N is big and K(Rank) is small..
# gemma_sg_BK: the Number of K accumuated in one sg for GEMMA.
# gemma_sgK: the number of sg in K dimension for GEMMA.
# gemmb_sgN: the number of sg in N dimension for GEMMB
class LORA_2ND:
    def __init__(self, rank, input_state, output_state, gemma_sg_BK, gemma_sgK, gemmb_sgN, use_ref = False):
        self.rank = rank
        self.input_state = input_state
        self.output_state = output_state
        self.sg_sz = 8
        assert gemmb_sgN <=64, f'gemmb_sgN:{gemmb_sgN} bigger than 64 limitation'
        self.gemmb_wg_sz = gemmb_sgN * self.sg_sz
        self.use_ref = use_ref
        # WGs would divide K dimension. N would not divided by WGs
        self.gemma_wgs = DIV_UP(input_state, gemma_sg_BK *gemma_sgK)
        self.gemma_sgK = gemma_sgK

        assert self.input_state % self.sg_sz == 0, f"'input state' {self.input_state} is not multiple of SG_SZ {self.sg_sz}"
        assert self.rank % self.sg_sz == 0, f"'RANK' {self.rank} is not multiple of SG_SZ {self.sg_sz}"
        assert gemma_sg_BK % self.sg_sz == 0, f"'gemma_sg_BK' { gemma_sg_BK} is not multiple of SG_SZ {self.sg_sz}"
        assert self.gemmb_wg_sz % self.sg_sz == 0, f"'gemmb_wg_sz' {self.gemmb_wg_sz} is not multiple of SG_SZ {self.sg_sz}"
        self.gemma_wg_BK = gemma_sg_BK * gemma_sgK
        
        options = f'-DSG_SZ={self.sg_sz}  -DGEMMA_SGK={gemma_sgK} -DRANK={rank} -DGEMMA_SG_BK={gemma_sg_BK} -DGEMMB_PART_NUM={self.gemma_wgs}'
        self.cl_kernels_opt = kernel_cache(opt_lora_kernel_2nd, options)
        self.cl_kernels_ref = kernel_cache(cl_kernel_sources_ref)
        self.gemma_lws = [1 , gemma_sgK, self.rank]
        self.gemma_gws = [self.gemma_wgs, gemma_sgK, self.rank]
        self.gemmb_lws = [self.gemmb_wg_sz]
        self.gemmb_gws = [ALIGN_UP(self.output_state, self.gemmb_wg_sz)]
        if use_ref == False:
            print(f'----------------------------------------------------------------------------------------------------------------------------------')
            print(f'| BATCH = 1 INPUT_STATE:{input_state}, RANK:{rank}, OUPUT_STATE:{output_state}:')
            print(f'| [2ND_GEMMA]: GWS:{self.gemma_gws}, LWS:{self.gemma_lws} SG_BK:{gemma_sg_BK} SGK:{gemma_sgK}')
            print(f'| [2ND_GEMMB]: GWS:{self.gemmb_lws}, LWS:{self.gemmb_lws} SGN:{gemmb_sgN}')
            print(f'----------------------------------------------------------------------------------------------------------------------------------')

    def __call__(self, mainInput, loraInput, stataA, stateAlpha, stateB, Aoutput, result):

        if self.use_ref:
            tA_output_ref = cl.tensor([1, self.rank], np.dtype(np.float32))
            
            # self.cl_kernels_ref.enqueue("gemmA", [self.batch, self.rank],[1, self.rank], loraInput, stataA,
            #                             tA_output_ref, self.rank, self.input_state, self.input_state_per_wg, self.batch)
            
                        
            self.cl_kernels_ref.enqueue("gemmA", [1, self.rank],[1, self.rank], loraInput, stataA,
                                        tA_output_ref, self.rank, self.input_state, self.gemma_wg_BK, 1)
            REF_LOCAL_SIZE = 8
            assert self.output_state % REF_LOCAL_SIZE == 0, f"'OUTPUT_STATE' {self.output_state} is not multiple of LOCAL_SIZE {REF_LOCAL_SIZE}"
            
            # self.cl_kernels_ref.enqueue("gemmB", [self.batch, self.output_state],[1, REF_LOCAL_SIZE],
            #                             mainInput, tA_output_ref, stateB, result, stateAlpha, self.output_state, self.rank, self.batch)
            self.cl_kernels_ref.enqueue("gemmB", [1, self.output_state],[1, REF_LOCAL_SIZE],
                                        mainInput, tA_output_ref, stateB, result, stateAlpha, self.output_state, self.rank, 1)
        else:
            cl_kernels = self.cl_kernels_opt
            # GEMMA: ONE WG would has {self.gemma_sgK*self.rank/SG_SZ}subgroups. self.rank/SG_SZ subgroups on N dimension, self.gemma_sgK on K dimension
            # Total {self.gemma_wg} WGS
            cl_kernels.enqueue("gemmA", self.gemma_gws, self.gemma_lws, 
                                        loraInput, stataA, Aoutput, self.input_state)
            # GEMMB: ONE WG would has {gemmb_wg_sz/SG_SZ}subgroups.
            # Total {output_state/gemmb_wg_sz} WGS
            cl_kernels.enqueue("gemmB", self.gemmb_gws, self.gemmb_lws, 
                                        mainInput, Aoutput, stateB, result, stateAlpha, self.output_state)
        # cl.finish()
        return result


"""
------------------------------------------------------------------------------------------------------------------
opt lora 1st token
------------------------------------------------------------------------------------------------------------------
"""

def generate_store_C(regM, regN, withscale, withSum):
    src = ""
    if withscale:
        # read all needed scale.
        src += r'''
        __global float *alpha_ptr = alpha + n_idx;
        ''' + "\n\t".join([f'float alpha_{n} = as_float(intel_sub_group_block_read((const __global uint*)(alpha_ptr + SG_SZ * {n})));' for n in range(regN)]) + r'''
        '''
    elif withSum:
        src += r'''
        __global float *main_ptr = mainInput + m_idx * N + n_idx;
        ''' + "\n\t".join([f'float main_N{n} = 0;' for n in range(regN)]) + r'''
        '''

    for m in range(regM):
        # read out `regN` mainInput
        if withSum:
            src += "\n\t".join([f'main_N{n} = as_float(intel_sub_group_block_read((const __global uint*)(main_ptr + SG_SZ * {n})));' for n in range(regN)]) + r'''
                    ''' 
        for n in range(regN):
            if withscale:
                src +=  f"\n\tintel_sub_group_block_write((__global uint*)(ptrC + SG_SZ * {n}), as_int(sum{m}_{n}*alpha_{n}));"
            elif withSum:
                src +=  f"\n\tintel_sub_group_block_write((__global uint*)(ptrC + SG_SZ * {n}), as_int(sum{m}_{n}+main_N{n}));"
            else:
                src +=  f"\n\tintel_sub_group_block_write((__global uint*)(ptrC + SG_SZ * {n}), as_int(sum{m}_{n}));"
        if withSum:
            src += f"\n\tmain_ptr+= N;"
        src += f"\n\tptrC += N;\n\n"

    return src
# This function would generate 1st token kernel based on 'regM, regN, withscale, withSum'.
# regM, regN would be the register  blocking.
# withscale = true for LORA GEMMA kernel to genereate multiply post ops
# withsum =  true for LORA GEMMB kernel to genereate add post ops
def generate_gemm_src(regM, regN, withscale = False, withSum=False):
    assert (withscale and withSum) == False, f"not support both LORA alpha and sum in one gemm kernel"
    if withscale:
        func = f'gemmA_rM{regM}_rN{regN}'
    elif withSum:
        func = f'gemmB_rM{regM}_rN{regN}'
    else:
        func = f'gemm_rM{regM}_rN{regN}'

    src_gemma =  r'''
    __attribute__((intel_reqd_sub_group_size(SG_SZ)))
    __kernel void 
    gemmA_rM4_rN1(__global float * lora_input, __global float *state_a,  __global float *output_a, __global float *state_alpha,  __global float *mainInput, int M, int N, int K) {

#define SUBGROUP_SIZE 8
#define A_REG_N 1
#define A_REG_M 4

#define INPUT1_TYPE_SIZE 4
#define STATE_TYPE_SIZE 4
#define ACCUMULATOR_TYPE_SIZE 4

#define ACCUMULATOR_TYPE float
#define INPUT1_TYPE float
#define STATE_TYPE float
#define AS_ACCUMULATOR_TYPE as_float
#define LORA_RANK 16

    int sgid = get_sub_group_id();
    int sgN = get_local_size(1) / SUBGROUP_SIZE;
    int sgM = get_local_size(0);
    int sgid_N = sgid % sgN;
    int sgid_M = sgid / sgN;
    int BM = A_REG_M * sgM;
    int BN = get_local_size(1) * A_REG_N;
    int m_idx = get_group_id(0) * BM  + sgid_M * A_REG_M;
    int n_idx = get_group_id(1) * BN  + sgid_N * SUBGROUP_SIZE * A_REG_N;

    // printf("M: %d, N: %d, K: %d", M, N, K);

    if (m_idx >= M || n_idx >= LORA_RANK)
        return;

    if (m_idx + A_REG_M > M)
        m_idx = M - A_REG_M;

    if (n_idx + A_REG_N * SUBGROUP_SIZE > LORA_RANK)
        n_idx = LORA_RANK - A_REG_N * SUBGROUP_SIZE;

    __global INPUT1_TYPE* ptrA = lora_input + m_idx * K;
    __global STATE_TYPE* ptrB = state_a + n_idx;
    __global ACCUMULATOR_TYPE* ptrC = output_a + m_idx * LORA_RANK + n_idx;


    ACCUMULATOR_TYPE sum_0_0 = 0;
    ACCUMULATOR_TYPE sum_1_0 = 0;
    ACCUMULATOR_TYPE sum_2_0 = 0;
    ACCUMULATOR_TYPE sum_3_0 = 0;

    for (int i = 0; i < K; i += SUBGROUP_SIZE) {


#if INPUT1_TYPE_SIZE == 4
            uint input_0 = intel_sub_group_block_read((const __global uint*)(ptrA + 0 * K));
            uint input_1 = intel_sub_group_block_read((const __global uint*)(ptrA + 1 * K));
            uint input_2 = intel_sub_group_block_read((const __global uint*)(ptrA + 2 * K));
            uint input_3 = intel_sub_group_block_read((const __global uint*)(ptrA + 3 * K));
#else
            ushort input_0 = intel_sub_group_block_read_us((const __global ushort*)(ptrA + 0 * K));
            ushort input_1 = intel_sub_group_block_read_us((const __global ushort*)(ptrA + 1 * K));
            ushort input_2 = intel_sub_group_block_read_us((const __global ushort*)(ptrA + 2 * K));
            ushort input_3 = intel_sub_group_block_read_us((const __global ushort*)(ptrA + 3 * K));
#endif

            //__attribute__((opencl_unroll_hint))
            for (int kk = 0; kk < SUBGROUP_SIZE; kk++) {

#if STATE_TYPE_SIZE == 4
                ACCUMULATOR_TYPE bb_0 = AS_ACCUMULATOR_TYPE(intel_sub_group_block_read((const __global uint*)(ptrB + 0 * SUBGROUP_SIZE)));
#else
                ACCUMULATOR_TYPE bb_0 = AS_ACCUMULATOR_TYPE(intel_sub_group_block_read_us((const __global ushort*)(ptrB + 0 * SUBGROUP_SIZE)));
#endif

#if INPUT1_TYPE_SIZE == 4
                ACCUMULATOR_TYPE aa_0 = AS_ACCUMULATOR_TYPE(sub_group_broadcast(input_0, kk));
                ACCUMULATOR_TYPE aa_1 = AS_ACCUMULATOR_TYPE(sub_group_broadcast(input_1, kk));
                ACCUMULATOR_TYPE aa_2 = AS_ACCUMULATOR_TYPE(sub_group_broadcast(input_2, kk));
                ACCUMULATOR_TYPE aa_3 = AS_ACCUMULATOR_TYPE(sub_group_broadcast(input_3, kk));
#else
                ACCUMULATOR_TYPE aa_0 = AS_ACCUMULATOR_TYPE(intel_sub_group_broadcast(input_0, kk));
                ACCUMULATOR_TYPE aa_1 = AS_ACCUMULATOR_TYPE(intel_sub_group_broadcast(input_1, kk));
                ACCUMULATOR_TYPE aa_2 = AS_ACCUMULATOR_TYPE(intel_sub_group_broadcast(input_2, kk));
                ACCUMULATOR_TYPE aa_3 = AS_ACCUMULATOR_TYPE(intel_sub_group_broadcast(input_3, kk));
#endif


                sum_0_0 = fma(aa_0, bb_0, sum_0_0);
                sum_1_0 = fma(aa_1, bb_0, sum_1_0);
                sum_2_0 = fma(aa_2, bb_0, sum_2_0);
                sum_3_0 = fma(aa_3, bb_0, sum_3_0);

                ptrB += LORA_RANK;
            }
            ptrA += SUBGROUP_SIZE;
    }


    __global STATE_TYPE *alpha_ptr = state_alpha + n_idx;


#if STATE_TYPE_SIZE == 4
    // TO_ACCUMULATOR_TYPE(AS_STATE_TYPE(..)) - right version
    ACCUMULATOR_TYPE alpha_0 = AS_ACCUMULATOR_TYPE(intel_sub_group_block_read((const __global uint*)(alpha_ptr + SUBGROUP_SIZE * 0)));
#else
    ACCUMULATOR_TYPE alpha_0 = AS_ACCUMULATOR_TYPE(intel_sub_group_block_read_us((const __global ushort*)(alpha_ptr + SUBGROUP_SIZE * 0)));
#endif


#if ACCUMULATOR_TYPE_SIZE == 4
    intel_sub_group_block_write((__global uint*)(ptrC + SUBGROUP_SIZE * 0), as_int(sum_0_0 * alpha_0));
    ptrC += LORA_RANK;
    intel_sub_group_block_write((__global uint*)(ptrC + SUBGROUP_SIZE * 0), as_int(sum_1_0 * alpha_0));
    ptrC += LORA_RANK;
    intel_sub_group_block_write((__global uint*)(ptrC + SUBGROUP_SIZE * 0), as_int(sum_2_0 * alpha_0));
    ptrC += LORA_RANK;
    intel_sub_group_block_write((__global uint*)(ptrC + SUBGROUP_SIZE * 0), as_int(sum_3_0 * alpha_0));
    ptrC += LORA_RANK;
#else
    intel_sub_group_block_write_us((__global ushort*)(ptrC + SUBGROUP_SIZE * 0), as_short(sum_0_0 * alpha_0));
    ptrC += LORA_RANK;
    intel_sub_group_block_write_us((__global ushort*)(ptrC + SUBGROUP_SIZE * 0), as_short(sum_1_0 * alpha_0));
    ptrC += LORA_RANK;
    intel_sub_group_block_write_us((__global ushort*)(ptrC + SUBGROUP_SIZE * 0), as_short(sum_2_0 * alpha_0));
    ptrC += LORA_RANK;
    intel_sub_group_block_write_us((__global ushort*)(ptrC + SUBGROUP_SIZE * 0), as_short(sum_3_0 * alpha_0));
    ptrC += LORA_RANK;
#endif
}
'''
    src_gemmb =  r'''
    __attribute__((intel_reqd_sub_group_size(SG_SZ)))
    __kernel void 
    gemmB_rM8_rN2(__global float * output_a, __global float *state_b,  __global float *output, __global float *alpha,  __global float *main_input, int M, int N, int K) {
#define SUBGROUP_SIZE 8
#define REG_N 2
#define REG_M 8

#define INPUT1_TYPE_SIZE 4
#define STATE_TYPE_SIZE 4
#define ACCUMULATOR_TYPE_SIZE 4

#define ACCUMULATOR_TYPE float
#define INPUT1_TYPE float
#define STATE_TYPE float
#define LORA_RANK 16
#define OUTPUT_TYPE float
#define INPUT0_TYPE float

#define AS_ACCUMULATOR_TYPE as_float
#define AS_INPUT0_TYPE as_float
#define TO_INPUT0_TYPE (float)
    int sgid = get_sub_group_id();
    int sgN = get_local_size(1) / SUBGROUP_SIZE;
    int sgM = get_local_size(0);
    int sgid_N = sgid % sgN;
    int sgid_M = sgid / sgN;
    int BM = REG_M * sgM;
    int BN = get_local_size(1) * REG_N;
    int m_idx = get_group_id(0) * BM  + sgid_M * REG_M;
    int n_idx = get_group_id(1) * BN  + sgid_N * SUBGROUP_SIZE * REG_N;

    // printf("M: %d, N: %d, K: %d", M, N, K);

    if (m_idx >= M || n_idx >= N)
        return;

    if (m_idx + REG_M > M)
        m_idx = M - REG_M;

    if (n_idx + REG_N * SUBGROUP_SIZE > N)
        n_idx = N - REG_N * SUBGROUP_SIZE;

    __global ACCUMULATOR_TYPE* ptrA = output_a + m_idx * LORA_RANK;
    __global STATE_TYPE* ptrB = state_b + n_idx;
    __global OUTPUT_TYPE* ptrC = output + m_idx * N + n_idx;


    ACCUMULATOR_TYPE sum_0_0 = 0;
    ACCUMULATOR_TYPE sum_0_1 = 0;
    ACCUMULATOR_TYPE sum_1_0 = 0;
    ACCUMULATOR_TYPE sum_1_1 = 0;
    ACCUMULATOR_TYPE sum_2_0 = 0;
    ACCUMULATOR_TYPE sum_2_1 = 0;
    ACCUMULATOR_TYPE sum_3_0 = 0;
    ACCUMULATOR_TYPE sum_3_1 = 0;
    ACCUMULATOR_TYPE sum_4_0 = 0;
    ACCUMULATOR_TYPE sum_4_1 = 0;
    ACCUMULATOR_TYPE sum_5_0 = 0;
    ACCUMULATOR_TYPE sum_5_1 = 0;
    ACCUMULATOR_TYPE sum_6_0 = 0;
    ACCUMULATOR_TYPE sum_6_1 = 0;
    ACCUMULATOR_TYPE sum_7_0 = 0;
    ACCUMULATOR_TYPE sum_7_1 = 0;

    for (int i = 0; i < LORA_RANK; i += SUBGROUP_SIZE) {


#if INPUT1_TYPE_SIZE == 4
        uint input_0 = intel_sub_group_block_read((const __global uint*)(ptrA + 0 * LORA_RANK));
        uint input_1 = intel_sub_group_block_read((const __global uint*)(ptrA + 1 * LORA_RANK));
        uint input_2 = intel_sub_group_block_read((const __global uint*)(ptrA + 2 * LORA_RANK));
        uint input_3 = intel_sub_group_block_read((const __global uint*)(ptrA + 3 * LORA_RANK));
        uint input_4 = intel_sub_group_block_read((const __global uint*)(ptrA + 4 * LORA_RANK));
        uint input_5 = intel_sub_group_block_read((const __global uint*)(ptrA + 5 * LORA_RANK));
        uint input_6 = intel_sub_group_block_read((const __global uint*)(ptrA + 6 * LORA_RANK));
        uint input_7 = intel_sub_group_block_read((const __global uint*)(ptrA + 7 * LORA_RANK));
#else
        ushort input_0 = intel_sub_group_block_read_us((const __global ushort*)(ptrA + 0 * LORA_RANK));
        ushort input_1 = intel_sub_group_block_read_us((const __global ushort*)(ptrA + 1 * LORA_RANK));
        ushort input_2 = intel_sub_group_block_read_us((const __global ushort*)(ptrA + 2 * LORA_RANK));
        ushort input_3 = intel_sub_group_block_read_us((const __global ushort*)(ptrA + 3 * LORA_RANK));
        ushort input_4 = intel_sub_group_block_read_us((const __global ushort*)(ptrA + 4 * LORA_RANK));
        ushort input_5 = intel_sub_group_block_read_us((const __global ushort*)(ptrA + 5 * LORA_RANK));
        ushort input_6 = intel_sub_group_block_read_us((const __global ushort*)(ptrA + 6 * LORA_RANK));
        ushort input_7 = intel_sub_group_block_read_us((const __global ushort*)(ptrA + 7 * LORA_RANK));
#endif

            //__attribute__((opencl_unroll_hint))
        for (int kk = 0; kk < SUBGROUP_SIZE; kk++) {

#if STATE_TYPE_SIZE == 4
            ACCUMULATOR_TYPE bb_0 = AS_ACCUMULATOR_TYPE(intel_sub_group_block_read((const __global uint*)(ptrB + 0 * SUBGROUP_SIZE)));
            ACCUMULATOR_TYPE bb_1 = AS_ACCUMULATOR_TYPE(intel_sub_group_block_read((const __global uint*)(ptrB + 1 * SUBGROUP_SIZE)));
#else
            ACCUMULATOR_TYPE bb_0 = AS_ACCUMULATOR_TYPE(intel_sub_group_block_read_us((const __global ushort*)(ptrB + 0 * SUBGROUP_SIZE)));
            ACCUMULATOR_TYPE bb_1 = AS_ACCUMULATOR_TYPE(intel_sub_group_block_read_us((const __global ushort*)(ptrB + 1 * SUBGROUP_SIZE)));
#endif

#if INPUT1_TYPE_SIZE == 4
            ACCUMULATOR_TYPE aa_0 = AS_ACCUMULATOR_TYPE(sub_group_broadcast(input_0, kk));
            ACCUMULATOR_TYPE aa_1 = AS_ACCUMULATOR_TYPE(sub_group_broadcast(input_1, kk));
            ACCUMULATOR_TYPE aa_2 = AS_ACCUMULATOR_TYPE(sub_group_broadcast(input_2, kk));
            ACCUMULATOR_TYPE aa_3 = AS_ACCUMULATOR_TYPE(sub_group_broadcast(input_3, kk));
            ACCUMULATOR_TYPE aa_4 = AS_ACCUMULATOR_TYPE(sub_group_broadcast(input_4, kk));
            ACCUMULATOR_TYPE aa_5 = AS_ACCUMULATOR_TYPE(sub_group_broadcast(input_5, kk));
            ACCUMULATOR_TYPE aa_6 = AS_ACCUMULATOR_TYPE(sub_group_broadcast(input_6, kk));
            ACCUMULATOR_TYPE aa_7 = AS_ACCUMULATOR_TYPE(sub_group_broadcast(input_7, kk));
#else
            ACCUMULATOR_TYPE aa_0 = AS_ACCUMULATOR_TYPE(intel_sub_group_broadcast(input_0, kk));
            ACCUMULATOR_TYPE aa_1 = AS_ACCUMULATOR_TYPE(intel_sub_group_broadcast(input_1, kk));
            ACCUMULATOR_TYPE aa_2 = AS_ACCUMULATOR_TYPE(intel_sub_group_broadcast(input_2, kk));
            ACCUMULATOR_TYPE aa_3 = AS_ACCUMULATOR_TYPE(intel_sub_group_broadcast(input_3, kk));
            ACCUMULATOR_TYPE aa_4 = AS_ACCUMULATOR_TYPE(intel_sub_group_broadcast(input_4, kk));
            ACCUMULATOR_TYPE aa_5 = AS_ACCUMULATOR_TYPE(intel_sub_group_broadcast(input_5, kk));
            ACCUMULATOR_TYPE aa_6 = AS_ACCUMULATOR_TYPE(intel_sub_group_broadcast(input_6, kk));
            ACCUMULATOR_TYPE aa_7 = AS_ACCUMULATOR_TYPE(intel_sub_group_broadcast(input_7, kk));
#endif




            sum_0_0 = fma(aa_0, bb_0, sum_0_0);
            sum_0_1 = fma(aa_0, bb_1, sum_0_1);
            sum_1_0 = fma(aa_1, bb_0, sum_1_0);
            sum_1_1 = fma(aa_1, bb_1, sum_1_1);
            sum_2_0 = fma(aa_2, bb_0, sum_2_0);
            sum_2_1 = fma(aa_2, bb_1, sum_2_1);
            sum_3_0 = fma(aa_3, bb_0, sum_3_0);
            sum_3_1 = fma(aa_3, bb_1, sum_3_1);
            sum_4_0 = fma(aa_4, bb_0, sum_4_0);
            sum_4_1 = fma(aa_4, bb_1, sum_4_1);
            sum_5_0 = fma(aa_5, bb_0, sum_5_0);
            sum_5_1 = fma(aa_5, bb_1, sum_5_1);
            sum_6_0 = fma(aa_6, bb_0, sum_6_0);
            sum_6_1 = fma(aa_6, bb_1, sum_6_1);
            sum_7_0 = fma(aa_7, bb_0, sum_7_0);
            sum_7_1 = fma(aa_7, bb_1, sum_7_1);

            ptrB += N;
        }
        ptrA += SUBGROUP_SIZE;
    }

    __global INPUT0_TYPE *main_ptr = main_input + m_idx * N + n_idx;


    INPUT0_TYPE main_N_0 = 0;
    INPUT0_TYPE main_N_1 = 0;

#if INPUT1_TYPE_SIZE == 4
    main_N_0 = AS_INPUT0_TYPE(intel_sub_group_block_read((const __global uint*)(main_ptr + SUBGROUP_SIZE * 0)));
    main_N_1 = AS_INPUT0_TYPE(intel_sub_group_block_read((const __global uint*)(main_ptr + SUBGROUP_SIZE * 1)));
    intel_sub_group_block_write((__global uint*)(ptrC + SUBGROUP_SIZE * 0), as_int(TO_INPUT0_TYPE(sum_0_0) + main_N_0));
    intel_sub_group_block_write((__global uint*)(ptrC + SUBGROUP_SIZE * 1), as_int(TO_INPUT0_TYPE(sum_0_1) + main_N_1));
    main_ptr += N;
    ptrC += N;

    main_N_0 = AS_INPUT0_TYPE(intel_sub_group_block_read((const __global uint*)(main_ptr + SUBGROUP_SIZE * 0)));
    main_N_1 = AS_INPUT0_TYPE(intel_sub_group_block_read((const __global uint*)(main_ptr + SUBGROUP_SIZE * 1)));
    intel_sub_group_block_write((__global uint*)(ptrC + SUBGROUP_SIZE * 0), as_int(TO_INPUT0_TYPE(sum_1_0) + main_N_0));
    intel_sub_group_block_write((__global uint*)(ptrC + SUBGROUP_SIZE * 1), as_int(TO_INPUT0_TYPE(sum_1_1) + main_N_1));
    main_ptr += N;
    ptrC += N;

    main_N_0 = AS_INPUT0_TYPE(intel_sub_group_block_read((const __global uint*)(main_ptr + SUBGROUP_SIZE * 0)));
    main_N_1 = AS_INPUT0_TYPE(intel_sub_group_block_read((const __global uint*)(main_ptr + SUBGROUP_SIZE * 1)));
    intel_sub_group_block_write((__global uint*)(ptrC + SUBGROUP_SIZE * 0), as_int(TO_INPUT0_TYPE(sum_2_0) + main_N_0));
    intel_sub_group_block_write((__global uint*)(ptrC + SUBGROUP_SIZE * 1), as_int(TO_INPUT0_TYPE(sum_2_1) + main_N_1));
    main_ptr += N;
    ptrC += N;

    main_N_0 = AS_INPUT0_TYPE(intel_sub_group_block_read((const __global uint*)(main_ptr + SUBGROUP_SIZE * 0)));
    main_N_1 = AS_INPUT0_TYPE(intel_sub_group_block_read((const __global uint*)(main_ptr + SUBGROUP_SIZE * 1)));
    intel_sub_group_block_write((__global uint*)(ptrC + SUBGROUP_SIZE * 0), as_int(TO_INPUT0_TYPE(sum_3_0) + main_N_0));
    intel_sub_group_block_write((__global uint*)(ptrC + SUBGROUP_SIZE * 1), as_int(TO_INPUT0_TYPE(sum_3_1) + main_N_1));
    main_ptr += N;
    ptrC += N;

    main_N_0 = AS_INPUT0_TYPE(intel_sub_group_block_read((const __global uint*)(main_ptr + SUBGROUP_SIZE * 0)));
    main_N_1 = AS_INPUT0_TYPE(intel_sub_group_block_read((const __global uint*)(main_ptr + SUBGROUP_SIZE * 1)));
    intel_sub_group_block_write((__global uint*)(ptrC + SUBGROUP_SIZE * 0), as_int(TO_INPUT0_TYPE(sum_4_0) + main_N_0));
    intel_sub_group_block_write((__global uint*)(ptrC + SUBGROUP_SIZE * 1), as_int(TO_INPUT0_TYPE(sum_4_1) + main_N_1));
    main_ptr += N;
    ptrC += N;

    main_N_0 = AS_INPUT0_TYPE(intel_sub_group_block_read((const __global uint*)(main_ptr + SUBGROUP_SIZE * 0)));
    main_N_1 = AS_INPUT0_TYPE(intel_sub_group_block_read((const __global uint*)(main_ptr + SUBGROUP_SIZE * 1)));
    intel_sub_group_block_write((__global uint*)(ptrC + SUBGROUP_SIZE * 0), as_int(TO_INPUT0_TYPE(sum_5_0) + main_N_0));
    intel_sub_group_block_write((__global uint*)(ptrC + SUBGROUP_SIZE * 1), as_int(TO_INPUT0_TYPE(sum_5_1) + main_N_1));
    main_ptr += N;
    ptrC += N;

    main_N_0 = AS_INPUT0_TYPE(intel_sub_group_block_read((const __global uint*)(main_ptr + SUBGROUP_SIZE * 0)));
    main_N_1 = AS_INPUT0_TYPE(intel_sub_group_block_read((const __global uint*)(main_ptr + SUBGROUP_SIZE * 1)));
    intel_sub_group_block_write((__global uint*)(ptrC + SUBGROUP_SIZE * 0), as_int(TO_INPUT0_TYPE(sum_6_0) + main_N_0));
    intel_sub_group_block_write((__global uint*)(ptrC + SUBGROUP_SIZE * 1), as_int(TO_INPUT0_TYPE(sum_6_1) + main_N_1));
    main_ptr += N;
    ptrC += N;

    main_N_0 = AS_INPUT0_TYPE(intel_sub_group_block_read((const __global uint*)(main_ptr + SUBGROUP_SIZE * 0)));
    main_N_1 = AS_INPUT0_TYPE(intel_sub_group_block_read((const __global uint*)(main_ptr + SUBGROUP_SIZE * 1)));
    intel_sub_group_block_write((__global uint*)(ptrC + SUBGROUP_SIZE * 0), as_int(TO_INPUT0_TYPE(sum_7_0) + main_N_0));
    intel_sub_group_block_write((__global uint*)(ptrC + SUBGROUP_SIZE * 1), as_int(TO_INPUT0_TYPE(sum_7_1) + main_N_1));
    main_ptr += N;
    ptrC += N;
#else
    main_N_0 = AS_INPUT0_TYPE(intel_sub_group_block_read_us((const __global ushort*)(main_ptr + SUBGROUP_SIZE * 0)));
    main_N_1 = AS_INPUT0_TYPE(intel_sub_group_block_read_us((const __global ushort*)(main_ptr + SUBGROUP_SIZE * 1)));
    intel_sub_group_block_write_us((__global ushort*)(ptrC + SUBGROUP_SIZE * 0), as_short(TO_INPUT0_TYPE(sum_0_0) + main_N_0));
    intel_sub_group_block_write_us((__global ushort*)(ptrC + SUBGROUP_SIZE * 1), as_short(TO_INPUT0_TYPE(sum_0_1) + main_N_1));
    main_ptr += N;
    ptrC += N;

    main_N_0 = AS_INPUT0_TYPE(intel_sub_group_block_read_us((const __global ushort*)(main_ptr + SUBGROUP_SIZE * 0)));
    main_N_1 = AS_INPUT0_TYPE(intel_sub_group_block_read_us((const __global ushort*)(main_ptr + SUBGROUP_SIZE * 1)));
    intel_sub_group_block_write_us((__global ushort*)(ptrC + SUBGROUP_SIZE * 0), as_short(TO_INPUT0_TYPE(sum_1_0) + main_N_0));
    intel_sub_group_block_write_us((__global ushort*)(ptrC + SUBGROUP_SIZE * 1), as_short(TO_INPUT0_TYPE(sum_1_1) + main_N_1));
    main_ptr += N;
    ptrC += N;

    main_N_0 = AS_INPUT0_TYPE(intel_sub_group_block_read_us((const __global ushort*)(main_ptr + SUBGROUP_SIZE * 0)));
    main_N_1 = AS_INPUT0_TYPE(intel_sub_group_block_read_us((const __global ushort*)(main_ptr + SUBGROUP_SIZE * 1)));
    intel_sub_group_block_write_us((__global ushort*)(ptrC + SUBGROUP_SIZE * 0), as_short(TO_INPUT0_TYPE(sum_2_0) + main_N_0));
    intel_sub_group_block_write_us((__global ushort*)(ptrC + SUBGROUP_SIZE * 1), as_short(TO_INPUT0_TYPE(sum_2_1) + main_N_1));
    main_ptr += N;
    ptrC += N;

    main_N_0 = AS_INPUT0_TYPE(intel_sub_group_block_read_us((const __global ushort*)(main_ptr + SUBGROUP_SIZE * 0)));
    main_N_1 = AS_INPUT0_TYPE(intel_sub_group_block_read_us((const __global ushort*)(main_ptr + SUBGROUP_SIZE * 1)));
    intel_sub_group_block_write_us((__global ushort*)(ptrC + SUBGROUP_SIZE * 0), as_short(TO_INPUT0_TYPE(sum_3_0) + main_N_0));
    intel_sub_group_block_write_us((__global ushort*)(ptrC + SUBGROUP_SIZE * 1), as_short(TO_INPUT0_TYPE(sum_3_1) + main_N_1));
    main_ptr += N;
    ptrC += N;

    main_N_0 = AS_INPUT0_TYPE(intel_sub_group_block_read_us((const __global ushort*)(main_ptr + SUBGROUP_SIZE * 0)));
    main_N_1 = AS_INPUT0_TYPE(intel_sub_group_block_read_us((const __global ushort*)(main_ptr + SUBGROUP_SIZE * 1)));
    intel_sub_group_block_write_us((__global ushort*)(ptrC + SUBGROUP_SIZE * 0), as_short(TO_INPUT0_TYPE(sum_4_0) + main_N_0));
    intel_sub_group_block_write_us((__global ushort*)(ptrC + SUBGROUP_SIZE * 1), as_short(TO_INPUT0_TYPE(sum_4_1) + main_N_1));
    main_ptr += N;
    ptrC += N;

    main_N_0 = AS_INPUT0_TYPE(intel_sub_group_block_read_us((const __global ushort*)(main_ptr + SUBGROUP_SIZE * 0)));
    main_N_1 = AS_INPUT0_TYPE(intel_sub_group_block_read_us((const __global ushort*)(main_ptr + SUBGROUP_SIZE * 1)));
    intel_sub_group_block_write_us((__global ushort*)(ptrC + SUBGROUP_SIZE * 0), as_short(TO_INPUT0_TYPE(sum_5_0) + main_N_0));
    intel_sub_group_block_write_us((__global ushort*)(ptrC + SUBGROUP_SIZE * 1), as_short(TO_INPUT0_TYPE(sum_5_1) + main_N_1));
    main_ptr += N;
    ptrC += N;

    main_N_0 = AS_INPUT0_TYPE(intel_sub_group_block_read_us((const __global ushort*)(main_ptr + SUBGROUP_SIZE * 0)));
    main_N_1 = AS_INPUT0_TYPE(intel_sub_group_block_read_us((const __global ushort*)(main_ptr + SUBGROUP_SIZE * 1)));
    intel_sub_group_block_write_us((__global ushort*)(ptrC + SUBGROUP_SIZE * 0), as_short(TO_INPUT0_TYPE(sum_6_0) + main_N_0));
    intel_sub_group_block_write_us((__global ushort*)(ptrC + SUBGROUP_SIZE * 1), as_short(TO_INPUT0_TYPE(sum_6_1) + main_N_1));
    main_ptr += N;
    ptrC += N;

    main_N_0 = AS_INPUT0_TYPE(intel_sub_group_block_read_us((const __global ushort*)(main_ptr + SUBGROUP_SIZE * 0)));
    main_N_1 = AS_INPUT0_TYPE(intel_sub_group_block_read_us((const __global ushort*)(main_ptr + SUBGROUP_SIZE * 1)));
    intel_sub_group_block_write_us((__global ushort*)(ptrC + SUBGROUP_SIZE * 0), as_short(TO_INPUT0_TYPE(sum_7_0) + main_N_0));
    intel_sub_group_block_write_us((__global ushort*)(ptrC + SUBGROUP_SIZE * 1), as_short(TO_INPUT0_TYPE(sum_7_1) + main_N_1));
    main_ptr += N;
    ptrC += N;
#endif
}

    
    '''
    if withscale:
        src = src_gemma
    elif withSum:
        src = src_gemmb 
    return [func, src]

 #A_regM, A_regN, A_sgM, A_sgN: GEMMA register blocking in one sg and sg number in M and N dimesnion.
 #B_regM, B_regN, B_sgM, B_sgN: GEMMB register blocking in one sg and sg number in M and N dimesnion.
class LORA_1ST:
    def __init__(self, batch, rank, input_state, output_state,  A_regM, A_regN, A_sgM, A_sgN, B_regM, B_regN, B_sgM, B_sgN, use_ref = False):
        self.batch = batch
        self.rank = rank
        self.input_state = input_state
        self.output_state = output_state
        self.sg_sz = 8
        if use_ref == False:
            assert batch >= A_regM and batch >=B_regM , f'batch:{batch} is smaller than A_regM/B_regM:{A_regM}/{B_regM}'
            assert rank % self.sg_sz == 0 , f'rank:{rank} is not multiple of SG_SZ:{self.sg_sz}'
            assert output_state % self.sg_sz == 0 , f'output_state:{output_state} is not multiple of SG_SZ:{self.sg_sz}'
            assert self.input_state % self.sg_sz == 0, f"'input state' {self.input_state} is not multiple of SG_SZ {self.sg_sz}"
            assert rank >= A_regN * self.sg_sz, f'rank:{rank} is smaller than :A_regN * SG_SZ {A_regN*self.sg_sz}'
            assert output_state >= B_regN * self.sg_sz, f'output_state:{output_state} is smaller than :B_regN * SG_SZ {B_regN*self.sg_sz}'
            
            
            A_BM = A_regM*A_sgM
            A_BN = A_regN*A_sgN*self.sg_sz
            B_BM = B_regM*B_sgM
            B_BN = B_regN*B_sgN*self.sg_sz
            
            gemma_func, gemma_kernel_src = generate_gemm_src(A_regM, A_regN,  True, False)
            gemmb_func, gemmb_kernel_src = generate_gemm_src(B_regM, B_regN,  False, True)
            self.gemma_func = gemma_func
            self.gemmb_func = gemmb_func
            print(gemma_kernel_src)
            print(gemmb_kernel_src)

            self.gemma_GWS = [ALIGN_UP(batch, A_BM)//A_regM , ALIGN_UP(rank, A_BN)//(A_regN)]
            self.gemma_LWS = [A_sgM, A_sgN * self.sg_sz]
            assert A_sgM *A_sgN * self.sg_sz <= 1024, f" A_LWS:{self.gemma_LWS} exceed 1024 limitation"
            
            self.gemmb_GWS = [ALIGN_UP(batch, B_BM)//B_regM , ALIGN_UP(output_state, B_BN)//(B_regN)]
            self.gemmb_LWS = [B_sgM, B_sgN *  self.sg_sz]
            assert B_sgM *B_sgN *  self.sg_sz <= 1024, f" B_LWS:{self.gemmb_LWS} exceed 1024 limitation"
            
            self.kernel_opt_gemma = kernel_cache(gemma_kernel_src, options=f"-DSG_SZ={self.sg_sz}")
            self.kernel_opt_gemmb = kernel_cache(gemmb_kernel_src, options=f"-DSG_SZ={self.sg_sz}")
        else:
            self.cl_kernels_ref = kernel_cache(cl_kernel_sources_ref)
        self.use_ref = use_ref
        if use_ref == False:
            print(f'----------------------------------------------------------------------------------------------------------------------------------')
            print(f'| BATCH = {batch} INPUT_STATE:{input_state}, RANK:{rank}, OUPUT_STATE:{output_state}:')
            print(f'[1ST_GEMMA] GWS:{self.gemma_GWS}, LWS:{self.gemma_LWS}, M:{batch}/{A_BM}, N:{rank}/{A_BN}, A_sgM={A_sgM},A_regM={A_regM},A_sgN={A_sgN},A_regN={A_regN}')
            print(f'[1st_GEMMB] GWS:{self.gemmb_GWS}, LWS:{self.gemmb_LWS}, M:{batch}/{B_BM}, N:{rank}/{B_BN}, B_sgM={B_sgM},B_regM={B_regM},B_sgN={B_sgN},B_regN={B_regN}')
            print(f'----------------------------------------------------------------------------------------------------------------------------------')


    def __call__(self, mainInput, loraInput, stataA, stateAlpha, stateB, Aoutput, result):

        if self.use_ref:
            tA_output_ref = cl.tensor([self.batch, self.rank], np.dtype(np.float32))
            REF_LOCAL_SIZE = 8
                        
            self.cl_kernels_ref.enqueue("gemmA", [self.batch, self.rank],[1, REF_LOCAL_SIZE], loraInput, stataA,
                                        tA_output_ref, self.rank, self.input_state, self.sg_sz, self.batch)
            self.cl_kernels_ref.enqueue("gemmB", [self.batch, self.output_state],[1, REF_LOCAL_SIZE],
                                        mainInput, tA_output_ref, stateB, result, stateAlpha, self.output_state, self.rank, self.batch)
        else:

            self.kernel_opt_gemma.enqueue(self.gemma_func, self.gemma_GWS, self.gemma_LWS, loraInput, stataA, Aoutput, stateAlpha, mainInput,
                                     self.batch, self.rank, self.input_state)
            self.kernel_opt_gemmb.enqueue(self.gemmb_func, self.gemmb_GWS, self.gemmb_LWS, Aoutput, stateB, result, stateAlpha, mainInput,
                                     self.batch, self.output_state, self.rank)
        return result



"""
------------------------------------------------------------------------------------------------------------------
test code
------------------------------------------------------------------------------------------------------------------
"""
def test_lora_2nd(input_state, rank, output_state, gemma_sgK = 8, gemma_sg_BK = 32, gemmb_sgN = 16, check_acc = False):
    cl.profiling(True)
    SG_SZ = 8
    vRANGE = 1
    if check_acc:
        REPEAT = 1
    else:
        REPEAT = 200
    # for GEMMA, K decides how many WGs are needed.
    gemma_wgs = ALIGN_UP(input_state, gemma_sg_BK *gemma_sgK)
    stateA = np.random.randint(-vRANGE, vRANGE+1, [input_state, rank]).astype(np.float32)
    alpha = np.random.rand(rank).astype(np.float32)
    stateB = np.random.randint(-vRANGE, vRANGE+1, [rank, output_state]).astype(np.float32)
    loraInput = np.random.randint(-vRANGE, vRANGE+1, [1, input_state]).astype(np.float32)
    mainInput = np.random.randint(-vRANGE, vRANGE+1, [1, output_state]).astype(np.float32)
    Aoutput = np.zeros([gemma_wgs, rank]).astype(np.float32)
    
    stateA_list= [cl.tensor(stateA) for _ in range(REPEAT)]
    alpha_list = [cl.tensor(alpha) for _ in range(REPEAT)]
    stateB_list = [cl.tensor(stateB)for _ in range(REPEAT)]
    loraInput_list = [cl.tensor(loraInput)for _ in range(REPEAT)]
    mainInput_list = [cl.tensor(mainInput)for _ in range(REPEAT)]
    #Must set the output to be zeros to avoid not all the data is updated in GEMMA. 
    A_output_list = [cl.tensor(Aoutput)for _ in range(REPEAT)]
    res_list = [cl.tensor([1, output_state], np.dtype(np.float32))for _ in range(REPEAT)]
    ref_list = [cl.tensor([1, output_state], np.dtype(np.float32))for _ in range(REPEAT)]

    ref = LORA_2ND(rank, input_state, output_state, gemma_sg_BK, gemma_sgK, gemmb_sgN,True)
    opt = LORA_2ND(rank, input_state, output_state, gemma_sg_BK, gemma_sgK, gemmb_sgN,False)
    
    if check_acc:
        opt(mainInput_list[0], loraInput_list[0], stateA_list[0], alpha_list[0], stateB_list[0], A_output_list[0], res_list[0])
        ref(mainInput_list[0], loraInput_list[0], stateA_list[0], alpha_list[0], stateB_list[0], A_output_list[0], ref_list[0])
        cl.finish()
        compare(ref_list[0].numpy(), res_list[0].numpy())
        print(f'INPUT_STATE:{input_state}, RANK:{rank}, OUPUT_STATE:{output_state} ACC PASS!')
    else:
        for i in range(0, REPEAT):
            opt(mainInput_list[i], loraInput_list[i], stateA_list[i], alpha_list[i], stateB_list[i],  A_output_list[i], res_list[i])
        profiling_data  = cl.finish()
        # FMA
        flops_a = rank*input_state*2
        # FMA + multiply alpha + add main
        flops_b = rank*output_state*2 + rank + output_state
        # lora input + weightA
        rd_bytes_a = (rank*input_state+input_state)*2
        # A output + main input + weightB  + scale
        rd_bytes_b = (rank*output_state+output_state*2+rank)*2

        for i in range(1, REPEAT):
            ns_a = profiling_data[i*2]
            ns_b = profiling_data[i*2 + 1]
            if VERBOSE:
                print(f"[GEMMA]: {flops_a*1e-6:.1f} Mflop/{ns_a*1e-3:.1f} us = {flops_a/ns_a:.3f} GFlops/s  {rd_bytes_a/1e6:.1f} MB/{ns_a*1e-3:.1f} us = {rd_bytes_a/ns_a:.1f} GB/s ,\
 [GEMMB]: {flops_b*1e-6:.1f} Mflop/{ns_b*1e-3:.1f} us = {flops_b/ns_b:.3f} GFlops/s  {rd_bytes_b/1e6:.1f} MB/{ns_b*1e-3:.1f} us = {rd_bytes_b/ns_b:.1f} GB/s ,\
 [total]: {(ns_a+ns_b)*1e-3:.1f}us")
            else:
                print(f'[latency]: {ns_a*1e-3:.1f} + {ns_b*1e-3:.1f} = {(ns_a+ns_b)*1e-3:.1f}us')


def test_lora_1st(batch, rank, input_state, output_state,  A_regM, A_regN, A_sgM, A_sgN, B_regM, B_regN, B_sgM, B_sgN, check_acc = False):
    cl.profiling(True)
    SG_SZ = 16
    vRANGE = 1
    if check_acc:
        REPEAT = 1
    else:
        REPEAT = 100
    # np.random.seed(0)
    stateA = np.random.randint(-vRANGE, vRANGE+1, [input_state, rank]).astype(np.float32)
    alpha = np.random.rand(rank).astype(np.float32)
    stateB = np.random.randint(-vRANGE, vRANGE+1, [rank, output_state]).astype(np.float32)
    loraInput = np.random.randint(-vRANGE, vRANGE+1, [batch, input_state]).astype(np.float32)
    mainInput = np.random.randint(-vRANGE, vRANGE+1, [batch, output_state]).astype(np.float32)
    Aoutput = np.zeros([batch, rank]).astype(np.float32) 
    
    stateA_list= [cl.tensor(stateA) for _ in range(REPEAT)]
    alpha_list = [cl.tensor(alpha) for _ in range(REPEAT)]
    stateB_list = [cl.tensor(stateB)for _ in range(REPEAT)]
    loraInput_list = [cl.tensor(loraInput)for _ in range(REPEAT)]
    mainInput_list = [cl.tensor(mainInput)for _ in range(REPEAT)]
    #Must set the output to be zeros to avoid not all the data is updated in GEMMA. 
    A_output_list = [cl.tensor(Aoutput)for _ in range(REPEAT)]
    res_list = [cl.tensor([batch, output_state], np.dtype(np.float32))for _ in range(REPEAT)]
    ref_list = [cl.tensor([batch, output_state], np.dtype(np.float32))for _ in range(REPEAT)]

    ref = LORA_1ST(batch, rank, input_state, output_state,  A_regM, A_regN, A_sgM, A_sgN, B_regM, B_regN, B_sgM, B_sgN,True)
    opt = LORA_1ST( batch, rank, input_state, output_state,  A_regM, A_regN, A_sgM, A_sgN, B_regM, B_regN, B_sgM, B_sgN,False)
    
    if check_acc:
        opt(mainInput_list[0], loraInput_list[0], stateA_list[0], alpha_list[0], stateB_list[0], A_output_list[0], res_list[0])
        ref(mainInput_list[0], loraInput_list[0], stateA_list[0], alpha_list[0], stateB_list[0], A_output_list[0], ref_list[0])
        cl.finish()
        compare(ref_list[0].numpy(), res_list[0].numpy())
        print(f'BATCH:{batch} INPUT_STATE:{input_state}, RANK:{rank}, OUPUT_STATE:{output_state} ACC PASS!')
    else:
        for i in range(0, REPEAT):
            opt(mainInput_list[i], loraInput_list[i], stateA_list[i], alpha_list[i], stateB_list[i],  A_output_list[i], res_list[i])
        profiling_data  = cl.finish()
        # FMA + multiply scale
        flops_a = batch*rank*input_state*2 + batch*rank
        # FMA + add
        flops_b = batch*rank*output_state*2 + batch*output_state
        # loarinput + stateA + scale
        rd_bytes_a = (rank*input_state + input_state*batch + rank)*2
        # maininput + Aoutput + stateB 
        rd_bytes_b = (rank*output_state + output_state*batch + rank*batch)*2

        for i in range(1, REPEAT):
            ns_a = profiling_data[i*2]
            ns_b = profiling_data[i*2 + 1]
            if VERBOSE:
                print(f"[GEMMA]: {flops_a*1e-6:.1f} Mflop/{ns_a*1e-3:.1f} us = {flops_a/ns_a:.3f} GFlops/s  {rd_bytes_a/1e6:.1f} MB/{ns_a*1e-3:.1f} us = {rd_bytes_a/ns_a:.1f} GB/s ,\
 [GEMMB]: {flops_b*1e-6:.1f} Mflop/{ns_b*1e-3:.1f} us = {flops_b/ns_b:.3f} GFlops/s  {rd_bytes_b/1e6:.1f} MB/{ns_b*1e-3:.1f} us = {rd_bytes_b/ns_b:.1f} GB/s ,\
 [total]: {(ns_a+ns_b)*1e-3:.1f}us")
            else:
                print(f'[latency]: {ns_a*1e-3:.1f} + {ns_b*1e-3:.1f} = {(ns_a+ns_b)*1e-3:.1f}us')


def blocking_2nd(rank, input_state, output_state):
    SG_SZ = 8
    MAX_WG_SZ = 512
    # Macro is the same with
    MAX_GEMMA_SGK = 32
    gemma_sg_BK = 32
    gemma_sgK = min(MAX_WG_SZ//rank, MAX_GEMMA_SGK)
    gemmb_sgN = MAX_WG_SZ//SG_SZ
    return [gemma_sg_BK, gemma_sgK, gemmb_sgN]

def blocking_1nd(batch, rank, input_state, output_state):
    SG_SZ = 8
    assert batch >= 8, f"batch:{batch} too small in 1st token, not support in opt kernel"     
# GEMMA will not divide N across WGs. 1. choose[regM, regN] based on rank and m, 
#                                     2, choose[sgN], 
#                                     3  sgM = 16//sgN or sgM = 8//sgN
# seems regM and regN influences FPS more than sgM, sgN. 
# If rank is 128, 256, and M >= 2000, can try[16, 2]??
# GEMMA rank == 64, use regM, regN for  [8,2] , sgM = 16, sgN = 2, when M >= 2048. 
# other wise,  choose regM = 4, regN = 1, sgM = ?, sgN = 1/2/8.
    if batch >= 2000:
        if rank == 64:
            A_regM, A_regN = [8, 2]
        elif rank == 128 or rank == 256:
            A_regM, A_regN = [8, 2]
        else:
            A_regM, A_regN = [4, 1]
        assert (rank//SG_SZ % A_regN) == 0
        A_sgN = rank//SG_SZ//A_regN
        A_sgM = DIV_UP(16, A_sgN)
    else:
        A_regM, A_regN = [4, 1]
        A_sgN = rank//SG_SZ//A_regN
        assert (rank//SG_SZ % A_regN) == 0
        # A_sgM = DIV_UP(16, A_sgN)
        A_sgM = DIV_UP(SG_SZ, A_sgN)
# GEMMB: M < 256 regM=8, regN=2, sgM=16, sgN=4, small M() fits into
#        M >= 256, regM=16, regN=2, sgM=8, sgN=4
    if batch < 256:
        B_regM, B_regN = [8, 2]
        B_sgM, B_sgN = [16, 4]

    else:
        B_regM, B_regN = [8, 2]
        B_sgM, B_sgN = [8, 4]

    return [A_regM, A_regN, A_sgM, A_sgN, B_regM, B_regN, B_sgM, B_sgN]

if __name__ == "__main__":

    batch = 32
    input_state = 512
    output_state = 2048
    rank = 16
    A_regM, A_regN, A_sgM, A_sgN, B_regM, B_regN, B_sgM, B_sgN = blocking_1nd(batch, rank, input_state, output_state)
    test_lora_1st(batch, rank ,input_state,  output_state, A_regM = A_regM, A_regN = A_regN, A_sgM=A_sgM, A_sgN = A_sgN,
                    B_regM=B_regM, B_regN=B_regN, B_sgM=B_sgM, B_sgN=B_sgN, check_acc=True)

    """
    test 1st acc:
    """
    if 0:
        for batch in range(8, 1024, 107):
            for input_state in (1024, 1536, 2048, 2560, 3072, 3840, 4096, 7*16, 11*16, 13*16, 15*16, 12*16,17*16):
                for output_state_idx in range(256//16, 1024//16):
                    for rank in (16, 32, 48, 64, 80, 128, 256):
                        A_regM, A_regN, A_sgM, A_sgN, B_regM, B_regN, B_sgM, B_sgN = blocking_1nd(batch, rank, input_state, output_state_idx*16)
                        test_lora_1st(batch, rank ,input_state,  output_state_idx*16, A_regM = A_regM, A_regN = A_regN, A_sgM=A_sgM, A_sgN = A_sgN,
                                      B_regM=B_regM, B_regN=B_regN, B_sgM=B_sgM, B_sgN=B_sgN, check_acc=True)
    
    """
    test 2nd acc:
    """
    if 0:
        for out_state in (256, 512, 1024, 2048, 1536, 3072, 3840, 15*16,  17*16, 7*16, 11*16, 13*16):
            for input_state in (1024, 1536, 2048, 2560, 3072, 3840, 4096, 7*16, 11*16, 13*16, 15*16, 12*16,17*16):
                for rank in (16, 32, 64, 128, 256):
                    # for sgk in range(16, 33):
                    #     for bk in (16, 32, 48, 64):
                    #         test_lora_2nd(input_state, rank, out_state, gemma_sgK = min(sgk, 512 // rank), gemma_sg_BK=bk, gemmb_sgN=32 , check_acc=True)
                    gemma_sg_BK, gemma_sgK, gemmb_sgN = blocking_2nd(rank, input_state, out_state)
                    test_lora_2nd(input_state, rank, out_state, gemma_sgK = gemma_sgK, gemma_sg_BK=gemma_sg_BK, gemmb_sgN=gemmb_sgN, check_acc=True)
    # """
    # test acc of minicpm and qwen2.5:
    # """
    # if 0:
    #     for batch in [1, 3192]:
    #         for rank in [64,]:
    #             for out_state in [512, 1536, 3840, 256, 8960]:
    #                 for input_state in [1536, 8960]:
    #                     if batch == 1:
    #                         gemma_sg_BK, gemma_sgK, gemmb_sgN = blocking_2nd(rank, input_state, out_state)
    #                         test_lora_2nd(input_state, rank, out_state, gemma_sgK, gemma_sg_BK, gemmb_sgN, check_acc=True)
    #                     else:
    #                         A_regM, A_regN, A_sgM, A_sgN, B_regM, B_regN, B_sgM, B_sgN = blocking_1nd(batch, rank, input_state, out_state)
    #                         test_lora_1st(batch, rank ,input_state,  out_state, A_regM = A_regM, A_regN = A_regN, A_sgM=A_sgM, A_sgN = A_sgN,
    #                                   B_regM=B_regM, B_regN=B_regN, B_sgM=B_sgM, B_sgN=B_sgN, check_acc=True)

    # """
    # test perf based on minicpm:
    # """
    # if 0:
    #     for rank in [64]:
    #         for out_state in [512, 1536, 3840]:
    #             gemma_sg_BK, gemma_sgK, gemmb_sgN = blocking_2nd(rank, 1536, out_state)
    #             test_lora_2nd(1536, rank, out_state, gemma_sgK, gemma_sg_BK, gemmb_sgN)
    #     for batch in [3051]:
    #         for rank in [64]:
    #             for out_state in [512, 1536, 3840]:
    #                 A_regM, A_regN, A_sgM, A_sgN, B_regM, B_regN, B_sgM, B_sgN = blocking_1nd(batch, rank, 1536, out_state)
    #                 test_lora_1st(batch, rank ,1536,  out_state, A_regM = A_regM, A_regN = A_regN, A_sgM=A_sgM, A_sgN = A_sgN,
    #                                     B_regM=B_regM, B_regN=B_regN, B_sgM=B_sgM, B_sgN=B_sgN)

    # """
    # test perf based on qwen:
    # """
    # if 0:
    #     for batch in [1, 3192]:
    #         for rank in [64,]:
    #             for out_state in [1536, 3840, 256, 8960]:
    #                 for input_state in [1536, 8960]:
    #                     if batch == 1:
    #                         gemma_sg_BK, gemma_sgK, gemmb_sgN = blocking_2nd(rank, input_state, out_state)
    #                         test_lora_2nd(input_state, rank, out_state, gemma_sgK, gemma_sg_BK, gemmb_sgN)
    #                     else:
    #                         A_regM, A_regN, A_sgM, A_sgN, B_regM, B_regN, B_sgM, B_sgN = blocking_1nd(batch, rank, input_state, out_state)
    #                         test_lora_1st(batch, rank ,input_state,  out_state, A_regM = A_regM, A_regN = A_regN, A_sgM=A_sgM, A_sgN = A_sgN,
    #                                   B_regM=B_regM, B_regN=B_regN, B_sgM=B_sgM, B_sgN=B_sgN)
