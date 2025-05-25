import argparse
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)
torch.set_printoptions(linewidth=1024)

parser = argparse.ArgumentParser('')
parser.add_argument('-i', "--impl", type=int, default=1)
parser.add_argument('-b', "--batch", type=int, default=1)
parser.add_argument('-nh', "--num-heads", type=int, default=28)
parser.add_argument('-nkvh', "--num-kv-heads", type=int, default=4)
parser.add_argument('-ql', "--q-len", type=int, default=8192)
parser.add_argument('-kvl', "--kv-len", type=int, default=8192)
parser.add_argument('-hs', "--head-size", type=int, default=128)
parser.add_argument('-v', "--verbose", type=int, default=-1)
parser.add_argument('-c', "--causal-mask", action="store_true")

#parser.add_argument('-q', "--quant_type", type=str, default="w4a", choices=['f16', 'f16b1', 'w4a', 'w4a_cpu', 'f16xmx', 'w4x'])
#parser.add_argument('-hf', '--hf_model_path', type=str, nargs='?', default='/mnt/llm_irs/models_original/Qwen2-0.5B-Instruct/')
#parser.add_argument('--save', type=str, nargs='?', default=None)
#parser.add_argument('--load', type=str, nargs='?', default=None)
args = parser.parse_args()
print(args)

enable_vprint = False
def vprint(*all_args):
    global enable_vprint
    if enable_vprint:
        print(*all_args)


batch = args.batch
q_len, q_step = args.q_len, 8
kv_len, kv_step = args.kv_len, 16
num_heads = args.num_heads
num_kv_heads = args.num_kv_heads
if num_kv_heads <= 0:
    num_kv_heads = num_heads
head_size = args.head_size
enable_gqa = num_heads > num_kv_heads
causal_mask = int(args.causal_mask)

#q_len, q_step = 160, 16
#kv_len, kv_step = 800, 16
low = -7
high = 8
act_dtype = torch.float16
q = torch.randint(low, high, [batch, q_len, num_heads, head_size]).to(dtype=act_dtype)/high
k = torch.randint(low, high, [batch, kv_len, num_kv_heads, head_size]).to(dtype=act_dtype)/high
v = torch.randint(low, high, [batch, kv_len, num_kv_heads, head_size]).to(dtype=act_dtype)/high

# random attnmask
attention_mask = torch.full([batch, 1, q_len, kv_len], torch.finfo(act_dtype).min).to(dtype=act_dtype)


# causal attn-mask
if causal_mask:
    for i in range(q_len):
        attention_mask[:, :, i, 0:(kv_len - q_len + i + 1)] = 0
    print(attention_mask[0,0,:10, :10])
else:
    attention_mask[torch.rand(batch, 1, q_len, kv_len) > 0.5] = 0    

#attention_mask[...] = 0

# BLHS=>BHLS
q = q.transpose(1,2)
k = k.transpose(1,2)
v = v.transpose(1,2)

#q[:,:,:,:] = q[:,:,2,:]
#attention_mask[:,:,:,:] = attention_mask[:,:,2,:]

print("q:", q.shape, q.dtype)
print("k:", k.shape, k.dtype)
print("v:", v.shape, v.dtype)
print("attention_mask:", attention_mask.shape, attention_mask.dtype)

def get_org(Q, K, V, attention_mask):
    B,H,L,S = Q.shape
    _,Hkv,_,_ = K.shape
    out = torch.zeros([B,H,L,S], dtype=Q.dtype)
    scale_factor = S**(-0.5)
    for b in range(B):
        for h in range(H):
            hkv = h // (H//Hkv)
            attn_score = Q[b, h, :, :].to(dtype=torch.float32) @ (K[b, hkv, :,:].transpose(0,1)).to(dtype=torch.float32)
            attn_score *= scale_factor
            attn_score += attention_mask[b,0,:,:]
            #print(attn_score.shape)
            attn_weights = F.softmax(attn_score, 1)
            out[b,h,:,:] = attn_weights @ V[b, hkv, :, :].to(dtype=torch.float32)
    return out

ref = F.scaled_dot_product_attention(q, k, v, attention_mask, dropout_p=0.0, enable_gqa = enable_gqa)
org = get_org(q,k,v,attention_mask)

def check_close(input, other, atol=1e-3, rtol=1e-3):
    print(f"[check_close] {input.shape}{input.dtype} vs {other.shape}{other.dtype}")
    rtol_max = (((input - other).abs() - 1e-5)/other.abs())[other != 0].max()
    atol_max = (((input - other).abs()) - 1e-5*other.abs()).max()
    print(f"[check_close] rtol_max: {rtol_max}")
    print(f"[check_close] atol_max: {atol_max}")
    if not torch.allclose(input, other, atol=atol, rtol=rtol):
        close_check = torch.isclose(input, other, atol=atol, rtol=rtol)
        not_close_indices = torch.where(~close_check) # Invert the close check to find failures
        print(f"Not close indices: {not_close_indices}")
        print(f"    input_tensor: {input[not_close_indices]}")
        print(f"    other_tensor: {other[not_close_indices]}")
        assert 0

check_close(ref, org, atol=1e-3, rtol=1e-2)

# [batch, seq-len, heads, size] BLHS
print("ref:", ref.shape, ref.dtype)

# blocking on kv-len dimension with online-softmax
def get_flash0(query, key, value, attention_mask):
    global enable_vprint
    B,H,q_len,hs = query.shape
    _,Hkv,kv_len,_ = key.shape
    out = torch.zeros([B,H,q_len,hs], dtype=value.dtype)
    scale_factor = hs**(-0.5)
    for b in range(B):
        for h in range(H):
            hkv = h // (H//Hkv)
            Q = query[b, h, :, :]
            K = key[b, hkv, :, :]
            V = value[b, hkv, :, :]
            mask = attention_mask[b,0,:,:]
            for i in range(0, q_len, q_step):
                i1 = min(i + q_step, q_len)
                # online softmax states:
                #     per-row max-value     : [q_step, 1]
                #     per-row sum           : [q_step, 1]
                #     current accumulated V : [q_step, S]
                #cur_max = torch.full([i1-i, 1], torch.finfo(torch.float32).min, dtype=torch.float32)
                #cur_sum = torch.full([i1-i, 1], 0, dtype=torch.float32)
                #cur_O = torch.full([i1-i, hs], 0, dtype=torch.float32)

                # we prefer transposed S since:
                #   1. per-column max/sum is easier than per-row in register, 
                #   2. loop in K direction is inner-most, so load K in noraml
                #      instead of VNNI is faster, load Q in transpose-VNNI is 
                #      done only once at loop begin.
                #   3. 
                rQt = Q[i:i1, :].transpose(0,1)  # sub Q block only transposed & VNNI packed once

                for j in range(0, kv_len, kv_step):
                    j1 = min(j + kv_step, kv_len)

                    if (args.verbose >= 0):
                        print(f"======== i={i} j={j} ==========")

                    if (j == args.verbose): enable_vprint = True
                    # compute in local SRAM
                    # Step8: On chip, compute S(_𝑗)𝑖= Q𝑖K𝑇 𝑗∈ R𝐵𝑟 ×𝐵𝑐.
                    rK = K[j:j1,:]
                    St = (rK @ rQt).to(dtype=torch.float32)
                    MaskT = mask[i:i1, j:j1].transpose(0,1)

                    vprint("rK=", rK.shape, rK)
                    vprint("rQt=",rQt.shape, rQt[:16,:])
                    vprint("St=",St.shape, St)
                    vprint("MaskT=",MaskT.shape, MaskT)

                    St *= scale_factor
                    St += mask[i:i1, j:j1].transpose(0,1)
                    vprint("St=",St.shape, St)

                    rowmax = St.max(0, keepdim=True).values
                    if j == 0:
                        cur_max = rowmax
                    else:
                        rowmax = torch.maximum(cur_max, rowmax)
                    vprint("rowmax=", rowmax)

                    # compute in local SRAM
                    St = torch.exp(St - rowmax)
                    vprint("St(Pt)=", St.shape, St)

                    rowsumP = St.sum(0, keepdim=True)
                    vprint("rowsumP=", rowsumP)

                    # corrected sum of previous block
                    if j > 0:
                        max_comp = torch.exp(cur_max - rowmax)

                    if j == 0:
                        cur_sum = rowsumP
                    else:
                        cur_sum = cur_sum * max_comp + rowsumP

                    # softmax normalize is saved accoridng to flash-attn2 section 3.1.1
                    # We can instead maintain an “un-scaled” version of O(2) and keep around the statistics ℓ(2)
                    partial_attn_weight = St.to(dtype=torch.float16).transpose(0,1)
                    
                    vprint("P=", partial_attn_weight.shape, partial_attn_weight)

                    rV = V[j:j1, :]
                    vprint("rV=",rV.shape, rV)

                    # correct last Output to current statistics
                    
                    if j == 0:
                        cur_O = partial_attn_weight @ rV
                    else:
                        cur_O = (cur_O * max_comp.transpose(0, 1))
                        vprint("cur_O1=", cur_O)
                        cur_O += partial_attn_weight @ rV
                    vprint("cur_O2=", cur_O.shape, cur_O)

                    cur_max = rowmax
                    if (j == args.verbose): assert 0

                cur_O_f16 = (cur_O/cur_sum.transpose(0, 1)).to(torch.float16)

                if (i == args.verbose):
                    enable_vprint = True
                    vprint("cur_O_f16=", cur_O_f16.shape, cur_O_f16)
                    assert 0

                out[b, h, i:i1, :] = cur_O_f16
    return out

if args.impl == 0:
    f0 = get_flash0(q,k,v,attention_mask)
    check_close(org, f0, atol=1e-2, rtol=1e-3)
    print("=========== PASS ===========")
    sys.exit(0)

#====================================================================================================
# using the same parameter & inputs, develop cm kernels which produces the same output
# prototyping CM kernels
from clops import cl
import numpy as np
import time

# transpose back to orginal shape: [batch, q_len, num_heads, head_size]
q = q.transpose(1,2)
k = k.transpose(1,2)
v = v.transpose(1,2)
print("q:", q.shape, q.dtype)
print("k:", k.shape, k.dtype)
print("v:", v.shape, v.dtype)
print("attention_mask:", attention_mask.shape, attention_mask.dtype)

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

'''
ugemm_qk: [q_step, head_size] x [head_size, kv_step]
ugemm_kq: [kv_step, head_size] x [head_size, q_step]
ugemm_pv: [q_step, kv_step] x [kv_step, head_size]
'''

scale_factor = 1.0/(head_size**0.5)

if 1:
    # only 116 ms since adjacent query-heads shares same KV-heads
    # dispatch them together can help increase cache-usage
    GWS=[batch, num_heads, q_len//(2*q_step)]
    WG_SIZE = min(GWS[-1], 16)
    LWS=[1, 1, WG_SIZE]

    dim_batch = 0
    dim_head = 1
    dim_q_batch = 2
else:
    # 140 ms + 
    GWS=[q_len//q_step, num_heads, batch]
    WG_SIZE = min(q_len//q_step, 16)
    LWS=[WG_SIZE, 1, 1]

    dim_batch = 2
    dim_head = 1
    dim_q_batch = 0

if False:
    # for GQA/multi-query, invoke all heads sharing same KV can further help cache-hit-rate
    # but it's not helping, WHY?
    num_heads_share_kv = num_heads//num_kv_heads
    GWS=[batch, num_heads_share_kv, q_len//q_step]
    WG_SIZE = min(q_len//q_step, 16)
    LWS=[1, 1, WG_SIZE]

    dim_batch = 0
    dim_head = 1
    dim_q_batch = 2

print("GWS=", GWS)
print("LWS=", LWS)
print(f"total HW threads: {batch} x {num_heads} x {q_len//q_step} = {batch * num_heads * (q_len//q_step)}")
#========================================================================
# Optimization Log
r'''
increase WG_SIZE to 16 (-70ms)
use GRF to store temp Output(rO) instead of SLM, requires `-Qxcm_register_file_size=256` (-40ms)
avoid type-promotion:   St = cm_mul<float>(St, scale_factor);    =>   St = cm_mul<float>(St, (float)scale_factor);   (-10ms) 
change dtype of attention_mask from float to half (-14ms)
avoid indirect register access: unroll for loop which access matrix rows using loop-index
use GRF to store temp Input rQ instead of SLM, this allows more Work-Groups to be packed into same Xe-core!!!

in Xe1, GRF is only half size of Xe2, so reduce q_step to fit rO/rQ into GRF
reading V into SLM is slow due to smaller REG_N of Xe1 DPAS (8 vs 16), increase the consecutive read width to 16 helps cache-line usage

keep all other parameter unchanged, if head-size is 128, latency is 11ms, if head-size is 144, latency increases to 24ms


 head-size : latency(ms) @kv-length 1024/512
     96 : 8
    112 : 8     5
    128 : 11    6
    144 : 24    12
    160 : 69    34
    176 : 63

ql 8192: 115ms
   4096: 58ms
   2048: 29ms <===========
   1024: 14ms <===========
    512: 8.6
    256: 4.5
    128: 2.0

kvl 8192: 118
    4096: 59
    2048: 23.5 <===========
    1024: 11.3 <===========
     512: 5.99
     256: 3.7
     128: 2.5 (error)

nkvh 28: 145ms
     14: 122ms
      7: 118 <=============
      4: 118 <=============
      2: 125 ?????
      1: 130 ?????

nh   28: 145
     14: 66.8
      7: 30
      4: 18
      2: 9.3
      1: 5.1

no-attn-mask: 100 ms 

query/output-size : 8192*128*28*2   ~  56 MB
KV-cache size     : 8192 * 128*4 *2 ~   8 MB
attn-mask size    : 8192 * 8192  *2 ~ 128 MB

'''
#========================================================================


src1 = cl.CMTracer.code + r'''
//# CM kernel for flash attn, reference

//# CM-compiler is C++17
static_assert(__cplusplus >= 201703L);

//# static_assert(__cplusplus >= 202002L);
//# static_assert(__cplusplus >= 202302L);

#pyeval f"#define num_heads {num_heads}"
#pyeval f"#define num_kv_heads {num_kv_heads}"
#pyeval f"#define head_size {head_size}"
#pyeval f"#define q_step {q_step}"
#pyeval f"#define kv_step {kv_step}"
#pyeval f"#define scale_factor {scale_factor}"
#pyeval f"#define args_verbose {args.verbose}"

#pyeval f"#define dim_batch {dim_batch}"
#pyeval f"#define dim_head {dim_head}"
#pyeval f"#define dim_q_batch {dim_q_batch}"

#pyeval f"#define causal_mask {causal_mask}"


#define SystolicDepth 8
#define RepeatCount 8
#define VNNI_WIDTH 2
#define REG_K (SystolicDepth * VNNI_WIDTH)
#define REG_M RepeatCount
#define REG_N (CM_GRF_WIDTH/32)

static_assert(CM_HAS_DPAS);
static_assert(q_step == 8);
static_assert(kv_step == 16);

template<typename T, int M, int N>
void show(const matrix<T, M, N> mat) {
    for(int m = 0; m < M; m ++) {
        printf("\t[");
        for(int n = 0; n < N; n ++) {
            printf("%8.4f,", mat[m][n]);
        }
        printf("],\n");
    }
    printf("]\n");
}


template <typename T1, typename T2>
CM_INLINE void Transpose_8x8(matrix_ref<T1, 8, 8> in, matrix_ref<T2, 8, 8> out) {
  matrix<T2, 8, 8> temp;
  temp.row(0) = in.template select<2, 1, 4, 2>(0, 0);
  temp.row(1) = in.template select<2, 1, 4, 2>(2, 0);
  temp.row(2) = in.template select<2, 1, 4, 2>(4, 0);
  temp.row(3) = in.template select<2, 1, 4, 2>(6, 0);
  temp.row(4) = in.template select<2, 1, 4, 2>(0, 1);
  temp.row(5) = in.template select<2, 1, 4, 2>(2, 1);
  temp.row(6) = in.template select<2, 1, 4, 2>(4, 1);
  temp.row(7) = in.template select<2, 1, 4, 2>(6, 1);

  out.row(0) = temp.template select<4, 1, 2, 4>(0, 0);
  out.row(2) = temp.template select<4, 1, 2, 4>(0, 1);
  out.row(4) = temp.template select<4, 1, 2, 4>(0, 2);
  out.row(6) = temp.template select<4, 1, 2, 4>(0, 3);
  out.row(1) = temp.template select<4, 1, 2, 4>(4, 0);
  out.row(3) = temp.template select<4, 1, 2, 4>(4, 1);
  out.row(5) = temp.template select<4, 1, 2, 4>(4, 2);
  out.row(7) = temp.template select<4, 1, 2, 4>(4, 3);
}

template <typename T, int M, int N>
CM_INLINE void svm_read_2d(matrix_ref<T, M, N> out, svmptr_t base, uint pitch) {
    #pragma unroll
    for(int i = 0; i < out.n_rows(); i++) {
        cm_svm_block_read(base + i * pitch, out[i]);
    }
}

template <typename T, int M, int N>
CM_INLINE void svm_write_2d(matrix_ref<T, M, N> out, svmptr_t base, uint pitch) {
    #pragma unroll
    for(int i = 0; i < out.n_rows(); i++, base += pitch) {
        cm_svm_block_write(base, out[i]);
    }
}

CM_INLINE uint64_t get_clock() {
    auto clk = cm_clock();
    return ((uint64_t)clk[1]) << 32 | clk[0];
}

template<bool USE_CAUSAL_MASK>
void row_kernel(
    uint slm_K,
    uint slm_V,
    int local_id,    
    int q_start,
    int kv_stop,
    int head_base_id,
    int kv_len,
    svmptr_t q_base [[type("svmptr_t")]],
    svmptr_t k_base [[type("svmptr_t")]],
    svmptr_t v_base [[type("svmptr_t")]],
    svmptr_t o_base [[type("svmptr_t")]],
    svmptr_t mask_base [[type("svmptr_t")]]) {

    uint qo_pitch = num_heads * head_size * sizeof(half);
    uint kv_pitch = num_kv_heads * head_size * sizeof(half);
    vector<float, q_step> cur_max;
    vector<float, q_step> cur_sum;

    const int local_size = cm_linear_local_size();

    cur_max = -1e9;
    cur_sum = 0;

    //# load Qt into register & pack as VNNI & store to SLM (as dpas-B tile)
    matrix<half, head_size/REG_K, REG_K*REG_N> rQ;
    {
        matrix<uint, REG_N, REG_K/2> Qmat;

        #pragma unroll
        for(int k = 0, ri = 0; k < head_size/2; k += REG_K/2, ri++, q_base += 16*sizeof(half)) {
            //# DWORD transposed load == (transposed + VNNI) load
            svm_read_2d(Qmat, q_base, qo_pitch);
            Transpose_8x8(Qmat, rQ[ri].format<uint, REG_K/2, REG_N>());
            rQ[ri].format<half>() = cm_mul<half>(rQ[ri].format<half>(), (half)scale_factor);

            //b2dQ.set_block_x(k);
            //cm_load<lsc::Transpose>(rQ[ri].format<uint>(), b2dQ.set_block_y(q_start));

            // show(rQ[ri].format<half, REG_K/2, REG_N*2>()); //# vnni
        }
    }

    matrix <float, head_size/REG_K*2, REG_M*REG_N> rO;

    int causal_left = q_start;
    for(int kv_pos = 0; kv_pos < kv_stop; kv_pos += kv_step,
            k_base += kv_step * kv_pitch,
            v_base += kv_step * kv_pitch) {
        {
            if (kv_pos > 0) cm_barrier();

            if (local_size == 1) {
                matrix<half, 2*REG_M, REG_K> temp;
                for(int k = 0; k < head_size; k += REG_K) {
                    //b2dK.set_block_x(k);
                    //cm_load<lsc::Normal>(temp0.format<half>(), b2dK.set_block_y(kv_pos));
                    //cm_prefetch(b2dK.set_block_y(kv_pos + kv_step));

                    svm_read_2d(temp, k_base + k*sizeof(half), kv_pitch);

                    // show(temp);

                    uint offset = k * 2 * REG_M * sizeof(half);
                    cm_slm_block_write(slm_K, offset, temp.format<half>());
                }

                matrix<half, REG_K, REG_N> temp2;
                matrix<half, REG_K/2, REG_N*2> temp_vnni;
                //b2dV.set_block_y(kv_pos);
                for(int k = 0; k < head_size; k += REG_N) {
                    //cm_load<lsc::VNNI>(temp2.format<half>(), b2dV.set_block_x(k).set_block_y(kv_pos));
                    //cm_prefetch(b2dV.set_block_y(kv_pos + kv_step));

                    svm_read_2d(temp2, v_base + k*sizeof(half), kv_pitch);

                    temp_vnni.select<REG_K/2, 1, REG_N, 2>(0, 0) = temp2.select<REG_K/2, 2, REG_N, 1>(0, 0);
                    temp_vnni.select<REG_K/2, 1, REG_N, 2>(0, 1) = temp2.select<REG_K/2, 2, REG_N, 1>(1, 0);

                    // show(temp_vnni);
                    cm_slm_block_write(slm_V, k * REG_K * sizeof(half), temp_vnni.format<half>());
                }
            } else {
                if (local_id < local_size/2) {
                    matrix<half, 2*REG_M, REG_K> temp;
                    for(int k = REG_K * local_id; k < head_size; k += REG_K*(local_size/2)) {
                        //b2dK.set_block_x(k);
                        //cm_load<lsc::Normal>(temp.format<half>(), b2dK.set_block_y(kv_pos));
                        //cm_prefetch(b2dK.set_block_y(kv_pos + kv_step));

                        svm_read_2d(temp, k_base + k*sizeof(half), kv_pitch);
                        // show(temp);
                        cm_slm_block_write(slm_K, k * 2 * REG_M * sizeof(half), temp.format<half>());
                    }
                } else {
                    matrix<half, REG_K, 2*REG_N> temp2;
                    matrix<half, REG_K/2, REG_N*2> temp_vnni;
                    matrix<half, REG_K/2, REG_N*2> temp_vnni2;
                    //b2dV.set_block_y(kv_pos);

                    static_assert((head_size % (2*REG_N)) == 0);

                    for(int k = 2*REG_N*(local_id-local_size/2); k < head_size; k += 2*REG_N*(local_size/2)) {
                        //cm_load<lsc::VNNI>(temp2.format<half>(), b2dV.set_block_x(k).set_block_y(kv_pos));
                        //cm_prefetch(b2dV.set_block_y(kv_pos + kv_step));

                        svm_read_2d(temp2, v_base + k*sizeof(half), kv_pitch);

                        temp_vnni.select<REG_K/2, 1, REG_N, 2>(0, 0) = temp2.select<REG_K/2, 2, REG_N, 1>(0, 0);
                        temp_vnni.select<REG_K/2, 1, REG_N, 2>(0, 1) = temp2.select<REG_K/2, 2, REG_N, 1>(1, 0);
                        // show(temp_vnni);
                        cm_slm_block_write(slm_V, k * REG_K * sizeof(half), temp_vnni.format<half>());

                        temp_vnni2.select<REG_K/2, 1, REG_N, 2>(0, 0) = temp2.select<REG_K/2, 2, REG_N, 1>(0, REG_N);
                        temp_vnni2.select<REG_K/2, 1, REG_N, 2>(0, 1) = temp2.select<REG_K/2, 2, REG_N, 1>(1, REG_N);
                        cm_slm_block_write(slm_V, (k + REG_N) * REG_K * sizeof(half), temp_vnni2.format<half>());
                    }
                }
            }
            cm_barrier();
        }

        //=========================================================== 1807 ~ 3247
        //# St = k @ Qt
        matrix<float, 2*REG_M, REG_N> St;
        auto St2 = St.format<float, 2, REG_M*REG_N>();
        #pragma unroll
        for(int k = 0, ri = 0; k < head_size; k += REG_K, ri++) {
            matrix<half, 2, REG_M * REG_K> Kmat;
            cm_slm_block_read(slm_K, GENX_NONE, ri * Kmat.n_elems() * sizeof(half), Kmat.format<half>());
            //show(Kmat.format<half, 2*REG_M, REG_K>());
            if (k == 0) {
                St2[0] = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount, float>(0, rQ[ri].format<int32_t>(), Kmat[0].format<int32_t>());
                St2[1] = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount, float>(0, rQ[ri].format<int32_t>(), Kmat[1].format<int32_t>());
            } else {
                St2[0] = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount>(St2[0], rQ[ri].format<int32_t>(), Kmat[0].format<int32_t>());
                St2[1] = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount>(St2[1], rQ[ri].format<int32_t>(), Kmat[1].format<int32_t>());
            }
        }
        //show(St);

        if constexpr (USE_CAUSAL_MASK) {
            auto cmask_off = kv_step - causal_left;
            if (cmask_off > 0) {
                //# full-attention: skip loading causal mask
                if (cmask_off > 2*kv_step) cmask_off = 2*kv_step;
                matrix<half, St.n_rows(), REG_N> temp;
                cm_svm_block_read(reinterpret_cast<svmptr_t>(mask_base + cmask_off * REG_N  * sizeof(half)), temp.format<half>());
                St = cm_add<float>(St, temp);
            }

            causal_left -= kv_step;
        } else {
            matrix<half, REG_M, REG_N + REG_N> Maskmat;
            //b2dMask.set_block_x(kv_pos);
            //cm_load<lsc::Normal>(Maskmat[0].format<half>(), b2dMask.set_block_y(q_start));
            //cm_load<lsc::Normal>(Maskmat[1].format<half>(), b2dMask.set_block_y(q_start + REG_M));
            svm_read_2d(Maskmat, mask_base + kv_pos * sizeof(half), kv_len*sizeof(half));

            matrix<float, 2*REG_M, REG_N> MaskT;
            Transpose_8x8(Maskmat.select<REG_M, 1, REG_N, 1>(0,0), MaskT.select<REG_M, 1, REG_N, 1>(0,0));
            Transpose_8x8(Maskmat.select<REG_M, 1, REG_N, 1>(0,REG_N), MaskT.select<REG_M, 1, REG_N, 1>(REG_M,0));

            //show(MaskT);
            //St = cm_mul<float>(St, (float)scale_factor);  // convert scale_factor into (float), or it will be promoted to double
            St = cm_add<float>(St, MaskT);
        }

        // show(St);

        vector<float, REG_N> new_max_t;
        new_max_t = cm_max<float>(St[0], St[1]);
        for(int r = 2; r < St.n_rows(); r++) new_max_t = cm_max<float>(new_max_t, St[r]);

        //show(new_max_t.format<float, 1, REG_N>()); return;

        new_max_t = cm_max<float>(new_max_t, cur_max);

        constexpr float log2e = 1.4426950408889634f;
        // Pt = torch.exp(St - new_max)
        for(int r = 0; r < St.n_rows(); r++) St[r] = cm_exp((St[r] - new_max_t)*log2e);

        //show(St); return;

        vector<float, REG_N> row_sum_t;
        row_sum_t = cm_add<float>(St[0], St[1]);
        for(int r = 2; r < St.n_rows(); r++) row_sum_t = cm_add<float>(row_sum_t, St[r]);

        vector<float, REG_N> max_comp;
        max_comp = cm_exp((cur_max - new_max_t)*log2e);
        cur_sum = cm_mul<float>(cur_sum, max_comp);
        cur_sum = cm_add<float>(cur_sum, row_sum_t);

        //show(cur_sum.format<float, 1, REG_N>());

        // [2*REG_M, REG_N] => [REG_M, REG_K = REG_N + REG_N]
        matrix<half, REG_M, REG_K> P; // REG_K = REG_N + REG_N
        Transpose_8x8(St.select<REG_M, 1, REG_N, 1>(0,0), P.select<REG_M, 1, REG_N, 1>(0,0));
        Transpose_8x8(St.select<REG_M, 1, REG_N, 1>(REG_M,0), P.select<REG_M, 1, REG_N, 1>(0,REG_N));

        //show(P);return;

        if (kv_pos == 0) {
            matrix<half, REG_K/2, REG_N*2> Vmat;
            #pragma unroll
            for(int k = 0, ri = 0; k < head_size; k += REG_N, ri++) {
                // V has been VNNI-prepacked
                cm_slm_block_read(slm_V, GENX_NONE, REG_K*k*sizeof(half), Vmat.format<half>());
                rO[ri] = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount, float>(0, Vmat.format<int32_t>(), P.format<int32_t>());
            }
        } else {
            matrix<half, REG_K/2, REG_N*2> Vmat;
            #pragma unroll
            for(int k = 0, ri=0; k < head_size; k += REG_N, ri++) {
                // V has been VNNI-prepacked
                cm_slm_block_read(slm_V, GENX_NONE, REG_K*k*sizeof(half), Vmat.format<half>());

                //# compensate cur_O
                //  matrix <float, head_size/REG_K*2, REG_M*REG_N> rO;
                auto cO = rO[ri].format<float, REG_M, REG_N>();
                #pragma unroll
                for(int r = 0; r < REG_M; r++)
                    cO.row(r) = cm_mul<float>(cO.row(r), max_comp[r]);

                //# show(cur_O.format<float, 2*REG_M, REG_N>()); return;
                
                rO[ri] = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount>(
                                rO[ri].format<float>(),
                                Vmat.format<int32_t>(),
                                P.format<int32_t>());
                //show(rO[ri].format<float, REG_M, REG_N>());
                // if (kv_pos == args_verbose) show(cur_O.format<float, 2*REG_M, REG_N>());
            }
            // if (kv_pos == args_verbose) return;
        }
        cur_max = new_max_t;
    }//# for(int kv_pos = 0; kv_pos < kv_len; kv_pos += kv_step) {

    //# save cur_O/cur_sum.transpose(0, 1)
    matrix<float, REG_M, REG_N> cur_O;
    matrix<half, REG_M, REG_N> cur_O_f16;
    cur_sum = cm_inv(cur_sum);
    #pragma unroll
    for(int k = 0, ri=0; k < head_size; k += REG_N, ri++) {
        auto cO = rO[ri].format<float, REG_M, REG_N>();
        for(int r = 0; r < cO.n_rows(); r++) {
            cur_O_f16[r] = cm_mul<float>(cO[r], cur_sum[r]);
        }

        // if (i == args_verbose) show(cur_O_f16);
        svm_write_2d(cur_O_f16, o_base + k*sizeof(half), qo_pitch);

        // cm_store(b2dO.set_block_x(k).set_block_y(q_start), cur_O_f16.format<half, 2, REG_M*REG_N>()[0]);
        // cm_store(b2dO.set_block_x(k).set_block_y(q_start + REG_M), cur_O_f16.format<half, 2, REG_M*REG_N>()[1]);
    }
}



extern "C" _GENX_MAIN_ void cm_sdpa(
    int head_base_id,
    int q_len,
    int kv_len,
    half* query [[type("svmptr_t")]],
    half* key [[type("svmptr_t")]],
    half* value [[type("svmptr_t")]],
    half* output [[type("svmptr_t")]],
    half* mask [[type("svmptr_t")]],
    __global uint64_t* cminfo [[type("svmptr_t")]]
    ) {
    CMTracer_begin(&cminfo);
    //# query [batch, q_len, num_heads, S]
    //#   key [batch, kv_len, num_heads, S]
    //# value [batch, kv_len, num_heads, S]
    //# to load Q

    constexpr uint K_SLM_SIZE = (kv_step * head_size * sizeof(half));
    constexpr uint V_SLM_SIZE = (kv_step * head_size * sizeof(half));

    cm_slm_init(K_SLM_SIZE + V_SLM_SIZE);

    auto slm_K = cm_slm_alloc(K_SLM_SIZE);
    auto slm_V = cm_slm_alloc(V_SLM_SIZE);

    auto batch = cm_group_id(dim_batch);
    auto h = head_base_id + cm_group_id(dim_head);
    auto hkv = h / (num_heads/num_kv_heads);
    auto local_id = cm_local_id(dim_q_batch);
    int q_group_id = cm_group_id(dim_q_batch);
    const int local_size = cm_linear_local_size();
    int kv_stop = kv_len;
#if causal_mask
    kv_stop = (q_group_id + 1) * local_size * q_step;
    if (kv_stop > kv_len) kv_stop = kv_len;
#endif
    {
        auto q_start = (q_group_id * local_size + local_id) * q_step;
        auto q_base = reinterpret_cast<svmptr_t>(query + ((batch*q_len + q_start)*num_heads + h)*head_size);
        auto k_base = reinterpret_cast<svmptr_t>(key + (batch*num_kv_heads*kv_len + hkv)*head_size);
        auto v_base = reinterpret_cast<svmptr_t>(value + (batch*num_kv_heads*kv_len + hkv)*head_size);
        auto o_base = reinterpret_cast<svmptr_t>(output + ((batch * q_len + q_start)*num_heads + h)*head_size);
        svmptr_t mask_base;
        if (causal_mask)
            mask_base = reinterpret_cast<svmptr_t>(mask);
        else
            mask_base = reinterpret_cast<svmptr_t>(mask + (batch * q_len + q_start) * kv_len);

        row_kernel<causal_mask>(
            slm_K,
            slm_V,
            local_id,
            q_start,
            kv_stop,
            head_base_id,
            kv_len,
            q_base,
            k_base,
            v_base,
            o_base,
            mask_base);
    }

    {
        auto q_start = (q_len - q_group_id * local_size * q_step) - (local_size * q_step) + local_id*q_step;
        auto q_base = reinterpret_cast<svmptr_t>(query + ((batch*q_len + q_start)*num_heads + h)*head_size);
        auto k_base = reinterpret_cast<svmptr_t>(key + (batch*num_kv_heads*kv_len + hkv)*head_size);
        auto v_base = reinterpret_cast<svmptr_t>(value + (batch*num_kv_heads*kv_len + hkv)*head_size);
        auto o_base = reinterpret_cast<svmptr_t>(output + ((batch * q_len + q_start)*num_heads + h)*head_size);
        svmptr_t mask_base;
        if (causal_mask)
            mask_base = reinterpret_cast<svmptr_t>(mask);
        else
            mask_base = reinterpret_cast<svmptr_t>(mask + (batch * q_len + q_start) * kv_len);

    #if causal_mask
        kv_stop = (q_len - q_group_id * local_size * q_step);
        if (kv_stop > kv_len) kv_stop = kv_len;
    #endif
        row_kernel<causal_mask>(
            slm_K,
            slm_V,
            local_id,
            q_start,
            kv_stop,
            head_base_id,
            kv_len,
            q_base,
            k_base,
            v_base,
            o_base,
            mask_base);
    }

    CMTracer_end(&cminfo);
}
'''

cl.profiling(True)

t_q = cl.tensor(q.detach().numpy())
t_k = cl.tensor(k.detach().numpy())
t_v = cl.tensor(v.detach().numpy())
t_out = cl.tensor([batch, q_len, num_heads, head_size], np.dtype(np.float16))
t_cminfo = cl.tensor([GWS[0]*GWS[1]*GWS[2]//WG_SIZE, 3], np.dtype(np.uint64))

if causal_mask:
    # build a [q_step, 2*kv_step] causal mask
    c_mask = torch.full([3*kv_step, q_step], torch.finfo(act_dtype).min).to(dtype=act_dtype)
    for i in range(q_step):
        c_mask[0:(kv_step + i + 1), i] = 0
    t_mask = cl.tensor(c_mask.detach().numpy())
else:
    t_mask = cl.tensor(attention_mask.detach().numpy())

# f"-cmc -mdump_asm -g2 "
print("compiling ...")
cm_kernels = cl.kernels(pyeval(src1), f"-cmc -Qxcm_register_file_size=256 -mdump_asm -g2")
print("first call ...")


cm_kernels.enqueue("cm_sdpa", GWS, LWS, 0, q_len, kv_len, t_q, t_k, t_v, t_out, t_mask, t_cminfo)

f1 = torch.from_numpy(t_out.numpy())

all_layers = []
mem_size = 0
while len(all_layers) < 100 and mem_size < 8e9:
    all_layers.append([
        cl.tensor(q.detach().numpy()),
        cl.tensor(k.detach().numpy()),
        cl.tensor(v.detach().numpy()),
        cl.tensor([batch, q_len, num_heads, head_size], np.dtype(np.float16)),
    ])
    mem_size += q.numel() * q.element_size()
    mem_size += k.numel() * k.element_size()
    mem_size += v.numel() * v.element_size()
    print(f"nlayers={len(all_layers)} mem_size={mem_size*1e-9:.3f} GB")

for i in range(100):
    j  = i % len(all_layers)
    j = 0
    cm_kernels.enqueue("cm_sdpa", GWS, LWS,
                    0, q_len, kv_len,
                    all_layers[j][0],
                    all_layers[j][1],
                    all_layers[j][2],
                    all_layers[j][3],
                    t_mask,
                    t_cminfo)
latency = cl.finish()
for i,ns in enumerate(latency):
    print(f"[{i}]  {ns*1e-6:.3f} ms")
print(f" average latency: {sum(latency[10:])/len(latency[10:])*1e-6:.3f} ms")

cl.CMTracer.dump(t_cminfo.numpy(), 2250e6, "cm.json")

check_close(org.transpose(1,2), f1, atol=1e-2, rtol=1e-3)
print(f"=========== cm_sdpa PASS GWS={GWS} LWS={LWS}  ===========")
sys.exit(0)
