import argparse
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)
torch.set_printoptions(linewidth=1024)

parser = argparse.ArgumentParser('')
parser.add_argument('-i', "--impl", type=int, default=0)
parser.add_argument('-b', "--batch", type=int, default=1)
parser.add_argument('-nh', "--num-heads", type=int, default=16)
parser.add_argument('-ql', "--q-len", type=int, default=256)
parser.add_argument('-kvl', "--kv-len", type=int, default=512)
parser.add_argument('-hs', "--head-size", type=int, default=80)
parser.add_argument('-v', "--verbose", type=int, default=-1)

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
q_len, q_step = args.q_len, 16
kv_len, kv_step = args.kv_len, 16
num_heads = args.num_heads
head_size = args.head_size

#q_len, q_step = 160, 16
#kv_len, kv_step = 800, 16
low = -7
high = 8
act_dtype = torch.float16
q = torch.randint(low, high, [batch, q_len, num_heads, head_size]).to(dtype=act_dtype)/high
k = torch.randint(low, high, [batch, kv_len, num_heads, head_size]).to(dtype=act_dtype)/high
v = torch.randint(low, high, [batch, kv_len, num_heads, head_size]).to(dtype=act_dtype)/high

# random attnmask
attention_mask = torch.full([batch, 1, q_len, kv_len], torch.finfo(act_dtype).min)
attention_mask[torch.rand(batch, 1, q_len, kv_len) > 0.5] = 0

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
    out = torch.zeros([B,H,L,S], dtype=Q.dtype)
    scale_factor = S**(-0.5)
    for b in range(B):
        for h in range(H):
            attn_score = Q[b, h, :, :].to(dtype=torch.float32) @ (K[b, h, :,:].transpose(0,1)).to(dtype=torch.float32)
            attn_score *= scale_factor
            attn_score += attention_mask[b,0,:,:]
            #print(attn_score.shape)
            attn_weights = F.softmax(attn_score, 1)
            out[b,h,:,:] = attn_weights @ V[b, h, :, :].to(dtype=torch.float32)
    return out

ref = F.scaled_dot_product_attention(q, k, v, attention_mask, dropout_p=0.0)
org = get_org(q,k,v,attention_mask)

def check_close(input, other, atol=1e-3, rtol=1e-3):
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
    _,_,kv_len,_ = key.shape
    out = torch.zeros([B,H,q_len,hs], dtype=value.dtype)
    scale_factor = hs**(-0.5)
    for b in range(B):
        for h in range(H):
            Q = query[b, h, :, :]
            K = key[b, h, :, :]
            V = value[b, h, :, :]
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

                    # if (j == args.verbose): enable_vprint = True
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
                    vprint("cur_O2=", cur_O)

                    cur_max = rowmax

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

src1 = r'''
//# CM kernel for flash attn, reference

#pyeval f"#define num_heads {num_heads}"
#pyeval f"#define head_size {head_size}"
#pyeval f"#define q_step {q_step}"
#pyeval f"#define kv_step {kv_step}"
#pyeval f"#define scale_factor {scale_factor}"
#pyeval f"#define args_verbose {args.verbose}"


#define SystolicDepth 8
#define RepeatCount 8
#define VNNI_WIDTH 2
#define REG_K (SystolicDepth * VNNI_WIDTH)
#define REG_M RepeatCount
#define REG_N 16

static_assert(q_step == 16);
static_assert(kv_step == 16);

template<int num_REGM, int num_REGN>
void ugemm(matrix_ref<float, num_REGM * num_REGN, REG_M*REG_N> res,
           lsc::block_2d_desc<half, 1, REG_M, REG_K>& b2dA,
           lsc::block_2d_desc<half, 1, REG_K, REG_N>& b2dB,
           int m0, int n0, int total_K) {
    matrix<half, num_REGM, REG_M*REG_K> Amat;
    matrix<half, num_REGN, REG_K*REG_N> Bmat;

    for(int k = 0; k < total_K; k += REG_K) {
        b2dA.set_block_x(k);
        b2dB.set_block_y(k);

        #pragma unroll (num_REGM)
        for(int m = 0; m < num_REGM; m++) {
            cm_load<lsc::Normal>(Amat[m], b2dA.set_block_y(m0 + m*REG_M));
        }

        #pragma unroll (num_REGN)
        for(int n = 0; n < num_REGN; n++) {
            cm_load<lsc::VNNI>(Bmat[n], b2dB.set_block_x(n0 + n*REG_N));
        }

        #pragma unroll (num_REGM)
        for(int m = 0; m < num_REGM; m++) {
            #pragma unroll (num_REGN)
            for(int n = 0; n < num_REGN; n++) {
                auto cid = m*num_REGN + n;
                res[cid] = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount>(
                            res[cid].format<float>(),
                            Bmat[n].format<int32_t>(),
                            Amat[m].format<int32_t>());
            }
        }
    }
}

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
CM_INLINE void Transpose_16x16(matrix_ref<T1, 16, 16> in,
                               matrix_ref<T2, 16, 16> out) {
  matrix<T2, 16, 16> bBuf;
  bBuf.row(0) = in.template select<4, 1, 4, 4>(0, 0);   // 0,4,8,c
  bBuf.row(1) = in.template select<4, 1, 4, 4>(4, 0);   // 0,4,8,c
  bBuf.row(2) = in.template select<4, 1, 4, 4>(8, 0);   // 0,4,8,c
  bBuf.row(3) = in.template select<4, 1, 4, 4>(12, 0);  // 0,4,8,c
  bBuf.row(4) = in.template select<4, 1, 4, 4>(0, 1);   // 1,5,9,d
  bBuf.row(5) = in.template select<4, 1, 4, 4>(4, 1);   // 1,5,9,d
  bBuf.row(6) = in.template select<4, 1, 4, 4>(8, 1);   // 1,5,9,d
  bBuf.row(7) = in.template select<4, 1, 4, 4>(12, 1);  // 1,5,9,d
  bBuf.row(8) = in.template select<4, 1, 4, 4>(0, 2);   // 2,6,a,e
  bBuf.row(9) = in.template select<4, 1, 4, 4>(4, 2);   // 2,6,a,e
  bBuf.row(10) = in.template select<4, 1, 4, 4>(8, 2);  // 2,6,a,e
  bBuf.row(11) = in.template select<4, 1, 4, 4>(12, 2); // 2,6,a,e
  bBuf.row(12) = in.template select<4, 1, 4, 4>(0, 3);  // 3,7,b,f
  bBuf.row(13) = in.template select<4, 1, 4, 4>(4, 3);  // 3,7,b,f
  bBuf.row(14) = in.template select<4, 1, 4, 4>(8, 3);  // 3,7,b,f
  bBuf.row(15) = in.template select<4, 1, 4, 4>(12, 3); // 3,7,b,f

  out.row(0) = bBuf.template select<4, 1, 4, 4>(0, 0);   // 0
  out.row(1) = bBuf.template select<4, 1, 4, 4>(4, 0);   // 1
  out.row(2) = bBuf.template select<4, 1, 4, 4>(8, 0);   // 2
  out.row(3) = bBuf.template select<4, 1, 4, 4>(12, 0);  // 3
  out.row(4) = bBuf.template select<4, 1, 4, 4>(0, 1);   // 4
  out.row(5) = bBuf.template select<4, 1, 4, 4>(4, 1);   // 5
  out.row(6) = bBuf.template select<4, 1, 4, 4>(8, 1);   // 6
  out.row(7) = bBuf.template select<4, 1, 4, 4>(12, 1);  // 7
  out.row(8) = bBuf.template select<4, 1, 4, 4>(0, 2);   // 8
  out.row(9) = bBuf.template select<4, 1, 4, 4>(4, 2);   // 9
  out.row(10) = bBuf.template select<4, 1, 4, 4>(8, 2);  // a
  out.row(11) = bBuf.template select<4, 1, 4, 4>(12, 2); // b
  out.row(12) = bBuf.template select<4, 1, 4, 4>(0, 3);  // c
  out.row(13) = bBuf.template select<4, 1, 4, 4>(4, 3);  // d
  out.row(14) = bBuf.template select<4, 1, 4, 4>(8, 3);  // e
  out.row(15) = bBuf.template select<4, 1, 4, 4>(12, 3); // f
}

extern "C" _GENX_MAIN_ void cm_sdpa(
    int batch,
    int q_len,
    int kv_len,
    half* query [[type("svmptr_t")]],
    half* key [[type("svmptr_t")]],
    half* value [[type("svmptr_t")]],
    half* output [[type("svmptr_t")]],
    float* mask [[type("svmptr_t")]]
    ) {
    //# query [batch, q_len, num_heads, S]
    //#   key [batch, kv_len, num_heads, S]
    //# value [batch, kv_len, num_heads, S]
    //# to load Q

    constexpr uint nthr_per_WG = 1;
    constexpr uint Q_SLM_SIZE = (q_step * head_size * sizeof(half)) * nthr_per_WG;
    constexpr uint O_SLM_SIZE = (q_step * head_size * sizeof(float)) * nthr_per_WG;
    constexpr uint K_SLM_SIZE = (kv_step * head_size * sizeof(half));
    constexpr uint V_SLM_SIZE = (kv_step * head_size * sizeof(half));

    cm_slm_init(K_SLM_SIZE + V_SLM_SIZE + Q_SLM_SIZE + O_SLM_SIZE);

    auto slm_K = cm_slm_alloc(K_SLM_SIZE);
    auto slm_V = cm_slm_alloc(V_SLM_SIZE);
    auto slm_Q = cm_slm_alloc(Q_SLM_SIZE);
    auto slm_O = cm_slm_alloc(O_SLM_SIZE);

    vector<float, q_step> cur_max;
    vector<float, q_step> cur_sum;

    //# debugging stage
    for(int b = 0; b < batch; b++) {
        lsc::block_2d_desc<float, 1, REG_N, REG_M> b2dMask(mask, q_len - 1, kv_len*sizeof(float) - 1, kv_len*sizeof(float) - 1, 0, 0);

        for(int h = 0; h < num_heads; h++) {
            //# b2dQ reinterpret as 32bit(DWORD) for transposed load(combined with VNNI)
            lsc::block_2d_desc<uint, 1, REG_N, REG_K/2> b2dQ(
                           reinterpret_cast<uint*>(query + h*head_size),
                           q_len - 1,
                           head_size*sizeof(half) - 1,
                           num_heads*head_size*sizeof(half) - 1, 0, 0);
            lsc::block_2d_desc<half, 1, REG_M, REG_K> b2dK(key + h*head_size,   kv_len - 1, head_size*sizeof(half) - 1, num_heads*head_size*sizeof(half) - 1, 0, 0);
            lsc::block_2d_desc<half, 1, REG_K, REG_N> b2dV(value + h*head_size, kv_len - 1, head_size*sizeof(half) - 1, num_heads*head_size*sizeof(half) - 1, 0, 0);
            lsc::block_2d_desc<half, 1, REG_M, REG_N> b2dO(output + h*head_size,   q_len - 1, head_size*sizeof(half) - 1, num_heads*head_size*sizeof(half) - 1, 0, 0);

            for(int i = 0; i < q_len; i += q_step) {
                //# load Qt into register & pack as VNNI & store to SLM (as dpas-B tile)
                {
                    matrix<uint, REG_K/2, REG_N> Qmat;
                    for(int k = 0; k < head_size/2; k += REG_K/2) {
                        b2dQ.set_block_x(k);
                        //# DWORD transposed load == (transposed + VNNI) load
                        cm_load<lsc::Transpose>(Qmat.format<uint>(), b2dQ.set_block_y(i*REG_N));
                        //# show(Qmat.format<half, REG_K/2, REG_N*2>());
                        cm_slm_block_write(slm_Q, k*q_step*sizeof(uint), Qmat.format<uint>());
                    }
                }

                b2dMask.set_block_y(i);

                for(int j = 0; j < kv_len; j += kv_step) {
                    if (args_verbose >= 0)
                        printf("======== %d =========\n", j);

                    //# load K into SLM as dpas-A tile (shared by all hw within same WG)
                    //# load V into SLM as dpas-B tile (shared by all hw within same WG)
                    {
                        matrix<half, REG_M, REG_K> temp0;
                        matrix<half, REG_M, REG_K> temp1;
                        uint offset = 0;
                        for(int k = 0; k < head_size; k += REG_K) {
                            b2dK.set_block_x(k);
                            cm_load<lsc::Normal>(temp0.format<half>(), b2dK.set_block_y(j));
                            cm_load<lsc::Normal>(temp1.format<half>(), b2dK.set_block_y(j + REG_M));
                            cm_slm_block_write(slm_K, offset, temp0.format<half>()); offset += REG_M * REG_K * sizeof(half);
                            cm_slm_block_write(slm_K, offset, temp1.format<half>()); offset += REG_M * REG_K * sizeof(half);
                        }
                        matrix<half, REG_K, REG_N> temp2;
                        offset = 0;
                        b2dV.set_block_y(j);
                        for(int k = 0; k < head_size; k += REG_K) {
                            cm_load<lsc::VNNI>(temp2.format<half>(), b2dV.set_block_x(k));
                            cm_slm_block_write(slm_V, offset, temp2.format<half>()); offset += REG_K * REG_N * sizeof(half);
                        }
                    }
                    //# St = k @ Qt
                    matrix<float, 2*REG_M, REG_N> St = 0;
                    matrix<half, 2, REG_M * REG_K> Kmat;
                    matrix<half, REG_K/2, REG_N*2> Qmat;
                    auto St2 = St.format<float, 2, REG_M*REG_N>();
                    uint offset = 0;
                    for(int k = 0; k < head_size; k += REG_K) {
                        cm_slm_block_read(slm_K, GENX_NONE, offset, Kmat[0]); offset += REG_M * REG_K * sizeof(half);
                        cm_slm_block_read(slm_K, GENX_NONE, offset, Kmat[1]); offset += REG_M * REG_K * sizeof(half);
                        //show(Kmat.format<half, 2*REG_M, REG_K>());
                        
                        cm_slm_block_read(slm_Q, GENX_NONE, k*q_step*sizeof(half), Qmat.format<half>());
                        //show(Qmat);

                        St2[0] = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount>(
                                    St2[0],
                                    Qmat.format<int32_t>(),
                                    Kmat[0].format<int32_t>());
                        St2[1] = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount>(
                                    St2[1],
                                    Qmat.format<int32_t>(),
                                    Kmat[1].format<int32_t>());
                    }
                    //show(St);
                    matrix<float, 2, REG_M * REG_N> Maskmat;
                    cm_load<lsc::Transpose>(Maskmat[0].format<float>(), b2dMask.set_block_x(j));
                    cm_load<lsc::Transpose>(Maskmat[1].format<float>(), b2dMask.set_block_x(j + REG_M));
                    //show(Maskmat);
                    St *= scale_factor;
                    St2[0] += Maskmat[0];
                    St2[1] += Maskmat[1];
                    //show(St);

                    vector<float, REG_N> new_max_t;
                    new_max_t = cm_max<float>(St[0], St[1]);
                    for(int i = 2; i < St.n_rows(); i++) {
                        new_max_t = cm_max<float>(new_max_t, St[i]);
                    }
                    //show(new_max_t.format<float, 1, REG_N>()); return;

                    if (j == 0) {
                        cur_max = new_max_t;
                    } else {
                        new_max_t = cm_max<float>(new_max_t, cur_max);
                    }

                    constexpr float log2e = 1.4426950408889634f;
                    // Pt = torch.exp(St - new_max)
                    for(int i = 0; i < St.n_rows(); i++) {
                        St[i] = cm_exp((St[i] - new_max_t)*log2e);
                    }
                    //show(St); return;
                    
                    vector<float, REG_N> row_sum_t;
                    row_sum_t = cm_add<float>(St[0], St[1]);
                    for(int i = 2; i < St.n_rows(); i++) {
                        row_sum_t = cm_add<float>(row_sum_t, St[i]);
                    }

                    vector<float, REG_N> max_comp;
                    if (j == 0) {
                        cur_sum = row_sum_t;
                    } else {
                        max_comp = cm_exp((cur_max - new_max_t)*log2e);
                        cur_sum = cm_mul<float>(cur_sum, max_comp);
                        cur_sum = cm_add<float>(cur_sum, row_sum_t);
                    }
                    //show(cur_sum.format<float, 1, REG_N>()); return;

                    matrix<half, 2*REG_M, REG_K> P;
                    Transpose_16x16(St, P);

                    //show(P);return;
                    
                    auto P2 = P.format<half, 2, REG_M * REG_K>();
                    matrix<float, 2, REG_M*REG_N> cur_O;
                    if (j == 0) {
                        matrix<float, REG_M, REG_N> zero_O = 0;
                        matrix<half, REG_K/2, REG_N*2> Vmat;
                        uint offset = 0;
                        for(int k = 0; k < head_size; k += REG_K) {
                            // V has been VNNI-prepacked
                            cm_slm_block_read(slm_V, GENX_NONE, REG_N*k*sizeof(half), Vmat.format<half>());
                            
                            cur_O[0] = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount>(
                                            zero_O.format<float>(),
                                            Vmat.format<int32_t>(),
                                            P2[0].format<int32_t>());
                            cur_O[1] = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount>(
                                            zero_O.format<float>(),
                                            Vmat.format<int32_t>(),
                                            P2[1].format<int32_t>());
                            cm_slm_block_write(slm_O, offset, cur_O[0]); offset += REG_M * REG_N * sizeof(float);
                            cm_slm_block_write(slm_O, offset, cur_O[1]); offset += REG_M * REG_N * sizeof(float);
                            //show(cur_O.format<float, 2*REG_M, REG_N>());
                        }
                    } else {
                        matrix<half, REG_K/2, REG_N*2> Vmat;
                        uint offset = 0;
                        for(int k = 0; k < head_size; k += REG_K) {
                            // V has been VNNI-prepacked
                            cm_slm_block_read(slm_V, GENX_NONE, REG_N*k*sizeof(half), Vmat.format<half>());

                            cm_slm_block_read(slm_O, GENX_NONE, offset,
                                              cur_O[0].format<half>());
                            cm_slm_block_read(slm_O, GENX_NONE, offset + REG_M * REG_N * sizeof(float),
                                              cur_O[1].format<half>());
                            //# compensate cur_O
                            auto cO = cur_O.format<float, 2*REG_M, REG_N>();
                            for(int r = 0; r < cO.n_rows(); r++) {
                                cO[r] *= max_comp[r];
                            }
                            //# show(cur_O.format<float, 2*REG_M, REG_N>()); return;
                            
                            cur_O[0] = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount>(
                                            cur_O[0].format<float>(),
                                            Vmat.format<int32_t>(),
                                            P2[0].format<int32_t>());
                            cur_O[1] = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount>(
                                            cur_O[1].format<float>(),
                                            Vmat.format<int32_t>(),
                                            P2[1].format<int32_t>());
                            cm_slm_block_write(slm_O, offset, cur_O[0]); offset += REG_M * REG_N * sizeof(float);
                            cm_slm_block_write(slm_O, offset, cur_O[1]); offset += REG_M * REG_N * sizeof(float);
                            // if (j == args_verbose) show(cur_O.format<float, 2*REG_M, REG_N>());
                        }
                        // if (j == args_verbose) return;
                    }
                    cur_max = new_max_t;
                }//# for(int j = 0; j < kv_len; j += kv_step) {
                
                //# save cur_O/cur_sum.transpose(0, 1)
                matrix<float, 2, REG_M*REG_N> cur_O;
                matrix<half, 2*REG_M, REG_N> cur_O_f16;
                uint offset = 0;
                for(int k = 0; k < head_size; k += REG_K) {
                    cm_slm_block_read(slm_O, GENX_NONE, offset, cur_O[0].format<half>()); offset += REG_M * REG_N * sizeof(float);
                    cm_slm_block_read(slm_O, GENX_NONE, offset, cur_O[1].format<half>()); offset += REG_M * REG_N * sizeof(float);

                    auto cO = cur_O.format<float, 2*REG_M, REG_N>();
                    for(int r = 0; r < cO.n_rows(); r++) {
                        cur_O_f16[r] = cm_div_ieee(cO[r], cur_sum[r]);
                    }
                    
                    if (i == args_verbose) {
                        show(cur_O_f16);
                    }

                    cm_store(b2dO.set_block_x(k).set_block_y(i), cur_O_f16.format<half, 2, REG_M*REG_N>()[0]);
                    cm_store(b2dO.set_block_x(k).set_block_y(i + REG_M), cur_O_f16.format<half, 2, REG_M*REG_N>()[1]);
                }
                if (i == args_verbose) return;
            }//# for(int i = 0; i < q_len; i += q_step) {
        }
    }
}
'''

GWS=[1]
LWS=[1]
t_q = cl.tensor(q.detach().numpy())
t_k = cl.tensor(k.detach().numpy())
t_v = cl.tensor(v.detach().numpy())
t_out = cl.tensor([batch, q_len, num_heads, head_size], np.dtype(np.float16))
t_mask = cl.tensor(attention_mask.detach().numpy())

cm_kernels = cl.kernels(pyeval(src1), f"-cmc -mdump_asm -g2 ")
cm_kernels.enqueue("cm_sdpa", GWS, LWS, batch, q_len, kv_len, t_q, t_k, t_v, t_out, t_mask)
