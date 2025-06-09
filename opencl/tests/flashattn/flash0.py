import argparse
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)
torch.set_printoptions(linewidth=1024)

parser = argparse.ArgumentParser('')
parser.add_argument('-i', "--impl", type=int, default=0)
parser.add_argument('-b', "--batch", type=int, default=2)
parser.add_argument('-nh', "--num-heads", type=int, default=64)
parser.add_argument('-nkvh', "--num-kv-heads", type=int, default=16)
parser.add_argument('-ql', "--q-len", type=int, default=16)
parser.add_argument('-kvl', "--kv-len", type=int, default=32)
parser.add_argument('-hs', "--head-size", type=int, default=32)
parser.add_argument('-v', "--verbose", type=int, default=-1)

#parser.add_argument('-q', "--quant_type", type=str, default="w4a", choices=['f16', 'f16b1', 'w4a', 'w4a_cpu', 'f16xmx', 'w4x'])
#parser.add_argument('-hf', '--hf_model_path', type=str, nargs='?', default='/mnt/llm_irs/models_original/Qwen2-0.5B-Instruct/')
#parser.add_argument('--save', type=str, nargs='?', default=None)
#parser.add_argument('--load', type=str, nargs='?', default=None)
args = parser.parse_args()
print(args)

enable_vprint = True
def vprint(*all_args):
    global enable_vprint
    if enable_vprint:
        print(*all_args)


batch = args.batch
q_len, q_step = args.q_len, 16
kv_len, kv_step = args.kv_len, 16
num_heads = args.num_heads
num_kv_heads = args.num_kv_heads
head_size = args.head_size
enable_gqa = num_heads > num_kv_heads

#q_len, q_step = 160, 16
#kv_len, kv_step = 800, 16
low = -7
high = 8
act_dtype = torch.float16
q = torch.randint(low, high, [batch, q_len, num_heads, head_size]).to(dtype=act_dtype)/high
k = torch.randint(low, high, [batch, kv_len, num_kv_heads, head_size]).to(dtype=act_dtype)/high
v = torch.randint(low, high, [batch, kv_len, num_kv_heads, head_size]).to(dtype=act_dtype)/high
torch.set_printoptions(precision=4, sci_mode=False)
# random attnmask
attention_mask = torch.full([batch, 1, q_len, kv_len], torch.finfo(act_dtype).min).to(dtype=act_dtype)
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
            O = out[b, h, :, :]
            mask = attention_mask[b,0,:,:]
            for i in range(0, q_len, q_step):
                br = min(q_len - i, q_step)
                sQ_tr = Q[i:(i+br), :].transpose(0,1)
                sO = O[i:(i+br), :]
                for j in range(0, kv_len, kv_step):
                    bc = min(kv_len -j, kv_step)
                    sK = K[j:(j+bc), :]
                    sV = V[j:(j+bc), :]
                    sMask_tr = mask[i:(i+br), j:(j+bc)].transpose(0,1)
                    # [Bc, Br]
                    S_tr = (sK @ sQ_tr).to(dtype=torch.float32)
                    if 1:
                        S_tr *= scale_factor
                        S_tr += sMask_tr
                        # [1, Br]
                        rowMax = torch.max(S_tr, 0, keepdim=True).values
                        if j == 0:
                            lastMax = rowMax
                        rowMax = torch.maximum(lastMax, rowMax)

                        max_comp = torch.exp(lastMax - rowMax)
                        P_tr = torch.exp(S_tr-rowMax)
                        # [1, Br]
                        tempSum = torch.sum(P_tr, 0, keepdim=True)
                        if j == 0:
                            rowSum = tempSum
                        else:
                            rowSum = max_comp* rowSum + tempSum
                        sP = P_tr.transpose(0,1).to(dtype=torch.float16)
                        if j == 0:
                            sO = sP @ sV
                        else:
                            sO = sP @ sV + sO*max_comp.transpose(0, 1)
                        lastMax = rowMax
                    else:
                        S_tr *= scale_factor
                        S_tr += sMask_tr
                        rowMax = torch.max(S_tr, 0, keepdim=True).values
                        if j == 0:
                            lastMax = rowMax
                        rowMax = torch.maximum(lastMax, rowMax)
                        max_comp = torch.exp(lastMax - rowMax)
                        P_tr = torch.exp(S_tr-rowMax)
                        # [1, Br]
                        tempSum = torch.sum(P_tr, 0, keepdim=True)
                        # print("------------------------------------------------------------------")
                        # print(P_tr)
                        # print(rowMax)
                        # print(tempSum)
                        if j == 0:
                            rowSum = tempSum
                        else:
                            rowSum = max_comp* rowSum + tempSum

                        sP = P_tr.transpose(0,1).to(dtype=torch.float16)

                        # print(sP)
                        if j == 0:
                            sO = sP @ sV
                        else:
                            sO = sP @ sV + sO*max_comp.transpose(0, 1)
                        lastMax = rowMax
                        print("------------------------------------------------------------------")


                if 1:
                    sO = sO / rowSum.transpose(0,1).to(dtype=torch.float16)
                    out[b, h, i:(i+br), :] = sO
                    #print(sO)

    return out

if args.impl == 0:
    f0 = get_flash0(q,k,v,attention_mask)
    # check_close(org, f0, atol=1e-2, rtol=1e-3)
    print("=========== PASS ===========")

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

GWS=[batch, num_heads, q_len//q_step]
WG_SIZE = min(q_len//q_step, 8)
LWS=[1, 1, WG_SIZE]

print("GWS=", GWS)
print("LWS=", LWS)

#========================================================================
# Optimization Log
r'''
increase WG_SIZE to 16 (-70ms)
use GRF to store temp Output(rO) instead of SLM, requires `-Qxcm_register_file_size=256` (-40ms)
avoid type-promotion:   St = cm_mul<float>(St, scale_factor);    =>   St = cm_mul<float>(St, (float)scale_factor);   (-10ms)
change dtype of attention_mask from float to half (-14ms)
avoid indirect register access: unroll for loop which access matrix rows using loop-index
use GRF to store temp Input rQ instead of SLM, this allows more Work-Groups to be packed into same Xe-core!!!
'''
#========================================================================

src1 = r'''
//# CM kernel for flash attn, reference

#pyeval f"#define num_heads {num_heads}"
#pyeval f"#define num_kv_heads {num_kv_heads}"
#pyeval f"#define head_size {head_size}"
#pyeval f"#define q_step {q_step}"
#pyeval f"#define kv_step {kv_step}"
#pyeval f"#define scale_factor {scale_factor}"
#pyeval f"#define args_verbose {args.verbose}"
#pyeval f"#define WG_SIZE {WG_SIZE}"

#define SystolicDepth 8
#define RepeatCount 8
#define VNNI_WIDTH 2
#define REG_K (SystolicDepth * VNNI_WIDTH)
#define REG_M RepeatCount
#define REG_N 16

static_assert(q_step == 16);
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

CM_INLINE uint64_t get_clock() {
    auto clk = cm_clock();
    return ((uint64_t)clk[1]) << 32 | clk[0];
}

extern "C" _GENX_MAIN_ void cm_sdpa(
    int q_len,
    int kv_len,
    half* query [[type("svmptr_t")]],
    half* key [[type("svmptr_t")]],
    half* value [[type("svmptr_t")]],
    half* output [[type("svmptr_t")]],
    half* mask [[type("svmptr_t")]]
    ) {
    //# query [batch, q_len, num_heads, S]
    //#   key [batch, kv_len, num_heads, S]
    //# value [batch, kv_len, num_heads, S]
    //# to load Q

    constexpr uint K_SLM_SIZE = (kv_step * head_size * sizeof(half));
    constexpr uint V_SLM_SIZE = (kv_step * head_size * sizeof(half));
    constexpr uint Q_SLM_SIZE = 0;
    cm_slm_init(K_SLM_SIZE + V_SLM_SIZE + Q_SLM_SIZE);

    auto slm_K = cm_slm_alloc(K_SLM_SIZE);
    auto slm_V = cm_slm_alloc(V_SLM_SIZE);

    vector<float, q_step> cur_max;
    vector<float, q_step> cur_sum;

    cur_max = -1e9;
    cur_sum = 0;

    auto batch = cm_group_id(0);
    auto h = cm_group_id(1);
    auto hkv = h / (num_heads/num_kv_heads);
    auto wg_local_id = cm_local_id(2);
    auto wg_size = cm_local_size (2);
    //# query [batch, q_len, num_heads, S]
    auto q_start = (cm_group_id(2) * wg_size + wg_local_id) * q_step;
    auto q_offset = (batch * q_len * num_heads + h) * head_size;
    auto kv_offset = (batch * kv_len * num_kv_heads + hkv) * head_size;


    auto o_offset = wg_local_id * q_step * head_size * sizeof(float);

    auto q_pitch = num_heads * head_size * sizeof(half);
    auto kv_pitch = num_kv_heads * head_size * sizeof(half);;


    // Q: [q_step, head_size], K: [k_step, head_size] Q_TR: [head_size, q_step]
    // REGM = k_step / 2, REGN = q_step, regK =16

    //# query [batch, q_len, num_heads, S], rect window [REG_N, REG_K/2]
    lsc::block_2d_desc<uint, 1, REG_N, REG_K/2> q_desc(reinterpret_cast<uint*>(query + q_offset), q_len - 1, head_size*sizeof(half)-1,  q_pitch - 1, q_start, 0);
    lsc::block_2d_desc<half, 1, REG_M, REG_K> k_desc((half*)(key + kv_offset), kv_len - 1, head_size*sizeof(half)-1, kv_pitch - 1, 0, 0);
    lsc::block_2d_desc<half, 1, REG_K, REG_N> v_desc((half*)(value + kv_offset), kv_len - 1, head_size*sizeof(half)-1, kv_pitch - 1, 0, 0);
    // #attention_mask [batch, 1, q_len, kv_len]
    lsc::block_2d_desc<half, 1, 2*REG_M, REG_N> atten_desc((half*)(mask + batch*q_len*kv_len),  q_len - 1, kv_len*sizeof(half)-1, kv_len*sizeof(half)-1, q_start, 0);



    matrix<half, head_size/REG_K, REG_K*REG_N> mat_Qtr;
    matrix<float, 2*head_size/REG_N, REG_M*REG_N> mat_O=0.f;

    #pragma unroll
    for (int i = 0,k = 0; i < head_size/REG_K; i++, k+=REG_K/2) {
        q_desc.set_block_x(k);
        cm_load<lsc::Transpose>(mat_Qtr[i].format<uint>(), q_desc);
    }

    vector<float, REG_N> lastmax;
    vector<float, 2*REG_M> rowsum;


    //show(mat_Qtr.format<half, REG_K, REG_N*2>());
    for (int kv_idx = 0; kv_idx < kv_len; kv_idx +=kv_step) {
        //load KV into SLM
        {
            matrix<half, 2, REG_M*REG_K> tempK;

            for (int offset_K = wg_local_id * REG_K; offset_K < head_size; offset_K += REG_K * wg_size) {
                k_desc.set_block_x(offset_K);
                k_desc.set_block_y(kv_idx);
                cm_load<lsc::Normal>(tempK[0], k_desc);
                k_desc.set_block_y(kv_idx + REG_M);
                cm_load<lsc::Normal>(tempK[1], k_desc);
                //show(tempK.format<half, 2*REG_M, REG_K>());

                //2*regM*regK as a block. The [kv_step, head_size] is also [2, REGM, head_size/REGK, REGK] -> [head_size/REGK, 2, REGM, REGK]
                int offset = offset_K*REG_M*2*sizeof(half);
                cm_slm_block_write(slm_K, offset, tempK[0].format<half>());
                cm_slm_block_write(slm_K, offset+REG_M*REG_K*sizeof(half), tempK[1].format<half>());
            }

            v_desc.set_block_y(kv_idx);

            matrix<half, REG_K, REG_N> tempV;
            for (int offset_N = wg_local_id * REG_N; offset_N < head_size; offset_N += REG_N * wg_size) {

                //[REGK, REGN] as a block
                v_desc.set_block_x(offset_N);
                cm_load<lsc::VNNI>(tempV.format<half>(), v_desc);
                int offset = offset_N*REG_K*sizeof(half);
                cm_slm_block_write(slm_V, offset, tempV.format<half>());

            }
            cm_barrier();
        }

        //#load K from SLM
        //#compute S_tr =K*Q_tr
        matrix<float, 2*REG_M, REG_N> St = 0;
        matrix<half, 2, REG_M * REG_K> Kmat;

        auto St2 = St.format<float, 2, REG_M*REG_N>();
        uint offset = 0;
        #pragma unroll
        for(int k = 0, ri = 0; k < head_size; k += REG_K, ri++) {
            cm_slm_block_read(slm_K, GENX_NONE, offset, Kmat[0]); offset += REG_M * REG_K * sizeof(half);
            cm_slm_block_read(slm_K, GENX_NONE, offset, Kmat[1]); offset += REG_M * REG_K * sizeof(half);
            //show(Kmat.format<half, 2*REG_M, REG_K>());
            St2[0] = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount>(
                        St2[0],
                        mat_Qtr[ri].format<int32_t>(),
                        Kmat[0].format<int32_t>());
            St2[1] = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount>(
                        St2[1],
                        mat_Qtr[ri].format<int32_t>(),
                        Kmat[1].format<int32_t>());
        }
        //#Transpose  S_tr
        //show(St);
        St *= scale_factor;

        matrix<half, 2*REG_M, REG_N> atten_mask;
        matrix<float, 2*REG_M, REG_N> atten_mask_tr;
        atten_desc.set_block_x(kv_idx);
        cm_load<lsc::Normal>(atten_mask.format<half>(), atten_desc);
        Transpose_16x16(atten_mask, atten_mask_tr.format<float, 2*REG_M, REG_N>());

        St += atten_mask_tr;
        vector<float, REG_N> rowmax;
        vector<float, 2*REG_M> tempsum;
        vector<float, 2*REG_M> max_comp=0.f;
        for (int colidx = 0; colidx < REG_N; colidx++) {
            rowmax[colidx] = cm_reduced_max<float>(St.column(colidx).format<float>());
        }

        if (kv_idx == 0)
            lastmax = rowmax;
        else
            rowmax = cm_max<float>(rowmax.format<float>(), lastmax.format<float>());

        constexpr float log2e = 1.4426950408889634f;
        #pragma unroll
        for (int rowidx = 0; rowidx < REG_M*2; rowidx++) {
            St[rowidx].format<float>() = cm_exp((St[rowidx] - rowmax)*log2e);
         }
        #pragma unroll
        //#seems removing progam unroll would caused accuracy error , here.
        for (int idx = 0; idx < REG_N; idx++) {
            tempsum[idx] = cm_sum<float>(St.format<float, 2*REG_M, REG_N>().column(idx));
        }

        if (kv_idx == 0) {
            rowsum = tempsum;
        } else {
            max_comp = cm_exp((lastmax-rowmax)*log2e);
            rowsum = tempsum + rowsum*max_comp;
        }
        //# show(lastmax.format<float, 1, 2*REG_M>());
        //# show(rowmax.format<float, 1, 2*REG_M>());
        //# show(max_comp.format<float, 1, 2*REG_M>());
        //# show(tempsum.format<float, 1, 2*REG_M>());
        //# show(rowsum.format<float, 1, 2*REG_M>());

        matrix<half, 2, REG_M*REG_K> matP;
        Transpose_16x16(St, matP.format<half, 2*REG_M, REG_K>());

        //show(matP.format<half, 2*REG_M, REG_K>());

        //#load V from SLM
        //#compute S*V
        //#matrix<float, head_size/REG_N*2, REG_M*REG_N> mat_O;

        for(int nn = 0, idx = 0; nn < head_size; nn += REG_N, idx++) {
            vector<half, REG_K*REG_N> tempV;
            cm_slm_block_read(slm_V, GENX_NONE,  REG_N * REG_K * sizeof(half) * idx, tempV);
            //show(tempV.format<half, REG_K/2, REG_N*2>());
            auto block0 =  mat_O[idx].format<float, REG_M, REG_N>();
            for (int i = 0; i < REG_M; i++)
                block0[i] = cm_mul<float>(block0[i], max_comp[i]);

            mat_O[idx] = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount>(
                   mat_O[idx].format<float>(),
                    tempV.format<int32_t>(),
                   matP[0].format<int32_t>());

            auto block1 =  mat_O[idx+(head_size/REG_N)].format<float, REG_M, REG_N>();
            for (int i = 0; i < REG_M; i++)
                block1[i] = cm_mul<float>(block1[i], max_comp[i+REG_M]);

            mat_O[idx+(head_size/REG_N)] = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount>(
                        mat_O[idx+(head_size/REG_N)].format<float>(),
                        tempV.format<int32_t>(),
                        matP[1].format<int32_t>());
        }
        lastmax = rowmax;
        cm_barrier();
    }

#if 0
    matrix<half, 2*REG_M, head_size> mat_O_dest = 0;
    #pragma unroll
    for (int row = 0; row < 2*REG_M; row++) {
        matrix_ref<half, head_size/REG_N, REG_N> row_mat_dest = mat_O_dest[row].format<half, head_size/REG_N, REG_N>();
        #pragma unroll
        for (int col = 0; col < head_size/REG_N; col++) {
            uint blk_idx = row / REG_M * (head_size/REG_N) + col;
            auto src_blk = mat_O[blk_idx].format<float, REG_M, REG_N>();
            row_mat_dest[col].format<half>() = src_blk[row%REG_M].format<float>()/rowsum[row];
        }
    }
#endif


    matrix<half, 2*head_size/REG_N, REG_M*REG_N> mat_O_dest = 0;
    //# output [batch, q_len, num_heads, S], rect window [REG_M, REG_N]
    lsc::block_2d_desc<half, 1, REG_M, REG_N> o_desc((half*)(output + q_offset), q_len - 1, head_size*sizeof(half)-1, q_pitch - 1, 0, 0);
    int blks_per_row = head_size/REG_N;
    #pragma unroll
    for (int blk = 0; blk < 2*head_size/REG_N; blk++) {
        auto src = mat_O[blk].format<float, REG_M, REG_N>();
        auto dest = mat_O_dest[blk].format<half, REG_M, REG_N>();
        auto rowoff = blk/blks_per_row == 0 ? 0 : REG_M;
        #pragma unroll
        for (int i = 0; i < REG_M; i++) {
            dest[i] = src[i] / rowsum[i+rowoff];
        }
        o_desc.set_block_y(q_start + blk/blks_per_row * REG_M);
        o_desc.set_block_x(blk%blks_per_row * REG_N);
        cm_store(o_desc, dest.format<half>());
    }

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < head_size / REG_N; j++) {
            auto idx = i * head_size/REG_N + j;
            //show(mat_O_dest[idx].format<half, REG_M, REG_N>());
        }
    }
}
'''

cl.profiling(True)

t_q = cl.tensor(q.detach().numpy())
t_k = cl.tensor(k.detach().numpy())
t_v = cl.tensor(v.detach().numpy())
t_out = cl.tensor([batch, q_len, num_heads, head_size], np.dtype(np.float16))
t_mask = cl.tensor(attention_mask.detach().numpy())

# f"-cmc -mdump_asm -g2 "
cm_kernels = cl.kernels(pyeval(src1), f"-cmc -Qxcm_register_file_size=256 -mdump_asm -g2")
cm_kernels.enqueue("cm_sdpa", GWS, LWS, q_len, kv_len, t_q, t_k, t_v, t_out, t_mask)
torch.set_printoptions(precision=4, sci_mode=False)

f1 = torch.from_numpy(t_out.numpy())
check_close(org.transpose(1,2), f1, atol=1e-2, rtol=1e-3)
print(f"=========== cm_sdpa PASS GWS={GWS} LWS={LWS}  ===========")
sys.exit(0)

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

for i in range(1):
    j  = i % len(all_layers)
    cm_kernels.enqueue("cm_sdpa", GWS, LWS, q_len, kv_len,
                       all_layers[j][0],
                       all_layers[j][1],
                       all_layers[j][2],
                       all_layers[j][3],
                       t_mask)

latency = cl.finish()
for ns in latency:
    print(f"  {ns*1e-6:.3f} ms")

sys.exit(0)
