import argparse
import sys
import numpy as np
import time

from einops import rearrange, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F

from clops import cl
from clops.utils import *

enable_vprint = False
def vprint(*all_args):
    global enable_vprint
    if enable_vprint:
        print(*all_args)

def get_org(Q, K, V, attention_mask):
    # BLHS=>BHLS
    Q, K, V = (rearrange(x, "b s h d -> b h s d")
                for x in [Q, K, V])

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
            
    out = rearrange(out, "b h s d -> b s h d ")
    # print("out:", out.shape, out.dtype)
    return out

def get_org0(q, k, v, attention_mask, enable_gqa):
    # BLHS=>BHLS
    q, k, v = (rearrange(x, "b s h d -> b h s d")
                for x in [q, k, v])

    # print("q:", q.shape, q.dtype)
    # print("k:", k.shape, k.dtype)
    # print("v:", v.shape, v.dtype)
    # print("attention_mask:", attention_mask.shape, attention_mask.dtype)

    out = F.scaled_dot_product_attention(q, k, v, attention_mask, dropout_p=0.0, enable_gqa = enable_gqa)
    out = rearrange(out, "b h s d -> b s h d ")
    # print("out:", out.shape, out.dtype)
    return out     

def check_close(input, other, atol=1e-3, rtol=1e-3):
    print(f"[check_close] {input.shape}{input.dtype} vs {other.shape}{other.dtype}")
    # rtol_max = (((input - other).abs() - 1e-5)/other.abs())[other != 0].max()
    # atol_max = (((input - other).abs()) - 1e-5*other.abs()).max()
    # print(f"[check_close] rtol_max: {rtol_max}")
    # print(f"[check_close] atol_max: {atol_max}")
    if not torch.allclose(input, other, atol=atol, rtol=rtol):
        close_check = torch.isclose(input, other, atol=atol, rtol=rtol)
        not_close_indices = torch.where(~close_check) # Invert the close check to find failures
        # print(f"Not close indices: {not_close_indices}")
        # print(f"    input_tensor: {input[not_close_indices]}")
        # print(f"    other_tensor: {other[not_close_indices]}")
        assert 0

# blocking on kv-len dimension with online-softmax
def get_flash0(query, key, value, attention_mask, q_step = 16, kv_step = 16):
    # BLHS=>BHLS
    query, key, value = (rearrange(x, "b s h d -> b h s d")
                for x in [query, key, value])

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
                    vprint("cur_O2=", cur_O)

                    cur_max = rowmax
                    if (j == args.verbose): assert 0

                cur_O_f16 = (cur_O/cur_sum.transpose(0, 1)).to(torch.float16)

                if (i == args.verbose):
                    enable_vprint = True
                    vprint("cur_O_f16=", cur_O_f16.shape, cur_O_f16)
                    assert 0

                out[b, h, i:i1, :] = cur_O_f16
                
    out = rearrange(out, "b h s d -> b s h d ")
    return out

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

src1 = r'''
//# CM kernel for flash attn, reference

#pyeval f"#define num_heads {num_heads}"
#pyeval f"#define num_kv_heads {num_kv_heads}"
#pyeval f"#define head_size {head_size}"
#pyeval f"#define q_step {q_step}"
#pyeval f"#define kv_step {kv_step}"
#pyeval f"#define scale_factor {scale_factor}"
#pyeval f"#define args_verbose {args.verbose}"
#pyeval f"#define HAS_ATTN_MASK_INPUT {args.has_attention_mask}"

#define args_verbose 1

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
    half* query [[type("svmptr_t")]],
    half* key [[type("svmptr_t")]],
    half* value [[type("svmptr_t")]],
#if HAS_ATTN_MASK_INPUT
    half* mask [[type("svmptr_t")]],
#endif
    half* output [[type("svmptr_t")]],
    int valid_q_len,
    int valid_kv_len,
    int q_len,
    int kv_len
    ) {
    //# query [batch, q_len, num_heads, S]
    //#   key [batch, kv_len, num_heads, S]
    //# value [batch, kv_len, num_heads, S]
    //# to load Q

    constexpr uint K_SLM_SIZE = (kv_step * head_size * sizeof(half));
    constexpr uint V_SLM_SIZE = (kv_step * head_size * sizeof(half));
    constexpr uint Q_SLM_SIZE = 0;//(q_step * head_size * sizeof(half)) * WG_SIZE;

    cm_slm_init(K_SLM_SIZE + V_SLM_SIZE + Q_SLM_SIZE);

    auto slm_K = cm_slm_alloc(K_SLM_SIZE);
    auto slm_V = cm_slm_alloc(V_SLM_SIZE);

    vector<float, q_step> cur_max;
    vector<float, q_step> cur_sum;

    cur_max = -1e9;
    cur_sum = 0;
    
    auto WG_SIZE = cm_local_size(2);
    //# printf("WG_SIZE = ", WG_SIZE);

    auto batch = cm_global_id(0);
    auto h = cm_global_id(1);
    auto hkv = h / (num_heads/num_kv_heads);
    auto wg_local_id = cm_local_id(2);
    auto q_start = cm_global_id(2) * q_step;
    auto q_offset = wg_local_id * q_step * head_size * sizeof(half);
    auto o_offset = wg_local_id * q_step * head_size * sizeof(float);

    //# debugging stage
#if HAS_ATTN_MASK_INPUT
    lsc::block_2d_desc<half, 1, REG_M, REG_N> b2dMask(mask + batch * q_len * kv_len, q_len - 1, kv_len*sizeof(half) - 1, kv_len*sizeof(half) - 1, 0, 0);
#endif

    //# b2dQ reinterpret as 32bit(DWORD) for transposed load(combined with VNNI)
    uint qo_pitch = num_heads * head_size;
    uint kv_pitch = num_kv_heads * head_size;
    lsc::block_2d_desc<uint, 1, REG_N, REG_K/2> b2dQ(reinterpret_cast<uint*>(query + (batch*num_heads*q_len + h)*head_size), valid_q_len - 1, head_size*sizeof(half) - 1, qo_pitch * sizeof(half) - 1, 0, 0);
    lsc::block_2d_desc<half, 1, REG_M, REG_K> b2dK(key + (batch*num_kv_heads*kv_len + hkv)*head_size,   valid_kv_len - 1, head_size*sizeof(half) - 1, kv_pitch * sizeof(half) - 1, 0, 0);
    lsc::block_2d_desc<half, 1, REG_K, REG_N> b2dV(value + (batch*num_kv_heads*kv_len + hkv)*head_size, valid_kv_len - 1, head_size*sizeof(half) - 1, kv_pitch * sizeof(half) - 1, 0, 0);
    lsc::block_2d_desc<half, 1, REG_M, REG_N> b2dO(output + (batch*num_heads*q_len + h)*head_size, valid_q_len - 1, head_size*sizeof(half) - 1, qo_pitch * sizeof(half) - 1, 0, 0);
    
    // if (h==0 && batch == 0 && wg_local_id == 0) printf("========= query = %p =============\n", query);

    //# load Qt into register & pack as VNNI & store to SLM (as dpas-B tile)

    matrix<half, head_size/REG_K, REG_K*REG_N> rQ;
    {
        //matrix<uint, REG_K/2, REG_N> Qmat;
        #pragma unroll
        for(int k = 0, ri = 0; k < head_size/2; k += REG_K/2, ri++) {
            b2dQ.set_block_x(k);

            //# DWORD transposed load == (transposed + VNNI) load
            cm_load<lsc::Transpose>(rQ[ri].format<uint>(), b2dQ.set_block_y(q_start));

            //show(Qmat.format<half, REG_K/2, REG_N*2>());
        }
    }

    int kv_stop = valid_kv_len;

    matrix <float, head_size/REG_K*2, REG_M*REG_N> rO;
    for(int kv_pos = 0; kv_pos < kv_stop; kv_pos += kv_step) {
        // if (args_verbose >= 0) printf("======== %d =========\n", kv_pos);

        //===========================================================
        //# load K into SLM as dpas-A tile (shared by all hw within same WG)
        //# load V into SLM as dpas-B tile (shared by all hw within same WG)
        {
            //# 1849 ~ 3259
            if (kv_pos > 0) cm_barrier();
            {
                // auto clk0 = get_clock();
                if (WG_SIZE == 1) {
                    matrix<half, REG_M, REG_K> temp0;
                    matrix<half, REG_M, REG_K> temp1;
                    for(int k = 0; k < head_size; k += REG_K) {
                        b2dK.set_block_x(k);
                        cm_load<lsc::Normal>(temp0.format<half>(), b2dK.set_block_y(kv_pos));
                        cm_load<lsc::Normal>(temp1.format<half>(), b2dK.set_block_y(kv_pos + REG_M));

                        //cm_prefetch(b2dK.set_block_y(kv_pos + kv_step));
                        //cm_prefetch(b2dK.set_block_y(kv_pos + kv_step + REG_M));

                        //show(temp0); return;
                        
                        uint offset = k * 2 * REG_M * sizeof(half);
                        cm_slm_block_write(slm_K, offset, temp0.format<half>());
                        offset += REG_M * REG_K * sizeof(half);
                        cm_slm_block_write(slm_K, offset, temp1.format<half>());
                    }
                    matrix<half, REG_K, REG_N> temp2;
                    b2dV.set_block_y(kv_pos);
                    for(int k = 0; k < head_size; k += REG_K) {
                        cm_load<lsc::VNNI>(temp2.format<half>(), b2dV.set_block_x(k).set_block_y(kv_pos));
                        //cm_prefetch(b2dV.set_block_y(kv_pos + kv_step));

                        cm_slm_block_write(slm_V, k * REG_N * sizeof(half), temp2.format<half>());
                    }
                } else {
                    if (wg_local_id < WG_SIZE/2) {
                        matrix<half, REG_M, REG_K> temp0;
                        matrix<half, REG_M, REG_K> temp1;
                        for(int k = REG_K*wg_local_id; k < head_size; k += REG_K*(WG_SIZE/2)) {
                            b2dK.set_block_x(k);
                            cm_load<lsc::Normal>(temp0.format<half>(), b2dK.set_block_y(kv_pos));
                            cm_load<lsc::Normal>(temp1.format<half>(), b2dK.set_block_y(kv_pos + REG_M));

                            //cm_prefetch(b2dK.set_block_y(kv_pos + kv_step));
                            //cm_prefetch(b2dK.set_block_y(kv_pos + kv_step + REG_M));

                            uint offset = k * 2 * REG_M * sizeof(half);
                            cm_slm_block_write(slm_K, offset, temp0.format<half>());
                            offset += REG_M * REG_K * sizeof(half);
                            cm_slm_block_write(slm_K, offset, temp1.format<half>());
                        }
                    } else {
                        matrix<half, REG_K, REG_N> temp2;
                        b2dV.set_block_y(kv_pos);
                        for(int k = REG_K*(wg_local_id-WG_SIZE/2); k < head_size; k += REG_K*(WG_SIZE/2)) {
                            cm_load<lsc::VNNI>(temp2.format<half>(), b2dV.set_block_x(k).set_block_y(kv_pos));
                            //cm_prefetch(b2dV.set_block_y(kv_pos + kv_step));

                            cm_slm_block_write(slm_V, k * REG_N * sizeof(half), temp2.format<half>());
                        }
                    }
                }
            }
            // printf(" diff= %lu\n", get_clock() - clk0);

            cm_barrier();
        }

        //=========================================================== 1807 ~ 3247
        //# St = k @ Qt
        matrix<float, 2*REG_M, REG_N> St = 0;
        matrix<half, 2, REG_M * REG_K> Kmat;
        matrix<half, REG_K/2, REG_N*2> Qmat;
        auto St2 = St.format<float, 2, REG_M*REG_N>();
        uint offset = 0;
        #pragma unroll
        for(int k = 0, ri = 0; k < head_size; k += REG_K, ri++) {
            cm_slm_block_read(slm_K, GENX_NONE, offset, Kmat[0]); offset += REG_M * REG_K * sizeof(half);
            cm_slm_block_read(slm_K, GENX_NONE, offset, Kmat[1]); offset += REG_M * REG_K * sizeof(half);
            //show(Kmat.format<half, 2*REG_M, REG_K>());

            St2[0] = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount>(
                        St2[0],
                        rQ[ri].format<int32_t>(),
                        Kmat[0].format<int32_t>());
            St2[1] = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount>(
                        St2[1],
                        rQ[ri].format<int32_t>(),
                        Kmat[1].format<int32_t>());
        }
        //show(St);
        //=========================================================== 361
#if HAS_ATTN_MASK_INPUT
        matrix<half, 2, REG_M * REG_N> Maskmat;
        b2dMask.set_block_x(kv_pos);
        cm_load<lsc::Normal>(Maskmat[0].format<half>(), b2dMask.set_block_y(q_start));
        cm_load<lsc::Normal>(Maskmat[1].format<half>(), b2dMask.set_block_y(q_start + REG_M));

        matrix<float, 2*REG_M, REG_N> MaskT;
        Transpose_16x16(Maskmat.format<half, 2*REG_M, REG_N>(), MaskT);

        //show(Maskmat);
#endif
        St = cm_mul<float>(St, (float)scale_factor);  // convert scale_factor into (float), or it will be promoted to double
#if HAS_ATTN_MASK_INPUT
        St = cm_add<float>(St, MaskT);
#endif

        //show(St);

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

        matrix<half, 2*REG_M, REG_K> P;
        Transpose_16x16(St, P);

        //show(cur_sum.format<float, 1, REG_N>()); return;
        //============================================================== 1074


        //============================================================== 666
        //show(P);return;
        //auto clk0 = get_clock();

        auto P2 = P.format<half, 2, REG_M * REG_K>();
        matrix<float, 2, REG_M*REG_N> cur_O;
        if (kv_pos == 0) {
            matrix<float, REG_M, REG_N> zero_O = 0;
            matrix<half, REG_K/2, REG_N*2> Vmat;
            uint offset = o_offset;
            #pragma unroll
            for(int k = 0, ri = 0; k < head_size; k += REG_K, ri += 2) {
                // V has been VNNI-prepacked
                cm_slm_block_read(slm_V, GENX_NONE, REG_N*k*sizeof(half), Vmat.format<half>());
                
                rO[ri] = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount>(
                                zero_O.format<float>(),
                                Vmat.format<int32_t>(),
                                P2[0].format<int32_t>());
                rO[ri+1] = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount>(
                                zero_O.format<float>(),
                                Vmat.format<int32_t>(),
                                P2[1].format<int32_t>());
                //show(cur_O.format<float, 2*REG_M, REG_N>());
            }
        } else {
            matrix<half, REG_K/2, REG_N*2> Vmat;
            uint offset = o_offset;
            #pragma unroll
            for(int k = 0, ri=0; k < head_size; k += REG_K, ri+=2) {
                // V has been VNNI-prepacked
                cm_slm_block_read(slm_V, GENX_NONE, REG_N*k*sizeof(half), Vmat.format<half>());

                //# compensate cur_O
                //  matrix <float, head_size/REG_K*2, REG_M*REG_N> rO;
                auto cO = rO[ri].format<float, REG_M, REG_N>();
                for(int r = 0; r < REG_M; r++)
                    cO.row(r) = cm_mul<float>(cO.row(r), max_comp[r]);
                auto cO2 = rO[ri+1].format<float, REG_M, REG_N>();
                for(int r = 0; r < REG_M; r++)
                    cO2.row(r) = cm_mul<float>(cO2.row(r), max_comp[r + REG_M]);

                //# show(cur_O.format<float, 2*REG_M, REG_N>()); return;
                
                rO[ri] = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount>(
                                rO[ri].format<float>(),
                                Vmat.format<int32_t>(),
                                P2[0].format<int32_t>());
                rO[ri+1] = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount>(
                                rO[ri+1].format<float>(),
                                Vmat.format<int32_t>(),
                                P2[1].format<int32_t>());

                // if (kv_pos == args_verbose) show(cur_O.format<float, 2*REG_M, REG_N>());
            }
            // if (kv_pos == args_verbose) return;
        }
        //============================================================== 1168

        cur_max = new_max_t;
    }//# for(int kv_pos = 0; kv_pos < kv_len; kv_pos += kv_step) {
    
    //# save cur_O/cur_sum.transpose(0, 1)
    matrix<float, 2, REG_M*REG_N> cur_O;
    matrix<half, 2*REG_M, REG_N> cur_O_f16;
    uint offset = o_offset;
    #pragma unroll
    for(int k = 0, ri=0; k < head_size; k += REG_K, ri+=2) {
        auto cO = rO[ri].format<float, REG_M, REG_N>();
        for(int r = 0; r < cO.n_rows(); r++) {
            cur_O_f16[r] = cm_div_ieee(cO[r], cur_sum[r]);
        }
        auto cO2 = rO[ri+1].format<float, REG_M, REG_N>();
        for(int r = 0; r < cO2.n_rows(); r++) {
            cur_O_f16[r + REG_M] = cm_div_ieee(cO2[r], cur_sum[r+REG_M]);
        }

        // if (h==0 && batch == 0 && wg_local_id == 0) show(cur_O_f16);

        cm_store(b2dO.set_block_x(k).set_block_y(q_start), cur_O_f16.format<half, 2, REG_M*REG_N>()[0]);
        cm_store(b2dO.set_block_x(k).set_block_y(q_start + REG_M), cur_O_f16.format<half, 2, REG_M*REG_N>()[1]);
    }
    // if (i == args_verbose) return;
}
'''

class FlashAttentionV2_CM:
    def __init__(self, num_heads, num_kv_heads, head_size, q_step, kv_step):
        self.num_heads, self.num_kv_heads, self.head_size, self.q_step, self.kv_step = num_heads, num_kv_heads, head_size, q_step, kv_step
        # f"-cmc -mdump_asm -g2 "
        # print("compiling ...")
        src = pyeval(src1)
        # print(src)
        self.cm_kernels = kernel_cache(src, f"-cmc -Qxcm_register_file_size=256")

    def __call__(self, q, k, v, attention_mask = None, out = None, valid_q_len = -1, valid_kv_len = -1, start_q_idx = 0, start_kv_idx = 0):
        t_q = cl.tensor(q.detach().numpy())
        t_k = cl.tensor(k.detach().numpy())
        t_v = cl.tensor(v.detach().numpy())
        
        #// BLHS
        batch, q_len, kv_len = q.shape[0], q.shape[1], k.shape[1]

        if valid_q_len == -1:
            valid_q_len = q_len
        if valid_kv_len == -1:
            valid_kv_len = kv_len
            
        print(f'call ...{batch=}, {valid_q_len=}, {valid_kv_len=}, {q_len=}, {kv_len=}, {start_q_idx=}, {start_kv_idx=}')

        # out = torch.zeros([batch, q_len, num_heads, head_size], device='cpu', dtype=torch.float16)
        if out is None:
            t_out = cl.tensor([batch, q_len, self.num_heads, self.head_size], np.dtype(np.float16))
        else:
            t_out = cl.tensor(out.detach().numpy())

        WG_SIZE = min(valid_q_len//self.q_step, 8)
        GWS=[batch, self.num_heads, valid_q_len//self.q_step]
        LWS=[1, 1, WG_SIZE]
        print("GWS=", GWS)
        print("LWS=", LWS)
        
        # BLHS
        qo_picth = self.num_heads * self.head_size * q.element_size()
        kv_pitch = self.num_kv_heads * self.head_size * k.element_size()
        t_q.offset = start_q_idx * qo_picth
        t_k.offset = start_kv_idx * kv_pitch
        t_v.offset = start_kv_idx * kv_pitch
        t_out.offset = start_q_idx * qo_picth
        
        # print("out:", t_out.shape, t_out.dtype, t_out.offset, hex(t_out.addr))

        if attention_mask is not None:
            t_mask = cl.tensor(attention_mask.detach().numpy())
            self.cm_kernels.enqueue("cm_sdpa", GWS, LWS, t_q, t_k, t_v, t_mask, t_out, valid_q_len, valid_kv_len, q_len, kv_len)
        else:
            self.cm_kernels.enqueue("cm_sdpa", GWS, LWS, t_q, t_k, t_v, t_out, valid_q_len, valid_kv_len, q_len, kv_len)

        # t_out.offset = 0 # reset
        # print("==>out:", t_out.shape, t_out.dtype, t_out.offset, hex(t_out.addr))
        f1 = torch.from_numpy(t_out.numpy())
        return f1, GWS, LWS

if __name__ == "__main__":
    torch.manual_seed(0)
    torch.set_printoptions(linewidth=1024)

    parser = argparse.ArgumentParser('')
    parser.add_argument('-i', "--impl", type=int, default=0)
    parser.add_argument('-b', "--batch", type=int, default=1)
    parser.add_argument('-nh', "--num-heads", type=int, default=16)
    parser.add_argument('-nkvh', "--num-kv-heads", type=int, default=16)
    parser.add_argument('-ql', "--q-len", type=int, default=32)
    parser.add_argument('-kvl', "--kv-len", type=int, default=512)
    parser.add_argument('-hs', "--head-size", type=int, default=80)
    parser.add_argument('-v', "--verbose", type=int, default=-1)
    parser.add_argument('-m', "--has-attention-mask", type=int, default=1)

    #parser.add_argument('-q', "--quant_type", type=str, default="w4a", choices=['f16', 'f16b1', 'w4a', 'w4a_cpu', 'f16xmx', 'w4x'])
    #parser.add_argument('-hf', '--hf_model_path', type=str, nargs='?', default='/mnt/llm_irs/models_original/Qwen2-0.5B-Instruct/')
    #parser.add_argument('--save', type=str, nargs='?', default=None)
    #parser.add_argument('--load', type=str, nargs='?', default=None)
    args = parser.parse_args()
    print(args)

    '''
    ugemm_qk: [q_step, head_size] x [head_size, kv_step]
    ugemm_kq: [kv_step, head_size] x [head_size, q_step]
    ugemm_pv: [q_step, kv_step] x [kv_step, head_size]
    '''
    batch = args.batch
    q_len, q_step = args.q_len, 16
    kv_len, kv_step = args.kv_len, 16
    num_heads = args.num_heads
    num_kv_heads = args.num_kv_heads
    head_size = args.head_size
    enable_gqa = num_heads > num_kv_heads

    scale_factor = 1.0/(head_size**0.5)

    cl.profiling(True)

    def test_flashattn0(args):
        low = -7
        high = 8
        act_dtype = torch.float16
        q = torch.randint(low, high, [batch, q_len, num_heads, head_size]).to(dtype=act_dtype)/high
        k = torch.randint(low, high, [batch, kv_len, num_kv_heads, head_size]).to(dtype=act_dtype)/high
        v = torch.randint(low, high, [batch, kv_len, num_kv_heads, head_size]).to(dtype=act_dtype)/high

        # random attnmask
        attention_mask = torch.full([batch, 1, q_len, kv_len], torch.finfo(act_dtype).min).to(dtype=act_dtype)
        attention_mask[torch.rand(batch, 1, q_len, kv_len) > 0.5] = 0
        print(f'HAS_ATTN_MASK_INPUT={args.has_attention_mask}')
        if args.has_attention_mask is 0:
            attention_mask[...] = 0

        # ref = get_org0(q,k,v,attention_mask,enable_gqa)
        org = get_org(q,k,v,attention_mask)
        # check_close(ref, org, atol=1e-3, rtol=1e-2)

        # [batch, seq-len, heads, size] BLHS
        # print("ref:", ref.shape, ref.dtype)

        if args.impl == 0:
            f0 = get_flash0(q,k,v,attention_mask, q_step, kv_step)
            check_close(org, f0, atol=1e-2, rtol=1e-3)
            print("=========== PASS ===========")
            sys.exit(0)

        #====================================================================================================
        # using the same parameter & inputs, develop cm kernels which produces the same output
        # prototyping CM kernels

        # in orginal shape: [batch, q_len, num_heads, head_size]
        # print("q:", q.shape, q.dtype)
        # print("k:", k.shape, k.dtype)
        # print("v:", v.shape, v.dtype)
        # print("attention_mask:", attention_mask.shape, attention_mask.dtype)

        fla_cm = FlashAttentionV2_CM(num_heads, num_kv_heads, head_size, q_step, kv_step)
        f1, GWS, LWS = fla_cm(q, k, v, attention_mask)

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
            # print(f"nlayers={len(all_layers)} mem_size={mem_size*1e-9:.3f} GB")

        for i in range(0):
            j  = i % len(all_layers)
            fla_cm(q, k, v, attention_mask)

        latency = cl.finish()
        for ns in latency:
            print(f"  {ns*1e-6:.3f} ms")

        check_close(org, f1, atol=1e-2, rtol=1e-3)
        print(f"=========== cm_sdpa PASS GWS={GWS} LWS={LWS}  ===========")

    def test_visionsdpa(args, num_images = 8, grid = 32):
        grid_thw = torch.zeros([num_images, 3], device='cpu', dtype=torch.long)
        for n in range(num_images):
            grid_thw[n, 0] = 1
            grid_thw[n, 1] = grid
            grid_thw[n, 2] = grid

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        seq_length = cu_seqlens[-1]
        q_len = kv_len = seq_length   # reset q kv lens
        attention_mask = torch.zeros([1, seq_length, seq_length], device='cpu', dtype=torch.float16) + torch.finfo(torch.float16).min
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0

        # print(f'{attention_mask=}')
        print(f'{cu_seqlens=}')
        
        low = -7
        high = 8
        act_dtype = torch.float16
        q = torch.randint(low, high, [batch, q_len, num_heads, head_size]).to(dtype=act_dtype)/high
        k = torch.randint(low, high, [batch, kv_len, num_kv_heads, head_size]).to(dtype=act_dtype)/high
        v = torch.randint(low, high, [batch, kv_len, num_kv_heads, head_size]).to(dtype=act_dtype)/high
        
        # q = torch.ones([batch, q_len, num_heads, head_size], device='cpu', dtype=act_dtype)
        # k = torch.ones([batch, kv_len, num_kv_heads, head_size], device='cpu', dtype=act_dtype)
        # v = torch.ones([batch, kv_len, num_kv_heads, head_size], device='cpu', dtype=act_dtype)

        # print("q:", q.shape, q.dtype)
        # print("k:", k.shape, k.dtype)
        # print("v:", v.shape, v.dtype)
        # print("attention_mask:", attention_mask.shape, attention_mask.dtype)
        
        def get_ref(q, k, v, cu_seqlens):
            # Execute attention entry by entry for speed & less VRAM.
            outputs = []
            for i in range(1, len(cu_seqlens)):
                start_idx = cu_seqlens[i - 1]
                end_idx = cu_seqlens[i]
                q_i = q[:, start_idx:end_idx]
                k_i = k[:, start_idx:end_idx]
                v_i = v[:, start_idx:end_idx]
                q_i, k_i, v_i = (rearrange(x, "b s h d -> b h s d")
                                    for x in [q_i, k_i, v_i])
                output_i = F.scaled_dot_product_attention(q_i, k_i, v_i, dropout_p=0.0)
                output_i = rearrange(output_i, "b h s d -> b s h d ")
                outputs.append(output_i)
            context_layer = torch.cat(outputs, dim=1)
            # print("context_layer:", context_layer.shape, context_layer.dtype)
            return context_layer

        org = get_org0(q, k, v, attention_mask, enable_gqa)
        
        if args.impl == 0:
            ref = get_ref(q, k, v, cu_seqlens)
            # [batch, seq-len, heads, size] BLHS
            # print("ref:", ref.shape, ref.dtype)
            check_close(ref, org, atol=1e-3, rtol=1e-2)
            print("=========== PASS ===========")
            sys.exit(0)
        
        args.has_attention_mask = 0
        out = torch.zeros([batch, q_len, num_heads, head_size], device='cpu', dtype=act_dtype)
        fla_cm = FlashAttentionV2_CM(num_heads, num_kv_heads, head_size, q_step, kv_step)
        for i in range(1, len(cu_seqlens)):
            start_idx = cu_seqlens[i - 1]
            end_idx = cu_seqlens[i]

            valid_q_len = end_idx - start_idx
            print(f"\n{start_idx=}")
            
            out, GWS, LWS = fla_cm(q, k, v, None, out, valid_q_len.item(), valid_q_len.item(), start_idx.item(), start_idx.item())
            # print("out:", out.shape, out.dtype, out[:, start_idx : end_idx, :1, :2])
        latency = cl.finish()
        for ns in latency:
            print(f"  {ns*1e-6:.3f} ms")

        # print("org:", org.shape, org.dtype, org)
        # print("out:", out.shape, out.dtype, out)
        check_close(out, org, atol=1e-2, rtol=1e-3)
        print(f"=========== cm_sdpa PASS GWS={GWS} LWS={LWS}  ===========")

    # test_flashattn0(args)
    # python tests/flashattn/flash1.py  -i 1 -hs 16  -nh 2 -nkv 2 -v 1
    # test_visionsdpa(args, 2, 4)
    test_visionsdpa(args);sys.exit(0)