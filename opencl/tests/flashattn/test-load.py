import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)
torch.set_printoptions(linewidth=1024)

# prototyping CM kernels
from clops import cl
import numpy as np
import time

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


seq_len = 8192
head_num = 28
kvhead_num = 4
head_sz = 128

grp_sz = head_num//kvhead_num
local_sz=64
low=-2
high = 3

q_factor = torch.randint(high, high+3, [seq_len, head_num, head_sz]).to(dtype=torch.float16)
kv_factor = torch.randint(high, high+3, [seq_len, kvhead_num, head_sz]).to(dtype=torch.float16)

q = torch.randint(low, high, [seq_len, head_num, head_sz]).to(dtype=torch.float16) / q_factor
k = torch.randint(low, high, [seq_len, kvhead_num, head_sz]).to(dtype=torch.float16) / kv_factor
v = torch.randint(low, high, [seq_len, kvhead_num, head_sz]).to(dtype=torch.float16) / kv_factor

qint8_ref = torch.randint(low, high, [seq_len, head_num, head_sz]).to(dtype=torch.int8)
kint8_ref = torch.randint(low, high, [seq_len, kvhead_num, head_sz]).to(dtype=torch.int8)

qscale_ref = torch.zeros(seq_len, head_num).to(dtype=torch.float32)
kscale_ref = torch.zeros(seq_len, kvhead_num).to(dtype=torch.float32)

qscale_out = torch.zeros(head_num, seq_len).to(dtype=torch.float32)
kscale_out = torch.zeros(kvhead_num, seq_len).to(dtype=torch.float32)


cl.profiling(True)

src = r'''

#pyeval f"#define KVGRP_SZ {grp_sz}"
#pyeval f"#define HEAD_SZ {head_sz}"
#pyeval f"#define HEAD_NUM {head_num}"
#pyeval f"#define KVHEAD_NUM {kvhead_num}"
#pyeval f"#define SEQ_BLK {seq_blk}"
#pyeval f"#define STATE_BLK {state_blk}"
#pyeval f"#define SEQ_LEN {seq_len}"
#pyeval f"#define SEQ_BLK_WG {seq_blk_wg}"
#define MAX_LOCAL_SZ 32


template<typename T, int M, int N>
void show(const matrix<T, M, N> mat) {
    printf("Matrix [%d, %d]:\n", M, N);
    for(int m = 0; m < M; m ++) {
        printf("\t[");
        for(int n = 0; n < N; n ++) {
            printf("%4d,", mat[m][n]);
        }
        printf("],\n");
    }
    printf("]\n");
}

template<typename T, int M, int N>
void show_float(const matrix<T, M, N> mat) {
    printf("Matrix [%d, %d]:\n", M, N);
    for(int m = 0; m < M; m ++) {
        printf("\t[");
        for(int n = 0; n < N; n ++) {
            printf("%8.4f,", mat[m][n]);
        }
        printf("],\n");
    }
    printf("]\n");
}

extern "C" _GENX_MAIN_ void quanQK(SurfaceIndex qkv [[type("buffer_t")]], SurfaceIndex qscale [[type("buffer_t")]], SurfaceIndex kscale [[type("buffer_t")]]) {
    auto id = cm_group_id(0)*cm_local_size(0) + cm_linear_local_id();
    if (id >= KVHEAD_NUM*SEQ_LEN)
        return;
    auto headkv = id%KVHEAD_NUM;
    auto head = id*KVGRP_SZ % HEAD_NUM;
    auto seq = id / KVHEAD_NUM;
    auto pitch = HEAD_SZ*sizeof(half);
    auto qoff = (seq * (HEAD_NUM + KVHEAD_NUM + KVHEAD_NUM) + head)*pitch;
    auto koff = (seq * (HEAD_NUM + KVHEAD_NUM + KVHEAD_NUM) + headkv + HEAD_NUM)*pitch;

    auto kscale_off = (headkv*SEQ_LEN + seq)*sizeof(float);
    auto qscale_off = (head*SEQ_LEN + seq)*sizeof(float);

    vector<half, HEAD_SZ> token;
    vector<float, 1> scaleV;

    auto quan_token= token.format<int8_t,2, HEAD_SZ>().row(0);

    #pragma unroll
    for(int i= 0;i<KVGRP_SZ;i++,qoff+=pitch, qscale_off += sizeof(float)*SEQ_LEN) {
        token.format<uint32_t>() = cm_load<uint, HEAD_SZ/2>(qkv, qoff);
        half max=cm_reduced_max<half>(cm_abs(token));
        quan_token =  cm_mul<int8_t>(token, (float)(127.0)/(float)(max));
        cm_store<uint32_t, HEAD_SZ/4>(qkv, qoff, quan_token.format<uint32_t>());
        scaleV[0] = (float)(max)/127.0;
        cm_store<uint32_t, 1>(qscale, qscale_off, scaleV.format<uint32_t>());
    }

    token.format<uint32_t>() = cm_load<uint, HEAD_SZ/2>(qkv, koff);
    half max=cm_reduced_max<half>(cm_abs(token));
    quan_token =  cm_mul<int8_t>(token, (float)(127.0)/(float)(max));
    cm_store<uint32_t, HEAD_SZ/4>(qkv, koff, quan_token.format<uint32_t>());
    scaleV[0] = (float)(max)/127.0;
    cm_store<uint32_t, 1>(kscale, kscale_off, scaleV.format<uint32_t>());
}


extern "C" _GENX_MAIN_ void Kmean(half* k_ptr [[type("svmptr_t")]], half* kmean_ptr [[type("svmptr_t")]]) {
#define SEQ_STEP 8
#define NO_LSC 1
    // q [B, L, H, S]
    auto kvhead = cm_group_id(0);
    auto sblk_idx = cm_group_id(1);
    auto lid = cm_linear_local_id();
    auto offset = ((lid * SEQ_BLK * KVHEAD_NUM + kvhead)*HEAD_SZ + sblk_idx*STATE_BLK);

    auto remaing_seq = (lid+1)*SEQ_BLK > SEQ_LEN ?  (SEQ_LEN-lid*SEQ_BLK): SEQ_BLK;
    constexpr uint BUF_SIZE = MAX_LOCAL_SZ*STATE_BLK*sizeof(float);
    cm_slm_init(BUF_SIZE);
    auto scratch_buf = cm_slm_alloc(BUF_SIZE);

#if NO_LSC
    vector <half, STATE_BLK> seq;
    vector <float, STATE_BLK> seq_f32;

#else
    matrix <half, 8, STATE_BLK> seq;
    matrix <float, 8, STATE_BLK> seq_f32;
    lsc::block_2d_desc<half, 1, SEQ_STEP, STATE_BLK> b2dq((k_ptr + offset), remaing_seq - 1,STATE_BLK*sizeof(half) - 1, KVHEAD_NUM*HEAD_SZ*sizeof(half) - 1, 0, 0);
#endif
    vector<float, STATE_BLK> seq_blk_sum = 0;
    auto pitch = KVHEAD_NUM*HEAD_SZ;

#if NO_LSC
    #pragma unroll
    for (int i = 0; i < SEQ_BLK; i++) {
        cm_svm_block_read(reinterpret_cast<svmptr_t>(k_ptr + offset + i* pitch), seq);
        seq_f32 = seq;
        seq_blk_sum += seq_f32;
        //# cm_load<lsc::Normal>(seq_blk.format<half>(), b2dq.set_block_y(i));
        //# seq_blk_f32 = seq_blk;
        //# #pragma unroll
        //# for (int r = 0; r<SEQ_STEP; r++) {
        //#     seq_blk_sum += seq_blk_f32[r];
        //# }
    }
#else
    #pragma unroll
    for (int i = 0; i < SEQ_BLK; i+=SEQ_STEP) {
        cm_load<lsc::Normal>(seq.format<half>(), b2dq.set_block_y(i));
        seq_f32 = seq;
        #pragma unroll
        for (int r = 0; r<SEQ_STEP; r++) {
             seq_blk_sum += seq_f32[r];
        }
    }
#endif
    cm_slm_block_write(scratch_buf, lid*STATE_BLK*sizeof(float), seq_blk_sum.format<float>());
    cm_barrier();
    if (lid == 0) {
        seq_blk_sum = 0;
        vector<float, STATE_BLK> tmpsum = 0;
        int off = 0;
        #pragma unroll
        for (int r = 0; r<MAX_LOCAL_SZ; r++, off +=STATE_BLK*sizeof(float)) {
            cm_slm_block_read(scratch_buf, GENX_NONE, off, tmpsum.format<float>());
            seq_blk_sum += tmpsum;
        }

        vector<half, STATE_BLK> kmean;
        kmean = seq_blk_sum / (float)(SEQ_LEN);
        cm_svm_block_write(reinterpret_cast<svmptr_t>(kmean_ptr + kvhead*HEAD_SZ+sblk_idx*STATE_BLK), kmean);
    }
}


extern "C" _GENX_MAIN_ void Kmean_1st(half* k_ptr [[type("svmptr_t")]], half* kmean_ptr [[type("svmptr_t")]]) {
#define SEQ_STEP 8
#define NO_LSC 1
    // q [B, L, H, S]
    auto kvhead = cm_group_id(0);
    auto sblk_idx = cm_group_id(1);
    auto lid = cm_linear_local_id();
    auto offset = ((lid * SEQ_BLK * KVHEAD_NUM + kvhead)*HEAD_SZ + sblk_idx*STATE_BLK);

    auto remaing_seq = (lid+1)*SEQ_BLK > SEQ_LEN ?  (SEQ_LEN-lid*SEQ_BLK): SEQ_BLK;
    constexpr uint BUF_SIZE = MAX_LOCAL_SZ*STATE_BLK*sizeof(float);
    cm_slm_init(BUF_SIZE);
    auto scratch_buf = cm_slm_alloc(BUF_SIZE);

#if NO_LSC
    vector <half, STATE_BLK> seq;
    vector <float, STATE_BLK> seq_f32;

#else
    matrix <half, 8, STATE_BLK> seq;
    matrix <float, 8, STATE_BLK> seq_f32;
    lsc::block_2d_desc<half, 1, SEQ_STEP, STATE_BLK> b2dq((k_ptr + offset), remaing_seq - 1,STATE_BLK*sizeof(half) - 1, KVHEAD_NUM*HEAD_SZ*sizeof(half) - 1, 0, 0);
#endif
    vector<float, STATE_BLK> seq_blk_sum = 0;
    auto pitch = KVHEAD_NUM*HEAD_SZ;

#if NO_LSC
    #pragma unroll
    for (int i = 0; i < SEQ_BLK; i++) {
        cm_svm_block_read(reinterpret_cast<svmptr_t>(k_ptr + offset + i* pitch), seq);
        seq_f32 = seq;
        seq_blk_sum += seq_f32;
        //# cm_load<lsc::Normal>(seq_blk.format<half>(), b2dq.set_block_y(i));
        //# seq_blk_f32 = seq_blk;
        //# #pragma unroll
        //# for (int r = 0; r<SEQ_STEP; r++) {
        //#     seq_blk_sum += seq_blk_f32[r];
        //# }
    }
#else
    #pragma unroll
    for (int i = 0; i < SEQ_BLK; i+=SEQ_STEP) {
        cm_load<lsc::Normal>(seq.format<half>(), b2dq.set_block_y(i));
        seq_f32 = seq;
        #pragma unroll
        for (int r = 0; r<SEQ_STEP; r++) {
             seq_blk_sum += seq_f32[r];
        }
    }
#endif
    cm_slm_block_write(scratch_buf, lid*STATE_BLK*sizeof(float), seq_blk_sum.format<float>());
    cm_barrier();
    if (lid == 0) {
        seq_blk_sum = 0;
        vector<float, STATE_BLK> tmpsum = 0;
        int off = 0;
        #pragma unroll
        for (int r = 0; r<MAX_LOCAL_SZ; r++, off +=STATE_BLK*sizeof(float)) {
            cm_slm_block_read(scratch_buf, GENX_NONE, off, tmpsum.format<float>());
            seq_blk_sum += tmpsum;
        }

        vector<half, STATE_BLK> kmean;
        kmean = seq_blk_sum / (float)(SEQ_LEN);
        cm_svm_block_write(reinterpret_cast<svmptr_t>(kmean_ptr + kvhead*HEAD_SZ+sblk_idx*STATE_BLK), kmean);
    }
}


'''

def pyeval(src):

    result_src = ""
    for line in src.splitlines():
        if line.startswith("#pyeval"):
            new_line = eval(line[8:])
            result_src += new_line + "\n"
        else:
            result_src += line + "\n"
    return result_src


for seq in range(seq_len):
    for h in range(head_num):
        hkv = h // (head_num//kvhead_num)
        qtoken=q[seq, h, :]
        qmax=torch.amax(qtoken.abs(), dim=0, keepdim=True)
        qtoken_int8=(qtoken/qmax*127.0).to(dtype=torch.int8)
        qint8_ref[seq,h,:]=qtoken_int8
        qscale_ref[seq, h]=float(qmax)/127.0

        ktoken=k[seq,hkv,:]
        kmax=torch.amax(ktoken.abs(), dim=0, keepdim=True)
        ktoken_int8=(ktoken/kmax*127.0).to(dtype=torch.int8)
        kint8_ref[seq,hkv,:]=ktoken_int8
        kscale_ref[seq, hkv]=float(kmax)/127.0


all_layers=[]
mem_size=0
while mem_size < 4e9:
    all_layers.append([
        cl.tensor(q.detach().numpy())
    ])
    mem_size += q.numel() * q.element_size()
t_klist = [cl.tensor(k.to(torch.float16).detach().numpy()) for _ in range(50)]
t_kmeanlist = [cl.tensor(torch.zeros(1, kvhead_num, head_sz).to(dtype=torch.float16).detach().numpy()) for _ in range(50)]
smmothk_local_sz = 32

qkv =  torch.cat((q,k,v), 1)
kmean = k.mean(dim=0, keepdim=True)

t_qkvlist = [cl.tensor(qkv.to(torch.float16).detach().numpy()) for _ in range(50)]
t_qscaleList = [cl.tensor(qscale_out.detach().numpy()) for _ in range(50)]
t_kscaleList = [cl.tensor(kscale_out.detach().numpy()) for _ in range(50)]


print(f'head_num:{head_num}, seq_len:{seq_len}')
seq_blk = (seq_len + smmothk_local_sz - 1) // smmothk_local_sz
seq_blk_wg=seq_blk
state_blk=32

cm_kernels = cl.kernels(pyeval(src), f"-cmc -mdump_asm -g2 ")

assert head_sz % state_blk == 0, f'headsz is multiple of 32'
lws = [1, 1, smmothk_local_sz]
gws = [kvhead_num, head_sz//state_blk, smmothk_local_sz]
print(f'GWS:{gws}, LWS:{lws}')
for i in range(50):
    cm_kernels.enqueue("Kmean", gws, lws, t_klist[i], t_kmeanlist[i])
lat=cl.finish()
ns=sum(lat[5:])/len(lat[5:])
print(ns)
check_close(kmean,torch.from_numpy(t_kmeanlist[0].numpy()))
print(f'-----------------------------------------------------------------------------------------------\n')
