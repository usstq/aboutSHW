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



src = r'''
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



template <int NElts, int step=NElts>
CM_INLINE void cm_load_1d(vector_ref<uint32_t, NElts> out, SurfaceIndex base, uint offset) {
    auto mat = out.format<uint32_t, NElts/step, step>();
    #pragma unroll
    for (int r = 0; r < NElts/step; r++, offset += step*sizeof(int32_t)) {
        mat.row(r).format<uint32_t>() = cm_load<uint32_t, step>(base, offset);
    }
}

template <int NElts, int step=NElts>
CM_INLINE void cm_store_1d(vector_ref<uint32_t, NElts> in, SurfaceIndex base, uint offset) {
    auto mat = in.format<uint32_t, NElts/step, step>();

    #pragma unroll
    for (int r = 0; r < NElts/step; r++, offset += step*sizeof(int32_t)) {
        cm_store<uint32_t, step>(base, offset,  mat.row(r).format<uint32_t>());
    }
}


extern "C" _GENX_MAIN_ _GENX_FLOAT_CONTROL_(CM_RTE) void test_rnd(SurfaceIndex ptr [[type("buffer_t")]])
{
        vector<half, 8> test;
        vector<float, 8> test_f32;

        vector<int8_t, 8> test_u8;
        test.format<uint32_t>() = cm_load<uint, 4>(ptr, 0);
        test_f32 = test;
        test_u8 = cm_rnde<int8_t>(test_f32);
        show<int8_t, 1, 8>(test_u8);
}

extern "C" _GENX_MAIN_ _GENX_FLOAT_CONTROL_(CM_RTE) void quanQK(SurfaceIndex q [[type("buffer_t")]], SurfaceIndex k [[type("buffer_t")]], SurfaceIndex qscale [[type("buffer_t")]], SurfaceIndex kscale [[type("buffer_t")]], SurfaceIndex kmean_ptr [[type("buffer_t")]]) {
    auto id = cm_group_id(0)*cm_local_size(0) + cm_linear_local_id();
    if (id >= KVHEAD_NUM*SEQ_LEN)
        return;
    auto headkv = id%KVHEAD_NUM;
    auto head = id*KVGRP_SZ % HEAD_NUM;
    auto seq = id / KVHEAD_NUM;
    auto pitch = HEAD_SZ*sizeof(half);
    auto qoff = (seq * HEAD_NUM + head)*pitch;
    auto koff = (seq * KVHEAD_NUM + headkv)*pitch;
    auto kscale_off = (headkv*SEQ_LEN + seq)*sizeof(float);
    auto qscale_off = (head*SEQ_LEN + seq)*sizeof(float);

    vector<half, HEAD_SZ> token;
    vector<float, 1> scaleV;
    constexpr int step = (HEAD_SZ==64 ||  HEAD_SZ ==128) ? HEAD_SZ : 32;
    auto quan_token= token.format<int8_t,2, HEAD_SZ>().row(0);

    #pragma unroll
    for(int i= 0;i<KVGRP_SZ;i++,qoff+=pitch, qscale_off += sizeof(float)*SEQ_LEN) {
        //token.format<uint32_t>() = cm_load<uint, HEAD_SZ/2>(q, qoff);
        cm_load_1d<HEAD_SZ/2, step/2>(token.format<uint32_t>(), q, qoff);

        half max=cm_reduced_max<half>(cm_abs(token));
        quan_token =  cm_mul<int8_t>(token, (float)(127.0)/(float)(max));
        cm_store_1d<HEAD_SZ/4, step/4>(quan_token.format<uint32_t>(), q, qoff);
        //cm_store<uint32_t, HEAD_SZ/4>(qkv, qoff, quan_token.format<uint32_t>());
        scaleV[0] = (float)(max)/127.0;
        cm_store<uint32_t, 1>(qscale, qscale_off, scaleV.format<uint32_t>());
    }
    vector<half, HEAD_SZ> kmean;
    //token.format<uint32_t>() = cm_load<uint, HEAD_SZ/2>(k, koff);
    cm_load_1d<HEAD_SZ/2, step/2>(token.format<uint32_t>(), k, koff);
    //kmean.format<uint32_t>() = cm_load<uint, HEAD_SZ/2>(kmean_ptr, headkv*pitch);
    cm_load_1d<HEAD_SZ/2, step/2>(kmean.format<uint32_t>(), kmean_ptr, headkv*pitch);
    token = token - kmean;
    half max=cm_reduced_max<half>(cm_abs(token));
    quan_token =  cm_rnde<int8_t>(cm_mul<float>(token, (float)(127.0)/(float)(max)));

    //cm_store<uint32_t, HEAD_SZ/4>(k, koff, quan_token.format<uint32_t>());
    cm_store_1d<HEAD_SZ/4, step/4>(quan_token.format<uint32_t>(), k, koff);
    scaleV[0] = (float)(max)/127.0;
    cm_store<uint32_t, 1>(kscale, kscale_off, scaleV.format<uint32_t>());
}


extern "C" _GENX_MAIN_ void Kmean(half* k_ptr [[type("svmptr_t")]], half* kmean_ptr [[type("svmptr_t")]]) {

    // k [B, L, H, S]
    auto kvhead = cm_group_id(0);
    auto sblk_idx = cm_group_id(1);
    auto lid = cm_linear_local_id();
    auto seq_start = lid * SEQ_BLK;

    auto sum_threads = (SEQ_LEN + SEQ_BLK - 1) / SEQ_BLK;
    auto offset = ((seq_start *KVHEAD_NUM  + kvhead )*HEAD_SZ + sblk_idx*STATE_BLK);

    k_ptr += offset;

    constexpr uint BUF_SIZE = LOCAL_SZ*STATE_BLK*sizeof(float);
    cm_slm_init(BUF_SIZE);
    auto scratch_buf = cm_slm_alloc(BUF_SIZE);

    vector <half, STATE_BLK> seq;
    vector <float, STATE_BLK> seq_f32;
    vector<float, STATE_BLK> seq_blk_sum = 0;
    //don't know why, when lowering down pitch can achive 385.64 GB/S, just a test.
    //TLB issue?
    auto pitch = KVHEAD_NUM*HEAD_SZ;

    if (seq_start < SEQ_LEN) {
        auto remaing_seq = (seq_start + SEQ_BLK ) > SEQ_LEN ?  (SEQ_LEN-seq_start): SEQ_BLK;

        if (SEQ_BLK == remaing_seq) {
            #pragma unroll(UNROLL_NUM)
            for (int i = 0; i < SEQ_BLK; i++) {
                cm_svm_block_read(reinterpret_cast<svmptr_t>(k_ptr), seq);
                seq_f32 = seq;
                seq_blk_sum += seq_f32;
                k_ptr += pitch;
            }
        } else {
            for (int i = 0; i < remaing_seq; i++) {
                cm_svm_block_read(reinterpret_cast<svmptr_t>(k_ptr), seq);
                seq_f32 = seq;
                seq_blk_sum += seq_f32;
                k_ptr += pitch;
            }
        }
        cm_slm_block_write(scratch_buf, lid*STATE_BLK*sizeof(float), seq_blk_sum.format<float>());
    }
    cm_barrier();
    if (lid == 0) {
        seq_blk_sum = 0;
        vector<float, STATE_BLK> tmpsum = 0;
        int off = 0;
        for (int r = 0; r<sum_threads; r++, off +=STATE_BLK*sizeof(float)) {
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

def test_sage_quan(seq_len, head_num, kvhead_num, head_sz, rep=10, checkacc=True):
    unroll_num = 32
    smmothk_local_sz = 64
    local_sz=64
    state_blk=32
    seq_blk = (seq_len + smmothk_local_sz - 1) // smmothk_local_sz
    seq_blk = (seq_blk + unroll_num - 1) // unroll_num * unroll_num


    grp_sz = head_num//kvhead_num
    low=-2
    high = 3

    q_factor = torch.randint(high+3, high+9, [seq_len, head_num, head_sz]).to(dtype=torch.float16)
    kv_factor = torch.randint(high+3, high+9, [seq_len, kvhead_num, head_sz]).to(dtype=torch.float16)

    q = torch.randint(low, high, [seq_len, head_num, head_sz]).to(dtype=torch.float16) / q_factor
    k = torch.randint(low, high, [seq_len, kvhead_num, head_sz]).to(dtype=torch.float16) / kv_factor
    k[:,:,2:32] += 100
    v = torch.randint(low, high, [seq_len, kvhead_num, head_sz]).to(dtype=torch.float16) / kv_factor
    qkv =  torch.cat((q,k,v), 1)

    qint8_ref = torch.randint(low, high, [seq_len, head_num, head_sz]).to(dtype=torch.int8)
    kint8_ref = torch.randint(low, high, [seq_len, kvhead_num, head_sz]).to(dtype=torch.int8)

    qscale_ref = torch.zeros(seq_len, head_num).to(dtype=torch.float32)
    kscale_ref = torch.zeros(seq_len, kvhead_num).to(dtype=torch.float32)

    qscale_out = torch.zeros(head_num, seq_len).to(dtype=torch.float32)
    kscale_out = torch.zeros(kvhead_num, seq_len).to(dtype=torch.float32)


    k_tmp_ref = torch.randint(low, high, [seq_len, kvhead_num, head_sz]).to(dtype=torch.float32)


    cl.profiling(True)


    kmean_ref = k.mean(dim=0, keepdim=True)
    smoothk = k - kmean_ref
    for seq in range(seq_len):
        for h in range(head_num):
            hkv = h // (head_num//kvhead_num)
            qtoken=q[seq, h, :]
            qmax=torch.amax(qtoken.abs(), dim=0, keepdim=True)
            qtoken = qtoken.to(dtype=torch.float32)
            qtoken_int8=(qtoken/float(qmax)*float(127.0)).to(dtype=torch.int8)
            qint8_ref[seq,h,:]=qtoken_int8
            qscale_ref[seq, h]=float(qmax)/127.0

            ktoken=smoothk[seq,hkv,:].to(dtype=torch.float32)
            kmax=torch.amax(ktoken.abs(), dim=0, keepdim=True)
            ktoken_f32 = (ktoken/float(kmax)*float(127.0)).to(dtype=torch.float32)
            ktoken_int8=(torch.round(ktoken/float(kmax)*float(127.0))).to(dtype=torch.int8)
            kint8_ref[seq,hkv,:]=ktoken_int8
            kscale_ref[seq, hkv]=float(kmax)/127.0
            k_tmp_ref[seq,hkv,:] = ktoken_f32
    assert smoothk.shape == k.shape



    t_kmeanlist = [cl.tensor(torch.zeros(1, kvhead_num, head_sz).to(dtype=torch.float16).detach().numpy()) for _ in range(rep)]
    t_qkvlist = [cl.tensor(qkv.to(torch.float16).detach().numpy()) for _ in range(rep)]
    t_qlist = [cl.tensor(q.to(torch.float16).detach().numpy()) for _ in range(rep)]
    t_klist = [cl.tensor(k.to(torch.float16).detach().numpy()) for _ in range(rep)]


    t_qscaleList = [cl.tensor(qscale_out.detach().numpy()) for _ in range(rep)]
    t_kscaleList = [cl.tensor(kscale_out.detach().numpy()) for _ in range(rep)]
    t_kf32_tmp = cl.tensor(torch.zeros(seq_len, kvhead_num, head_sz).to(dtype=torch.float32).detach().numpy())

    # all_layers=[]
    # mem_size=0
    # while mem_size < 4e9:
    #     all_layers.append([
    #         cl.tensor(q.detach().numpy())
    #     ])
    #     mem_size += q.numel() * q.element_size()


    print(f'head_num:{head_num}, seq_len:{seq_len}')
    cm_kernels = cl.kernels(src, f'-cmc -mdump_asm -g2'
                             f' -DKVGRP_SZ={grp_sz}'
                            f' -DHEAD_SZ={head_sz}'
                            f' -DHEAD_NUM={head_num}'
                            f' -DKVHEAD_NUM={kvhead_num}'
                            f' -DSEQ_BLK={seq_blk}'
                            f' -DSTATE_BLK={state_blk}'
                            f' -DSEQ_LEN={seq_len}'
                            f' -DLOCAL_SZ={smmothk_local_sz}'
                            f' -DUNROLL_NUM={unroll_num}')
    assert head_sz % state_blk == 0, f'headsz is multiple of 32'

    quan_lws = [local_sz]
    tokens_align_up=(kvhead_num*seq_len+local_sz-1)//local_sz*local_sz
    quan_gws = [tokens_align_up]

    mean_lws = [1, 1, smmothk_local_sz]
    mean_gws = [kvhead_num, head_sz//state_blk, smmothk_local_sz]

    print(f'MEAN_GWS:{mean_gws}, MEAN_LWS:{mean_lws} seq_blk:{seq_blk}')
    print(f'QUAN_GWS:{quan_gws}, QUAN_LWS:{quan_lws}')

    for i in range(rep):
        cm_kernels.enqueue("Kmean", mean_gws, mean_lws, t_klist[i], t_kmeanlist[i])
    for i in range(rep):
        cm_kernels.enqueue("quanQK", quan_gws, quan_lws, t_qlist[i], t_klist[i], t_qscaleList[i], t_kscaleList[i], t_kmeanlist[i])

    lat=cl.finish()
    lat_mean= lat[0:rep]
    lat_quantize= lat[rep:2*rep]

    ns_mean=sum(lat_mean[5:])/len(lat_mean[5:])
    ns_quan=sum(lat_quantize[5:])/len(lat_quantize[5:])

    rdbytes_K=seq_len*kvhead_num*head_sz*2
    rdbytes_V=seq_len*head_num*head_sz*2
    print(f'-----------------------------------------------------------------------------------------------')
    print(f'[Kmean]avg latency:{ns_mean*1e-3:.2f} us, read:{rdbytes_K/ns_mean:.2f} GB/S')
    print(f'[Quantize]avg latency:{ns_quan*1e-3:.2f} us, read:{(rdbytes_K+rdbytes_V)/ns_quan:.2f} GB/S write: {((rdbytes_K+rdbytes_V))/2/ns_quan:.2f} GB/S')
    print(f'-----------------------------------------------------------------------------------------------')


    qint8_out=t_qlist[0].numpy()[:,:,0:head_sz//2].view(np.int8).reshape((seq_len, head_num, head_sz))
    kint8_out=t_klist[0].numpy()[:,:,0:head_sz//2].view(np.int8).reshape((seq_len, kvhead_num, head_sz))

    if checkacc:
        check_close(qint8_ref,torch.from_numpy(qint8_out))
        check_close(kmean_ref,torch.from_numpy(t_kmeanlist[0].numpy()))
        check_close(kscale_ref.transpose(0, 1),torch.from_numpy(t_kscaleList[0].numpy()))
        check_close(qscale_ref.transpose(0, 1),torch.from_numpy(t_qscaleList[0].numpy()))
        # rounding to even and calculation error in float could introduce error of 1
        check_close(kint8_ref,torch.from_numpy(kint8_out), atol=1)

test_sage_quan(128, 28, 28, 96)
test_sage_quan(113, 28, 28, 128)
test_sage_quan(8192, 28, 4, 128, 8)


