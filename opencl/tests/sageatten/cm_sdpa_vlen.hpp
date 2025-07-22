
#include "cm_sdpa_common.hpp"

#ifdef CM_HAS_LSC_UNTYPED_2D
#define USE_LSC 1
#else
#define USE_LSC 0
#endif



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



extern "C" _GENX_MAIN_ _GENX_FLOAT_CONTROL_(CM_RTE) void cm_quantize_qk(int seqlen, SurfaceIndex q [[type("buffer_t")]], SurfaceIndex k [[type("buffer_t")]],
                                            SurfaceIndex qscale [[type("buffer_t")]], SurfaceIndex kscale [[type("buffer_t")]], SurfaceIndex kmean_ptr [[type("buffer_t")]]) {
    auto id = cm_group_id(0)*cm_local_size(0) + cm_linear_local_id();
    if (id >= CMFLA_NUM_KV_HEADS*seqlen)
        return;
    constexpr int KVGRP_SZ =  CMFLA_NUM_HEADS / CMFLA_NUM_KV_HEADS;
    auto headkv = id % CMFLA_NUM_KV_HEADS;
    auto head = id * KVGRP_SZ % CMFLA_NUM_HEADS;
    auto seq = id / CMFLA_NUM_KV_HEADS;
    auto pitch = CMFLA_HEAD_SIZE*sizeof(half);
    auto qoff = (seq * CMFLA_NUM_HEADS  + head)*pitch;
    auto koff = (seq * CMFLA_NUM_KV_HEADS + headkv)*pitch;

    auto kscale_off = (headkv*seqlen + seq)*sizeof(float);
    auto qscale_off = (head*seqlen + seq)*sizeof(float);

    vector<half, CMFLA_HEAD_SIZE> token;
    vector<float, 1> scaleV;
    constexpr int step = (CMFLA_HEAD_SIZE==64 ||  CMFLA_HEAD_SIZE ==128) ? CMFLA_HEAD_SIZE : REG_K_U8;

    auto quan_token= token.format<int8_t,2, CMFLA_HEAD_SIZE>().row(0);

    #pragma unroll
    for(int i= 0;i<KVGRP_SZ;i++,qoff+=pitch, qscale_off += sizeof(float)*seqlen) {
        cm_load_1d<CMFLA_HEAD_SIZE/2, step/2>(token.format<uint32_t>(), q, qoff);
        half max=cm_reduced_max<half>(cm_abs(token));
        quan_token =  cm_mul<int8_t>(token, (float)(127.0)/(float)(max));
        cm_store_1d<CMFLA_HEAD_SIZE/4, step/4>(quan_token.format<uint32_t>(), q, qoff);

        // cm_store<uint32_t, CMFLA_HEAD_SIZE/4>(qkv, qoff, quan_token.format<uint32_t>());
        scaleV[0] = (float)(max)/127.0;
        cm_store<uint32_t, 1>(qscale, qscale_off, scaleV.format<uint32_t>());
    }

    vector<half, CMFLA_HEAD_SIZE> kmean;
    cm_load_1d<CMFLA_HEAD_SIZE/2, step/2>(token.format<uint32_t>(), k, koff);
    cm_load_1d<CMFLA_HEAD_SIZE/2, step/2>(kmean.format<uint32_t>(), kmean_ptr, headkv*pitch);
    token -= kmean;
    half max=cm_reduced_max<half>(cm_abs(token));
    quan_token = cm_rnde<int8_t>(cm_mul<float>(token, (float)(127.0)/(float)(max)));
    cm_store_1d<CMFLA_HEAD_SIZE/4, step/4>(quan_token.format<uint32_t>(), k, koff);
    scaleV[0] = (float)(max)*scale_factor/float(127.0);
    cm_store<uint32_t, 1>(kscale, kscale_off, scaleV.format<uint32_t>());
}


extern "C" _GENX_MAIN_ void Kmean(int seqlen, int seq_blk, half* k_ptr [[type("svmptr_t")]], half* kmean_ptr [[type("svmptr_t")]]) {

    // q [B, L, H, S]
    auto kvhead = cm_group_id(0);
    auto sblk_idx = cm_group_id(1);
    auto lid = cm_linear_local_id();
    auto seq_start = lid * seq_blk;

    auto threads_cnt = (seqlen + seq_blk - 1) / seq_blk;
    auto offset = ((seq_start *CMFLA_NUM_KV_HEADS  + kvhead)*CMFLA_HEAD_SIZE + sblk_idx*CMFLA_STATE_BLK);
    k_ptr += offset;

    constexpr uint BUF_SIZE = LOCAL_SZ*CMFLA_STATE_BLK*sizeof(float);
    cm_slm_init(BUF_SIZE);
    auto scratch_buf = cm_slm_alloc(BUF_SIZE);

    vector <half, CMFLA_STATE_BLK> seq;
    vector <float, CMFLA_STATE_BLK> seq_f32;
    vector<float, CMFLA_STATE_BLK> seq_blk_sum = 0;
    //don't know why, when lowering down pitch can achive 385.64 GB/S, just a test.
    //TLB issue?
    //auto pitch = (TOTAL_HEADS-1)*HEAD_SZ;
    auto pitch = CMFLA_NUM_KV_HEADS*CMFLA_HEAD_SIZE;

    if (seq_start < seqlen) {
        auto remaing_seq = (seq_start + seq_blk ) > seqlen ?  (seqlen-seq_start): seq_blk;

        if (seq_blk == remaing_seq) {
            #pragma unroll(UNROLL_NUM)
            for (int i = 0; i < seq_blk; i++) {
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
        cm_slm_block_write(scratch_buf, lid*CMFLA_STATE_BLK*sizeof(float), seq_blk_sum.format<float>());
    }
    cm_barrier();
    if (lid == 0) {
        seq_blk_sum = 0;
        vector<float, CMFLA_STATE_BLK> tmpsum = 0;
        int off = 0;
        for (int r = 0; r<threads_cnt; r++, off +=CMFLA_STATE_BLK*sizeof(float)) {
            cm_slm_block_read(scratch_buf, GENX_NONE, off, tmpsum.format<float>());
            seq_blk_sum += tmpsum;
        }
        vector<half, CMFLA_STATE_BLK> kmean;
        kmean = seq_blk_sum / (float)(seqlen);
        cm_svm_block_write(reinterpret_cast<svmptr_t>(kmean_ptr + kvhead*CMFLA_HEAD_SIZE+sblk_idx*CMFLA_STATE_BLK), kmean);
    }
}

extern "C" _GENX_MAIN_ void cm_sdpa_qkv_fused(
    int seqlen,
#if USE_LSC == 1
    half* query [[type("svmptr_t")]],
    half* key [[type("svmptr_t")]],
    half* value [[type("svmptr_t")]],
    float* dqscale_q [[type("svmptr_t")]],
    float* dqscale_k [[type("svmptr_t")]],
    half* output [[type("svmptr_t")]]
#else
    SurfaceIndex query [[type("buffer_t")]],
    SurfaceIndex key [[type("buffer_t")]],
    SurfaceIndex value [[type("buffer_t")]],

    SurfaceIndex dqscale_q [[type("buffer_t")]],
    SurfaceIndex dqscale_k [[type("buffer_t")]],
    SurfaceIndex output [[type("buffer_t")]]
#endif
    ) {
    constexpr int is_causal = CMFLA_IS_CAUSAL;
    constexpr int num_heads = CMFLA_NUM_HEADS;
    constexpr int head_size = CMFLA_HEAD_SIZE;
    constexpr int num_kv_heads = CMFLA_NUM_KV_HEADS;

    //# query [q_len, num_heads, S]
    //#   key [kv_len, num_heads, S]
    //# value [kv_len, num_heads, S]
#if USE_LSC != 1
    constexpr uint K_SLM_SIZE = (4*kv_step * head_size * sizeof(int8_t));
    constexpr uint V_SLM_SIZE = (4*kv_step * head_size * sizeof(half));
    constexpr uint Q_SLM_SIZE = 0;//(q_step * head_size * sizeof(half)) * local_size;

    cm_slm_init(K_SLM_SIZE + V_SLM_SIZE + Q_SLM_SIZE);

    auto slm_K = cm_slm_alloc(K_SLM_SIZE);
    auto slm_V = cm_slm_alloc(V_SLM_SIZE);
#endif

    auto batch = cm_group_id(0);
    auto h = cm_group_id(1);
    auto hkv = h / (num_heads/num_kv_heads);
    auto wg_id = cm_group_id(2); // each work-group handles a sequence
    auto wg_local_id = cm_local_id(2);
    int local_size = cm_local_size(2);

    int q_start, kv_start, kv_seq_len, q_len;

    // multiple work-groups are required to split a sequence,
    // need to figure out which part of query-tokens to process
    int wg_seq_len = local_size * q_step;
    kv_start = 0;
    kv_seq_len = seqlen;
    q_start = (wg_id * local_size + wg_local_id) * q_step;
    q_len = q_step;
    if (q_start + q_len > seqlen) {
        q_len = seqlen - q_start;
    }
    //printf("wg:%d.%d  q: %d, +%d   kv: %d, +%d\n", wg_id, wg_local_id, q_start, q_len, kv_start, kv_seq_len);

    // qkv is fused
    int kv_stop = kv_seq_len;
    if (is_causal) {
        kv_stop = (wg_id + 1) * wg_seq_len;
        if (kv_stop > kv_seq_len) kv_stop = kv_seq_len;
    }

    // qkv fused
    constexpr uint num_total_heads = num_heads + num_kv_heads * 2;
    // uint q_offset = (q_start*num_total_heads + h)*head_size;
    // uint k_offset = (kv_start*num_total_heads + num_heads + hkv)*head_size;
    // uint v_offset = (kv_start*num_total_heads + num_heads + num_kv_heads + hkv)*head_size;

    uint q_offset = (q_start*num_heads + h)*head_size;
    uint k_offset = (kv_start*num_kv_heads + hkv)*head_size;
    uint v_offset = k_offset;
    uint o_offset = (q_start*num_heads + h)*head_size;
    //# scale [head, sequence, 1]

    uint qscale_offset =h*seqlen + q_start;
    uint kscale_offset = hkv*seqlen;

#if USE_LSC == 1
    sdpa_kernel_lsc_prefetch<is_causal, num_heads, num_kv_heads, head_size, 0, 16>(
                                wg_local_id,
                                q_start, //q_start,
                                kv_stop,
                                q_len, //q_len,
                                kv_seq_len, //kv_len,
                                reinterpret_cast<svmptr_t>(query + q_offset),
                                reinterpret_cast<svmptr_t>(key + k_offset),
                                reinterpret_cast<svmptr_t>(value + v_offset),
                                reinterpret_cast<svmptr_t>(dqscale_q + qscale_offset),
                                reinterpret_cast<svmptr_t>(dqscale_k + kscale_offset),
                                reinterpret_cast<svmptr_t>(output + o_offset));
#else
    sdpa_kernel<is_causal, num_heads, num_kv_heads, head_size, 0>(
                                slm_K,
                                slm_V,
                                wg_local_id,
                                local_size,
                                q_start,
                                kv_stop,
                                q_len, //q_len,
                                kv_seq_len, //kv_len,
                                query,
                                key,
                                value,
                                dqscale_q,
                                dqscale_k,
                                output,
                                q_offset * sizeof(half),
                                k_offset * sizeof(half),
                                v_offset * sizeof(half),
                                qscale_offset * sizeof(float),
                                kscale_offset * sizeof(float),
                                o_offset * sizeof(half),
                                seqlen);
#endif
}
