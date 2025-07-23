
#include "cm_sdpa_common.hpp"

#ifdef CM_HAS_LSC_UNTYPED_2D
#define USE_LSC 1
#else
#define USE_LSC 0
#endif



extern "C" _GENX_MAIN_ void cm_sdpa_qkv_fused(
    int seqlen,
#if USE_LSC == 1
    half* query [[type("svmptr_t")]],
    half* key [[type("svmptr_t")]],
    half* value [[type("svmptr_t")]],
    half* output [[type("svmptr_t")]]
#else
    SurfaceIndex query [[type("buffer_t")]],
    SurfaceIndex key [[type("buffer_t")]],
    SurfaceIndex value [[type("buffer_t")]],
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
    constexpr uint K_SLM_SIZE = (4*kv_step * head_size * sizeof(half));
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
    uint v_offset = (kv_start*num_total_heads + num_heads + num_kv_heads + hkv)*head_size;

    uint q_offset = (q_start*num_heads + h)*head_size;
    uint k_offset = (kv_start*num_kv_heads + hkv)*head_size;
    uint o_offset = (q_start*num_heads + h)*head_size;

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
                                output,
                                q_offset * sizeof(half),
                                k_offset * sizeof(half),
                                v_offset * sizeof(half),
                                o_offset * sizeof(half));
#endif
}
