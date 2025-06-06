
#include "cm_sdpa_common.hpp"

#ifdef CM_HAS_LSC_UNTYPED_2D
#define USE_LSC 1
#else
#define USE_LSC 0
#endif

extern "C" _GENX_MAIN_ void cm_sdpa_vlen(
    int* cu_seqlens [[type("svmptr_t")]],
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
    constexpr int num_heads = CMFLA_NUM_HEADS;
    constexpr int head_size = CMFLA_HEAD_SIZE;
    constexpr int num_kv_heads = CMFLA_NUM_KV_HEADS;

    //# query [q_len, num_heads, S]
    //#   key [kv_len, num_heads, S]
    //# value [kv_len, num_heads, S]

    constexpr uint K_SLM_SIZE = (4*kv_step * head_size * sizeof(half));
    constexpr uint V_SLM_SIZE = (4*kv_step * head_size * sizeof(half));
    constexpr uint Q_SLM_SIZE = 0;//(q_step * head_size * sizeof(half)) * local_size;

    cm_slm_init(K_SLM_SIZE + V_SLM_SIZE + Q_SLM_SIZE);

    auto slm_K = cm_slm_alloc(K_SLM_SIZE);
    auto slm_V = cm_slm_alloc(V_SLM_SIZE);

    auto h = cm_group_id(0);
    auto hkv = h / (num_heads/num_kv_heads);
    auto wg_local_id = cm_local_id(1);
    const int local_size = cm_local_size(1);

    auto wg_id = cm_group_id(1); // each work-group handles a sequence

#if CMFLA_WG_MAPPING
    // multiple work-groups are required to split a sequence, 
    // need to figure out which part of query-tokens to process
    auto wg_count = cm_group_count(1);
    constexpr int wg_seq_len = CMFLA_WG_SIZE * q_step;
    int seq_id = 0;
    int seq_start = cu_seqlens[seq_id];
    int seq_end;
    int wg_base = 0;
    int wg_num;
    while(wg_base < wg_count) {
        seq_end = cu_seqlens[seq_id + 1];
        // seq_start ~ seq_end
        auto sel_len = seq_end - seq_start;
        wg_num = (sel_len + wg_seq_len - 1) / wg_seq_len;
        if (wg_id < wg_base + wg_num) {
            // found the sequence to process
            break;
        }
        wg_base += wg_num;
        seq_start = seq_end;
        seq_id ++;
    }
    if (wg_base > wg_id) return;

    int kv_start = seq_start;
    int kv_seq_len = seq_end - seq_start;
    int q_start = seq_start + (wg_id - wg_base) * wg_seq_len + wg_local_id * q_step;
    int q_len = q_step;
    if (q_start + q_len > seq_end) {
        q_len = seq_end - q_start;
    }

    //printf("wg:%d.%d  q: %d, +%d   kv: %d, +%d\n", wg_id, wg_local_id, q_start, q_len, kv_start, kv_seq_len);

#else
    int q_start = cu_seqlens[wg_id] + wg_local_id * q_step;
    int q_len = q_step;
    if (q_start + q_len > cu_seqlens[wg_id + 1]) {
        q_len = cu_seqlens[wg_id + 1] - q_start;
    }
    auto kv_start = cu_seqlens[wg_id];
    int kv_seq_len = cu_seqlens[wg_id + 1] - kv_start;
#endif
    uint qo_offset = (q_start*num_heads + h)*head_size;
    uint kv_offset = (kv_start*num_kv_heads + hkv)*head_size;

#if USE_LSC == 1
    sdpa_kernel_lsc<false, CMFLA_WG_SIZE, num_heads, num_kv_heads, head_size>(
                                slm_K,
                                slm_V,
                                wg_local_id,
                                0, //q_start,
                                kv_seq_len, //kv_stop,
                                q_len, //q_len,
                                kv_seq_len, //kv_len,
                                reinterpret_cast<svmptr_t>(query + qo_offset),
                                reinterpret_cast<svmptr_t>(key + kv_offset),
                                reinterpret_cast<svmptr_t>(value + kv_offset),
                                reinterpret_cast<svmptr_t>(output + qo_offset));
#else
    sdpa_kernel<false, CMFLA_WG_SIZE, num_heads, num_kv_heads, head_size>(
                                slm_K,
                                slm_V,
                                wg_local_id,
                                0, //q_start,
                                kv_seq_len, //kv_stop,
                                q_len, //q_len,
                                kv_seq_len, //kv_len,
                                query,
                                key,
                                value,
                                output,
                                qo_offset * sizeof(half),
                                kv_offset * sizeof(half));
#endif
}
