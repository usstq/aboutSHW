
#include "cm_sdpa_common.hpp"

static_assert(q_step == REG_N);





#ifdef CM_HAS_LSC_UNTYPED_2D
extern "C" _GENX_MAIN_ void cm_sdpa(
    int q_len,
    int kv_len,
    half* query [[type("svmptr_t")]],
    half* key [[type("svmptr_t")]],
    half* value [[type("svmptr_t")]],
    half* output [[type("svmptr_t")]],
    __global uint64_t* cminfo [[type("svmptr_t")]]
    ) {
    constexpr int num_heads = CMFLA_NUM_HEADS;
    constexpr int head_size = CMFLA_HEAD_SIZE;
    constexpr int num_kv_heads = CMFLA_NUM_KV_HEADS;

    CMTracer_begin(&cminfo);
    //# query [batch, q_len, num_heads, S]
    //#   key [batch, kv_len, num_heads, S]
    //# value [batch, kv_len, num_heads, S]
    //# to load Q

    constexpr uint K_SLM_SIZE = (4*kv_step * head_size * sizeof(half));
    constexpr uint V_SLM_SIZE = (4*kv_step * head_size * sizeof(half));
    constexpr uint Q_SLM_SIZE = 0;//(q_step * head_size * sizeof(half)) * local_size;

    cm_slm_init(K_SLM_SIZE + V_SLM_SIZE + Q_SLM_SIZE);

    auto slm_K = cm_slm_alloc(K_SLM_SIZE);
    auto slm_V = cm_slm_alloc(V_SLM_SIZE);

    auto batch = cm_group_id(0);
    auto h = cm_group_id(1);
    auto hkv = h / (num_heads/num_kv_heads);
    auto wg_local_id = cm_local_id(2);
    const int local_size = cm_local_size(2);
    auto q_group_id = cm_group_id(2);
    auto q_start = (q_group_id * local_size + wg_local_id) * q_step;

    auto q_base = reinterpret_cast<svmptr_t>(query + ((batch*q_len + q_start)*num_heads + h)*head_size);
    auto k_base = reinterpret_cast<svmptr_t>(key + (batch*num_kv_heads*kv_len + hkv)*head_size);
    auto o_base = reinterpret_cast<svmptr_t>(output + ((batch * q_len + q_start)*num_heads + h)*head_size);
    auto v_base = reinterpret_cast<svmptr_t>(value + (batch*num_kv_heads*kv_len + hkv)*head_size);

    int kv_stop = kv_len;
    if constexpr (causal_mask) {
        kv_stop = (q_group_id + 1) * local_size * q_step;
        if (kv_stop > kv_len) kv_stop = kv_len;
    }

    sdpa_kernel_lsc<causal_mask, num_heads, num_kv_heads, head_size, 0>(
                                slm_K,
                                slm_V,
                                wg_local_id,
                                local_size,
                                q_start,
                                kv_stop,
                                q_len,
                                kv_len,
                                q_base,
                                k_base,
                                v_base,
                                o_base);

    // if (i == args_verbose) return;
    CMTracer_end(&cminfo);
}

#else // no CM_HAS_LSC_UNTYPED_2D available

extern "C" _GENX_MAIN_ void cm_sdpa(
    int q_len,
    int kv_len,
    SurfaceIndex query [[type("buffer_t")]],
    SurfaceIndex key [[type("buffer_t")]],
    SurfaceIndex value [[type("buffer_t")]],
    SurfaceIndex output [[type("buffer_t")]],
    __global uint64_t* cminfo [[type("svmptr_t")]]
    ) {
    CMTracer_begin(&cminfo);
    //# query [batch, q_len, num_heads, S]
    //#   key [batch, kv_len, num_heads, S]
    //# value [batch, kv_len, num_heads, S]
    //# to load Q
    constexpr int num_heads = CMFLA_NUM_HEADS;
    constexpr int head_size = CMFLA_HEAD_SIZE;
    constexpr int num_kv_heads = CMFLA_NUM_KV_HEADS;

    constexpr uint K_SLM_SIZE = (4*kv_step * head_size * sizeof(half));
    constexpr uint V_SLM_SIZE = (4*kv_step * head_size * sizeof(half));
    constexpr uint Q_SLM_SIZE = 0;//(q_step * head_size * sizeof(half)) * local_size;

    cm_slm_init(K_SLM_SIZE + V_SLM_SIZE + Q_SLM_SIZE);

    auto slm_K = cm_slm_alloc(K_SLM_SIZE);
    auto slm_V = cm_slm_alloc(V_SLM_SIZE);

    auto batch = cm_group_id(0);
    auto h = cm_group_id(1);
    auto hkv = h / (num_heads/num_kv_heads);
    auto wg_local_id = cm_local_id(2);
    const int local_size = cm_local_size(2);
    auto q_group_id = cm_group_id(2);
    auto q_start = (q_group_id * local_size + wg_local_id) * q_step;

    auto qo_off = ((batch * q_len + q_start) * num_heads + h)*head_size*sizeof(half);
    auto kv_off = (batch * num_kv_heads*kv_len + hkv)*head_size*sizeof(half);

    int kv_stop = kv_len;
    if constexpr (causal_mask) {
        kv_stop = (q_group_id + 1) * local_size * q_step;
        if (kv_stop > kv_len) kv_stop = kv_len;
    }

    sdpa_kernel<causal_mask, WG_SIZE_HINT, num_heads, num_kv_heads, head_size>(
                                slm_K,
                                slm_V,
                                wg_local_id,
                                q_start,
                                kv_stop,
                                q_len,
                                kv_len,
                                query,
                                key,
                                value,
                                output,
                                qo_off,
                                kv_off);

    // if (i == args_verbose) return;
    CMTracer_end(&cminfo);
}

#endif
