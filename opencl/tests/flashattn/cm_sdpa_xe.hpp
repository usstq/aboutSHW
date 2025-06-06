
#include "cm_sdpa_common.hpp"

static_assert(q_step == REG_N);

template<int num_Qt, int _q_step = REG_N, int _kv_step = REG_K>
inline matrix<float, _kv_step, _q_step> ugemm_KQ(uint slm_K, matrix_ref<half, num_Qt, REG_K*REG_N> Qt, uint slm_offset = 0) {
    matrix<float, _kv_step, _q_step> St;
    constexpr int num_K = _kv_step/REG_M;
    auto St2 = St.format<float, num_K, REG_M*REG_N>();

    matrix<half, num_K, REG_M * REG_K> Kmat;
    cm_slm_block_read(slm_K, GENX_NONE, slm_offset, Kmat.format<half>());
    #pragma unroll
    for(int k = 0; k < num_K; k++)
        St2.row(k) = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount, float>(0, Qt[0].format<int32_t>(), Kmat[k].format<int32_t>());

    #pragma unroll
    for(int ri = 1; ri < num_Qt; ri++) {
        cm_slm_block_read(slm_K, GENX_NONE, slm_offset + ri * Kmat.n_elems() * sizeof(half), Kmat.format<half>());
        #pragma unroll
        for(int k = 0; k < num_K; k++) {
            St2.row(k) = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount, float>(St2.row(k), Qt[ri].format<int32_t>(), Kmat[k].format<int32_t>());
        }
    }
    return St;
}

template<int num_P_tiles = REG_N/REG_M>
inline void ugemm_PV0(uint slm_V, matrix_ref<half, REG_N, REG_K> P, matrix_ref<float, head_size/REG_N*num_P_tiles, REG_M*REG_N> rO, uint slm_offset = 0) {
    auto P2 = P.format<half, num_P_tiles, REG_M * REG_K>();
    #pragma unroll
    for(int k = 0, ri = 0; k < head_size; k += REG_N, ri += num_P_tiles) {
        matrix<half, REG_K/2, REG_N*2> Vmat;
        cm_slm_block_read(slm_V, GENX_NONE, slm_offset + REG_K*k*sizeof(half), Vmat.format<half>());

        #pragma unroll
        for(int p = 0; p < num_P_tiles; p++) {
            rO[ri + p] = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount, float>(
                            0,
                            Vmat.format<int32_t>(),
                            P2.row(p).format<int32_t>());
            //show(rO[ri + p].format<float, REG_M, REG_N>());
        }
    }
}

template<int num_P_tiles = REG_N/REG_M>
inline void ugemm_PV1(uint slm_V, matrix_ref<half, REG_N, REG_K> P, vector_ref<float, REG_N> max_comp,
                      matrix_ref<float, head_size/REG_N*num_P_tiles, REG_M*REG_N> rO, uint slm_offset = 0) {
    auto P2 = P.format<half, num_P_tiles, REG_M * REG_K>();
    #pragma unroll
    for(int k = 0, ri=0; k < head_size; k += REG_N, ri += num_P_tiles) {
        matrix<half, REG_K/2, REG_N*2> Vmat;
        cm_slm_block_read(slm_V, GENX_NONE, slm_offset + REG_K*k*sizeof(half), Vmat.format<half>());

        //# compensate cur_O
        //  matrix <float, head_size/REG_K*2, REG_M*REG_N> rO;
        #pragma unroll
        for(int p = 0; p < num_P_tiles; p++) {
            auto cO = rO[ri + p].format<float, REG_M, REG_N>();
            #pragma unroll
            for(int r = 0; r < REG_M; r++)
                cO.row(r) = cm_mul<float>(cO.row(r), max_comp[r + p*REG_M]);
        }

        //show(rO[ri].format<float, REG_M, REG_N>());

        //# show(cur_O.format<float, 2*REG_M, REG_N>()); return;
        #pragma unroll
        for(int p = 0; p < num_P_tiles; p++) {
            rO[ri + p] = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount>(
                        rO[ri + p].format<float>(),
                        Vmat.format<int32_t>(),
                        P2.row(p).format<int32_t>());
            //if (kv_pos == args_verbose) show(rO[ri + p].format<float, REG_M, REG_N>());
        }
        // if (kv_pos == args_verbose) show(cur_O.format<float, 2*REG_M, REG_N>());
    }
}

template<typename T, int rows, int cols>
vector<float, cols> online_softmax_update(matrix_ref<T, rows, cols> St, vector_ref<T, cols> cur_max, vector_ref<T, cols> cur_sum) {
    vector<float, cols> new_max_t;
    new_max_t = cm_max<float>(St[0], St[1]);
    for(int r = 2; r < St.n_rows(); r++) new_max_t = cm_max<float>(new_max_t, St[r]);
    new_max_t = cm_max<float>(new_max_t, cur_max);

    // Pt = torch.exp(St - new_max)
    constexpr float log2e = 1.4426950408889634f;
    for(int r = 0; r < St.n_rows(); r++) St[r] = cm_exp((St[r] - new_max_t)*log2e);

    vector<float, cols> row_sum_t;
    row_sum_t = cm_add<float>(St[0], St[1]);
    for(int r = 2; r < St.n_rows(); r++) row_sum_t = cm_add<float>(row_sum_t, St[r]);

    vector<float, cols> max_comp;
    max_comp = cm_exp((cur_max - new_max_t)*log2e);
    cur_sum = cm_mul<float>(cur_sum, max_comp);
    cur_sum = cm_add<float>(cur_sum, row_sum_t);
    cur_max = new_max_t;
    return max_comp;
}

template<bool use_causal_mask, int local_size>
void sdpa_kernel(
    uint slm_K,
    uint slm_V,
    int wg_local_id,
    int q_start,
    int kv_stop,
    int head_base_id,
    int q_len,
    int kv_len,
    SurfaceIndex query [[type("buffer_t")]],
    SurfaceIndex key [[type("buffer_t")]],
    SurfaceIndex value [[type("buffer_t")]],
    SurfaceIndex output [[type("buffer_t")]],
    uint qo_off,
    uint kv_off) {

    uint qo_pitch = num_heads * head_size * sizeof(half);
    uint kv_pitch = num_kv_heads * head_size * sizeof(half);
    vector<float, q_step> cur_max;
    vector<float, q_step> cur_sum;

    cur_max = -3e38f;
    cur_sum = 0;

    matrix<half, head_size/REG_K, REG_K*REG_N> rQ;
    auto q_tokens_left = q_len - q_start;
    static_assert(q_step == REG_N);
    static_assert(kv_step == REG_K);

    if (q_tokens_left < 0) q_tokens_left = 0;
    if (q_tokens_left > q_step) q_tokens_left = q_step;

    if (q_tokens_left > 0) {
        // load as many as possible given one address
        matrix<uint, q_step, head_size/2> QmatI32;
        cm_load_2d(QmatI32, query, qo_off, qo_pitch);
        #pragma unroll
        for(int k = 0, ri = 0; k < head_size/2; k += REG_K/2, ri++) {
            Transpose2DMatrix(QmatI32.select<q_step, 1, REG_K/2, 1>(0, k), rQ[ri].format<uint, REG_K/2, q_step>());
            rQ[ri].format<half>() = cm_mul<half>(rQ[ri].format<half>(), (half)scale_factor);
        }
    }

    constexpr int num_P_tiles = REG_N / REG_M;
    matrix <float, head_size/REG_N*num_P_tiles, REG_M*REG_N> rO;
    int causal_left = q_start;

    constexpr uint slm_buff_size = kv_step * head_size * sizeof(half);
    int slm_buff_id_write = 0;
    int slm_buff_id_read = 0;

    auto load_slm_KV = [&](int kv_pos) {
        //if (kv_pos < 1024000) return;
        int kv_tokens = kv_stop - kv_pos;
        if (kv_tokens <= 0) return;
        uint slm_offset = (slm_buff_id_write & 3) * slm_buff_size;
        slm_buff_id_write ++;

        // non-tail branch is faster
        if (wg_local_id < local_size/2) {
            //if (kv_pos > 1024000) {
            matrix<half, 2*REG_M, REG_K> temp;
            for(int k = REG_K * wg_local_id; k < head_size; k += REG_K*(local_size/2)) {
                cm_load_2d(temp, key, kv_off + k*sizeof(half), kv_pitch);
                cm_slm_block_write(slm_K, 
                                    slm_offset + k * 2 * REG_M * sizeof(half),
                                    temp.format<half>());
            }
        } else {
            //if (kv_pos > 1024000) {
            // read 16x16 XMX-B matrix (1x REG_N in Xe2, 2x REG_N in Xe1)
            constexpr int VK_STEP = 16;
            static_assert((VK_STEP % REG_N) == 0);
            matrix<half, REG_K, VK_STEP> temp2;
            matrix<half, REG_K/2, REG_N*2> temp_vnni;
            //b2dV.set_block_y(kv_pos);

            static_assert((head_size % VK_STEP) == 0);
            #pragma unroll
            for(int k = VK_STEP * (wg_local_id-local_size/2); k < head_size; k += VK_STEP * (local_size/2)) {
                cm_load_2d(temp2, value, kv_off + k*sizeof(half), kv_pitch);

                #pragma unroll
                for(int p = 0; p < VK_STEP/REG_N; p++) {
                    temp_vnni.select<REG_K/2, 1, REG_N, 2>(0, 0) = temp2.select<REG_K/2, 2, REG_N, 1>(0, p*REG_N);
                    temp_vnni.select<REG_K/2, 1, REG_N, 2>(0, 1) = temp2.select<REG_K/2, 2, REG_N, 1>(1, p*REG_N);
                    // show(temp_vnni);
                    cm_slm_block_write(slm_V, slm_offset + (k + p*REG_N) * REG_K * sizeof(half), temp_vnni.format<half>());
                }
            }
        }
        kv_off += kv_step * kv_pitch;
        // printf(" diff= %lu\n", get_clock() - clk0);
    };

    load_slm_KV(0);
    load_slm_KV(kv_step);

    cm_slm_fence(CM_LOCAL_BARRIER);
    cm_sbarrier(1);

    for(int kv_pos = 0; kv_pos < kv_stop; kv_pos += kv_step,
            slm_buff_id_read ++) {
        //
        //  load0->0, signal1, 
        //  [load1->1, wait2, signal2, read0]
        //  [load2->2, wait3, signal3, read1]
        //  [load3->3, wait4, signal4, read2]  
        //  [load4->0, wait5, signal5, read3]  
        //
        //  after wait4, all workers have reached signal3, so:
        //     - all workers have finished load2 & read0. 
        //     - we can start to load 4 into SLM slot 0 (i & 3) safely 
        //     - we can start to read 2 ((i-2) & 3) safely
        //
        cm_fence(CM_LOCAL_BARRIER);
        cm_sbarrier(0);

        load_slm_KV(kv_pos + 2*kv_step);

        if (kv_pos + kv_step < kv_stop)
            cm_sbarrier(1);

        //if (kv_pos < 1024000) continue;
        uint slm_offset = (slm_buff_id_read & 3) * slm_buff_size;

        //=========================================================== 1807 ~ 3247
        //# St = k @ Qt
        matrix<float, kv_step, q_step> St = ugemm_KQ(slm_K, rQ, slm_offset);

        if constexpr (use_causal_mask) {
            if (causal_left < kv_step) {
                vector<float, q_step> cmask = 0.0f;
                int p = causal_left + 1;
                int v = 0;
                for(; p < 0; p++) {
                    cmask[v] = -3.4e38f;
                    if (v < q_step - 1) v++;
                }
                for(; p < kv_step; p++) {
                    cmask[v] = -3.4e38f;
                    St[p] = cm_add<float>(St[p], cmask);
                    if (v < q_step - 1) v++;
                }
                //if (wg_local_id == 0) show(St);return;
            }
            causal_left -= kv_step;
        }

        // mask off k-tails
        int kv_tokens = kv_stop - kv_pos;
        for(int p = kv_tokens; p < kv_step; p++) St[p] = -3.4e38f;

        //show(St);
        auto max_comp = online_softmax_update(St, cur_max, cur_sum);

        matrix<half, REG_N, REG_K> P;
        Transpose2DMatrix(St, P);

        if (kv_pos == 0)
            ugemm_PV0(slm_V, P, rO, slm_offset);
        else
            ugemm_PV1(slm_V, P, max_comp, rO, slm_offset);
    }

    //# save cur_O/cur_sum.transpose(0, 1)
    matrix<half, num_P_tiles*REG_M, REG_N> cur_O_f16;
    cur_sum = cm_inv(cur_sum);

    #pragma unroll
    for(int k = 0, ri=0; k < head_size; k += REG_N, ri += num_P_tiles) {
        #pragma unroll
        for(int p = 0; p < num_P_tiles; p++) {
            auto cO = rO[ri + p].format<float, REG_M, REG_N>();
            #pragma unroll
            for(int r = 0; r < cO.n_rows(); r++) {
                cur_O_f16[r + p*REG_M] = cm_mul<float>(cO[r], cur_sum[r + p*REG_M]);
            }
        }
        // if (i == args_verbose) show(cur_O_f16);
        cm_store_2d(cur_O_f16, output, qo_off + k*sizeof(half), qo_pitch);
    }
}


template<bool use_causal_mask, int local_size>
void sdpa_kernel_lsc(
    uint slm_K,
    uint slm_V,
    int wg_local_id,
    int q_start,
    int kv_stop,
    int head_base_id,
    int q_len,
    int kv_len,
    svmptr_t q_base [[type("svmptr_t")]],
    svmptr_t k_base [[type("svmptr_t")]],
    svmptr_t v_base [[type("svmptr_t")]],
    svmptr_t o_base [[type("svmptr_t")]]) {

    uint qo_pitch = num_heads * head_size * sizeof(half);
    uint kv_pitch = num_kv_heads * head_size * sizeof(half);
    vector<float, q_step> cur_max;
    vector<float, q_step> cur_sum;

    cur_max = -3e38f;
    cur_sum = 0;
    constexpr int num_P_tiles = REG_N / REG_M;
    matrix<half, head_size/REG_K, REG_K*REG_N> rQ;
    matrix <float, head_size/REG_N*num_P_tiles, REG_M*REG_N> rO;

    auto q_tokens_left = q_len - q_start;
    static_assert(q_step == REG_N);
    static_assert(kv_step == REG_K);

    if (q_tokens_left < 0) q_tokens_left = 0;
    if (q_tokens_left > q_step) q_tokens_left = q_step;

    if (q_tokens_left > 0) {
        lsc::block_2d_desc<uint, 1, REG_N, REG_K/2> b2dQ(reinterpret_cast<uint*>(q_base), q_tokens_left - 1, head_size*sizeof(half) - 1, qo_pitch - 1, 0, 0);
        #pragma unroll
        for(int k = 0, ri = 0; k < head_size/2; k += REG_K/2, ri++) {
            cm_load<lsc::Transpose>(rQ[ri].format<uint>(), b2dQ.set_block_x(k));
            rQ[ri].format<half>() = cm_mul<half>(rQ[ri].format<half>(), (half)scale_factor);
        }
    }

    lsc::block_2d_desc<half, 1, kv_step, REG_K> b2dK(k_base, kv_stop - 1, head_size*sizeof(half) - 1, kv_pitch - 1, 0, 0);
    lsc::block_2d_desc<half, 1, REG_K, REG_N> b2dV(v_base, kv_stop - 1, head_size*sizeof(half) - 1, kv_pitch - 1, 0, 0);

    int causal_left = q_start;

    constexpr uint slm_buff_size = kv_step * head_size * sizeof(half);
    int slm_buff_id_write = 0;
    int slm_buff_id_read = 0;

    auto load_slm_KV = [&](int kv_pos) {
        if (kv_pos < kv_stop) {
            uint slm_offset = (slm_buff_id_write & 3) * slm_buff_size;
            slm_buff_id_write ++;
            if (wg_local_id < local_size/2) {
                vector<half, kv_step * REG_K> temp0;
                b2dK.set_block_y(kv_pos);
                for(int k = REG_K*wg_local_id; k < head_size; k += REG_K*(local_size/2)) {
                    cm_load<lsc::Normal>(temp0, b2dK.set_block_x(k));
                    cm_slm_block_write(slm_K, slm_offset + k * kv_step * sizeof(half), temp0);
                }
            } else {
                vector<half, REG_K*REG_N> temp2;
                b2dV.set_block_y(kv_pos);
                #pragma unroll
                for(int k = REG_N*(wg_local_id-(local_size/2)); k < head_size; k += REG_N*(local_size/2)) {
                    cm_load<lsc::VNNI>(temp2, b2dV.set_block_x(k));
                    cm_slm_block_write(slm_V, slm_offset + k * REG_K * sizeof(half), temp2);
                }
            }
        }
    };
    load_slm_KV(0);
    load_slm_KV(kv_step);
    cm_slm_fence(CM_LOCAL_BARRIER);
    cm_sbarrier(1);

    for(int kv_pos = 0; kv_pos < kv_stop; kv_pos += kv_step,
            k_base += kv_step * kv_pitch,
            v_base += kv_step * kv_pitch,
            slm_buff_id_read ++) {

        //  load0, load1, signal1, 
        //  [wait2, signal2, load2, read0]
        //  [wait3, signal3, load3, read1]
        //  [wait4, signal4, load4, read2]  
        //  [wait5, signal5, load5, read3]  
        //
        //  after wait4, all workers have reached signal3, so:
        //     - all workers have finished load2 & read0. 
        //     - we can start to load 4 into SLM slot 0 (i & 3) safely 
        //     - we can start to read 2 ((i-2) & 3) safely

        cm_fence(CM_LOCAL_BARRIER);
        cm_sbarrier(0);
        //if (kv_pos > 1024000)
        if (kv_pos + kv_step < kv_stop)
            cm_sbarrier(1);

        load_slm_KV(kv_pos + kv_step*2);

        {
            uint slm_offset = (slm_buff_id_read & 3) * slm_buff_size;
            //# St = k @ Qt
            matrix<float, kv_step, q_step> St = ugemm_KQ(slm_K, rQ, slm_offset);

            if constexpr (use_causal_mask) {
                if (causal_left < kv_step) {
                    vector<float, q_step> cmask = 0.0f;
                    int p = causal_left + 1;
                    int v = 0;
                    for(; p < 0; p++) {
                        cmask[v] = -3.4e38f;
                        if (v < q_step - 1) v++;
                    }
                    for(; p < kv_step; p++) {
                        cmask[v] = -3.4e38f;
                        St[p] = cm_add<float>(St[p], cmask);
                        if (v < q_step - 1) v++;
                    }
                    //if (wg_local_id == 0) show(St);return;
                }
                causal_left -= kv_step;
            }

            int kv_tokens = kv_stop - kv_pos;
            // LSC ensures no overflow-access, but mask off k-tails attn-score is still required
            for(int p = kv_tokens; p < kv_step; p++) St[p] = -3.4e38f;

            //show(St);
            auto max_comp = online_softmax_update(St, cur_max, cur_sum);

            matrix<half, REG_N, REG_K> P;
            Transpose2DMatrix(St, P);

            if (kv_pos == 0)
                ugemm_PV0(slm_V, P, rO, slm_offset);
            else
                ugemm_PV1(slm_V, P, max_comp, rO, slm_offset);
        }
    }
    // cm_sbarrier(0);
    if (q_tokens_left == 0) return;

    //# save cur_O/cur_sum.transpose(0, 1)
    matrix<half, num_P_tiles*REG_M, REG_N> cur_O_f16;
    cur_sum = cm_inv(cur_sum);

    lsc::block_2d_desc<half, 1, REG_M, REG_N> b2dO(o_base, q_tokens_left - 1, head_size*sizeof(half) - 1, qo_pitch - 1, 0, 0);

    #pragma unroll
    for(int k = 0, ri=0; k < head_size; k += REG_N, ri += num_P_tiles) {
        #pragma unroll
        for(int p = 0; p < num_P_tiles; p++) {
            auto cO = rO[ri + p].format<float, REG_M, REG_N>();
            #pragma unroll
            for(int r = 0; r < cO.n_rows(); r++) {
                cur_O_f16[r + p*REG_M] = cm_mul<float>(cO[r], cur_sum[r + p*REG_M]);
            }
        }
        b2dO.set_block_x(k);
        cm_store(b2dO.set_block_y(0), cur_O_f16.format<half, num_P_tiles, REG_M * REG_N>().row(0));
        cm_store(b2dO.set_block_y(REG_M), cur_O_f16.format<half, num_P_tiles, REG_M * REG_N>().row(1));
    }
}



#ifdef CM_HAS_LSC_UNTYPED_2D
extern "C" _GENX_MAIN_ void cm_sdpa(
    int head_base_id,
    int q_len,
    int kv_len,
    half* query [[type("svmptr_t")]],
    half* key [[type("svmptr_t")]],
    half* value [[type("svmptr_t")]],
    half* output [[type("svmptr_t")]],
    __global uint64_t* cminfo [[type("svmptr_t")]]
    ) {
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
    auto h = head_base_id + cm_group_id(1);
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

    sdpa_kernel_lsc<causal_mask, WG_SIZE_HINT>(
                                slm_K,
                                slm_V,
                                wg_local_id,
                                q_start,
                                kv_stop,
                                head_base_id,
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
    int head_base_id,
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

    constexpr uint K_SLM_SIZE = (4*kv_step * head_size * sizeof(half));
    constexpr uint V_SLM_SIZE = (4*kv_step * head_size * sizeof(half));
    constexpr uint Q_SLM_SIZE = 0;//(q_step * head_size * sizeof(half)) * local_size;

    cm_slm_init(K_SLM_SIZE + V_SLM_SIZE + Q_SLM_SIZE);

    auto slm_K = cm_slm_alloc(K_SLM_SIZE);
    auto slm_V = cm_slm_alloc(V_SLM_SIZE);

    auto batch = cm_group_id(0);
    auto h = head_base_id + cm_group_id(1);
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

    sdpa_kernel<causal_mask, WG_SIZE_HINT>(
                                slm_K,
                                slm_V,
                                wg_local_id,
                                q_start,
                                kv_stop,
                                head_base_id,
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
