#if 0
    #ifdef CM_HAS_LSC_UNTYPED_2D
    #include "cm_sdpa_xe2.hpp"
    #else
    #include "cm_sdpa_xe1.hpp"
    #endif
#else


/*
    Xe1&2 unified version
*/


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

    matrix<half, head_size/REG_K, REG_K*REG_N> rQ;
    auto q_tokens_left = q_len - q_start;
    static_assert(q_step == REG_N);
    static_assert(kv_step == REG_K);

    if (q_tokens_left < 0) q_tokens_left = 0;
    if (q_tokens_left > q_step) q_tokens_left = q_step;

    if (q_tokens_left > 0) {
#if 1
    #ifdef CM_HAS_LSC_UNTYPED_2D && 0
        // LSC can only transpose 1 block at a time, didn't show better performance
        lsc::block_2d_desc<uint, 1, REG_N, REG_K/2> b2dQ(reinterpret_cast<uint*>(q_base), q_tokens_left - 1, head_size*sizeof(half) - 1, qo_pitch - 1, 0, 0);
        #pragma unroll
        for(int k = 0, ri = 0; k < head_size/2; k += REG_K/2, ri++) {
            cm_load<lsc::Transpose>(rQ[ri].format<uint>(), b2dQ.set_block_x(k));
            rQ[ri].format<half>() = cm_mul<half>(rQ[ri].format<half>(), (half)scale_factor);
        }
    #else
        // load as many as possible given one address
        matrix<uint, q_step, head_size/2> QmatI32;
        if (q_tokens_left == q_step)
            svm_read_2d(QmatI32, q_base, qo_pitch);
        else
            svm_read_2d(QmatI32, q_base, qo_pitch, q_tokens_left);
        #pragma unroll
        for(int k = 0, ri = 0; k < head_size/2; k += REG_K/2, ri++) {
            Transpose2DMatrix(QmatI32.select<q_step, 1, REG_K/2, 1>(0, k), rQ[ri].format<uint, REG_K/2, q_step>());
            rQ[ri].format<half>() = cm_mul<half>(rQ[ri].format<half>(), (half)scale_factor);
        }
    #endif
#else
        matrix<uint, REG_N, REG_K/2> QmatI32;
        #pragma unroll
        for(int k = 0, ri = 0; k < head_size; k += REG_K, ri++, q_base += REG_K * sizeof(half)) {
            //# DWORD transposed load == (transposed + VNNI) load
            svm_read_2d(QmatI32, q_base, qo_pitch, q_tokens_left);
            Transpose2DMatrix(QmatI32, rQ[ri].format<uint, REG_K/2, REG_N>());
            rQ[ri].format<half>() = cm_mul<half>(rQ[ri].format<half>(), (half)scale_factor);
            //show(rQ[ri].format<half, REG_K/2, REG_N*2>());
        }
#endif
    }

    constexpr int num_P_tiles = REG_N / REG_M;
    matrix <float, head_size/REG_N*num_P_tiles, REG_M*REG_N> rO;
    int causal_left = q_start;

    for(int kv_pos = 0; kv_pos < kv_stop; kv_pos += kv_step,
            k_base += kv_step * kv_pitch,
            v_base += kv_step * kv_pitch) {
        //# load K into SLM as dpas-A tile (shared by all hw within same WG)
        //# load V into SLM as dpas-B tile (shared by all hw within same WG)
        {
            int kv_tokens = kv_stop - kv_pos;
            if (kv_tokens < kv_step) {
                // handle tails in kv-len
                vector<uint, 2*REG_M> kv_offsets = 0;
                #pragma unroll
                for(int i = 0; i < kv_step; i++) {
                    kv_offsets[i] = (i < kv_tokens) * i * kv_pitch;
                }

                if (kv_pos > 0) cm_barrier();
                if (wg_local_id < local_size/2) {
                    matrix<half, 2*REG_M, REG_K> temp;
                    #pragma unroll
                    for(int k = REG_K*wg_local_id; k < head_size; k += REG_K*(local_size/2)) {
                        svm_read_2d(temp, k_base + k*sizeof(half), kv_offsets);
                        cm_slm_block_write(slm_K, k * 2 * REG_M * sizeof(half), temp.format<half>());
                    }
                } else {
                    // read 16x16 XMX-B matrix (1x REG_N in Xe2, 2x REG_N in Xe1)
                    constexpr int VK_STEP = 16;
                    static_assert((VK_STEP % REG_N) == 0);
                    matrix<half, REG_K, VK_STEP> temp2;
                    matrix<half, REG_K/2, REG_N*2> temp_vnni;
                    //b2dV.set_block_y(kv_pos);

                    static_assert((head_size % VK_STEP) == 0);
                    #pragma unroll
                    for(int k = VK_STEP * (wg_local_id-local_size/2); k < head_size; k += VK_STEP * (local_size/2)) {
                        svm_read_2d(temp2, v_base + k*sizeof(half), kv_offsets);

                        #pragma unroll
                        for(int p = 0; p < VK_STEP/REG_N; p++) {
                            temp_vnni.select<REG_K/2, 1, REG_N, 2>(0, 0) = temp2.select<REG_K/2, 2, REG_N, 1>(0, p*REG_N);
                            temp_vnni.select<REG_K/2, 1, REG_N, 2>(0, 1) = temp2.select<REG_K/2, 2, REG_N, 1>(1, p*REG_N);
                            // show(temp_vnni);
                            cm_slm_block_write(slm_V, (k + p*REG_N) * REG_K * sizeof(half), temp_vnni.format<half>());
                        }
                    }
                }
            } else {
                // non-tail branch is faster
                if (kv_pos > 0) cm_barrier();
                if (wg_local_id < local_size/2) {
                    matrix<half, 2, head_size> temp;
                    svm_read_2d(temp, k_base + (wg_local_id * 2) * kv_pitch, kv_pitch);
                    #pragma unroll
                    for(int k = 0; k < head_size; k += REG_K) {
                        auto off = k * (REG_K*2*REG_M* sizeof(half)/REG_K);
                        cm_slm_block_write(slm_K, off + (wg_local_id*2 + 0)*REG_K*sizeof(half) , temp[0].select<REG_K, 1>(k));
                        cm_slm_block_write(slm_K, off + (wg_local_id*2 + 1)*REG_K*sizeof(half) , temp[1].select<REG_K, 1>(k));
                    }
                } else {
                    // read 16x16 XMX-B matrix (1x REG_N in Xe2, 2x REG_N in Xe1)
                    constexpr int VK_STEP = 16;
                    static_assert((VK_STEP % REG_N) == 0);
                    matrix<half, REG_K, VK_STEP> temp2;
                    matrix<half, REG_K/2, REG_N*2> temp_vnni;
                    //b2dV.set_block_y(kv_pos);

                    static_assert((head_size % VK_STEP) == 0);
                    #pragma unroll
                    for(int k = VK_STEP * (wg_local_id-local_size/2); k < head_size; k += VK_STEP * (local_size/2)) {
                        svm_read_2d(temp2, v_base + k*sizeof(half), kv_pitch);

                        #pragma unroll
                        for(int p = 0; p < VK_STEP/REG_N; p++) {
                            temp_vnni.select<REG_K/2, 1, REG_N, 2>(0, 0) = temp2.select<REG_K/2, 2, REG_N, 1>(0, p*REG_N);
                            temp_vnni.select<REG_K/2, 1, REG_N, 2>(0, 1) = temp2.select<REG_K/2, 2, REG_N, 1>(1, p*REG_N);
                            // show(temp_vnni);
                            cm_slm_block_write(slm_V, (k + p*REG_N) * REG_K * sizeof(half), temp_vnni.format<half>());
                        }
                    }
                }
            }
            // printf(" diff= %lu\n", get_clock() - clk0);
            cm_slm_fence(CM_GLOBAL_COHERENT_FENCE);
            cm_barrier();
        }

        //=========================================================== 1807 ~ 3247
        //# St = k @ Qt
        matrix<float, kv_step, q_step> St = ugemm_KQ(slm_K, rQ);

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
#if 0
        {
            // how to handle the tails in attn-mask?
            matrix<half, REG_N, REG_K> Maskmat;
            // cm_load_normal(Maskmat.format<half>(), b2dMask.set_block_x(kv_pos).set_block_y(q_start));
            //svm_read_2d(Maskmat, mask_base + kv_pos * sizeof(half), kv_len*sizeof(half), q_tokens_left);
            svm_read_2d(Maskmat, mask_base + kv_pos * sizeof(half), kv_len*sizeof(half));

            matrix<float, REG_K, REG_N> MaskT;
            Transpose2DMatrix(Maskmat, MaskT);
            // show(MaskT); return;
            St = cm_mul<float>(St, (float)scale_factor);  // convert scale_factor into (float), or it will be promoted to double
            St = cm_add<float>(St, MaskT);
        }
#endif
        int kv_tokens = kv_stop - kv_pos;
        if (kv_tokens < kv_step) {
            // mask off k-tails
            for(int p = kv_tokens; p < kv_step; p++) {
                St[p] = -3.4e38f;
            }
        }

        //show(St);

        vector<float, REG_N> new_max_t;
        new_max_t = cm_max<float>(St[0], St[1]);
        for(int r = 2; r < St.n_rows(); r++) new_max_t = cm_max<float>(new_max_t, St[r]);
        new_max_t = cm_max<float>(new_max_t, cur_max);

        // Pt = torch.exp(St - new_max)
        constexpr float log2e = 1.4426950408889634f;
        for(int r = 0; r < St.n_rows(); r++) St[r] = cm_exp((St[r] - new_max_t)*log2e);

        vector<float, REG_N> row_sum_t;
        row_sum_t = cm_add<float>(St[0], St[1]);
        for(int r = 2; r < St.n_rows(); r++) row_sum_t = cm_add<float>(row_sum_t, St[r]);

        vector<float, REG_N> max_comp;
        max_comp = cm_exp((cur_max - new_max_t)*log2e);
        cur_sum = cm_mul<float>(cur_sum, max_comp);
        cur_sum = cm_add<float>(cur_sum, row_sum_t);

        matrix<half, REG_N, REG_K> P;
        Transpose2DMatrix(St, P);

        if (kv_pos == 0)
            ugemm_PV0(slm_V, P, rO);
        else
            ugemm_PV1(slm_V, P, max_comp, rO);

        cur_max = new_max_t;
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
        svm_write_2d(cur_O_f16, o_base + k*sizeof(half), qo_pitch, q_tokens_left);
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
        //# load K into SLM as dpas-A tile (shared by all hw within same WG)
        //# load V into SLM as dpas-B tile (shared by all hw within same WG)
        //if (kv_pos > 1024000)

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

            vector<float, REG_N> new_max_t;
            new_max_t = cm_max<float>(St[0], St[1]);
            for(int r = 2; r < St.n_rows(); r++) new_max_t = cm_max<float>(new_max_t, St[r]);
            new_max_t = cm_max<float>(new_max_t, cur_max);

            // Pt = torch.exp(St - new_max)
            constexpr float log2e = 1.4426950408889634f;
            for(int r = 0; r < St.n_rows(); r++) St[r] = cm_exp((St[r] - new_max_t)*log2e);

            vector<float, REG_N> row_sum_t;
            row_sum_t = cm_add<float>(St[0], St[1]);
            for(int r = 2; r < St.n_rows(); r++) row_sum_t = cm_add<float>(row_sum_t, St[r]);

            vector<float, REG_N> max_comp;
            max_comp = cm_exp((cur_max - new_max_t)*log2e);
            cur_sum = cm_mul<float>(cur_sum, max_comp);
            cur_sum = cm_add<float>(cur_sum, row_sum_t);

            matrix<half, REG_N, REG_K> P;
            Transpose2DMatrix(St, P);

            if (kv_pos == 0)
                ugemm_PV0(slm_V, P, rO, slm_offset);
            else
                ugemm_PV1(slm_V, P, max_comp, rO, slm_offset);
            cur_max = new_max_t;
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
#define SDPA_KERNEL sdpa_kernel_lsc
#else
#define SDPA_KERNEL sdpa_kernel
#endif

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

    SDPA_KERNEL<causal_mask, WG_SIZE_HINT>(
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


#if 0
    {
        auto q_start = (q_len - q_group_id * local_size * q_step) - (local_size * q_step) + wg_local_id*q_step;
        auto q_base = reinterpret_cast<svmptr_t>(query + ((batch*q_len + q_start)*num_heads + h)*head_size);
        auto k_base = reinterpret_cast<svmptr_t>(key + (batch*num_kv_heads*kv_len + hkv)*head_size);
        auto o_base = reinterpret_cast<svmptr_t>(output + ((batch * q_len + q_start)*num_heads + h)*head_size);
        auto v_base = reinterpret_cast<svmptr_t>(value + (batch*num_kv_heads*kv_len + hkv)*head_size);
        if constexpr (causal_mask) {
            kv_stop = (q_len - q_group_id * local_size * q_step);
            if (kv_stop > kv_len) kv_stop = kv_len;
        }
        SDPA_KERNEL<causal_mask, WG_SIZE_HINT>(   slm_K,
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
    }
#endif
    // if (i == args_verbose) return;
    CMTracer_end(&cminfo);
}

#endif