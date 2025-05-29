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

    matrix<uint, REG_N, REG_K/2> QmatI32;
    #pragma unroll
    for(int k = 0, ri = 0; k < head_size; k += REG_K, ri++, q_base += REG_K * sizeof(half)) {
        //# DWORD transposed load == (transposed + VNNI) load
        svm_read_2d(QmatI32, q_base, qo_pitch, q_tokens_left);
        Transpose2DMatrix(QmatI32, rQ[ri].format<uint, REG_K/2, REG_N>());
        rQ[ri].format<half>() = cm_mul<half>(rQ[ri].format<half>(), (half)scale_factor);
        //show(rQ[ri].format<half, REG_K/2, REG_N*2>());
    }

    //int kv_stop = (cm_group_id(2) + 1) * local_size * q_step;

    constexpr int num_P_tiles = REG_N / REG_M;
    matrix <float, head_size/REG_N*num_P_tiles, REG_M*REG_N> rO;
    int causal_left = q_start;


    #pragma unroll (1)
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
                    matrix<half, 2*REG_M, REG_K> temp;
                    #pragma unroll
                    for(int k = REG_K*wg_local_id; k < head_size; k += REG_K*(local_size/2)) {
                        svm_read_2d(temp, k_base + k*sizeof(half), kv_pitch);
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

            cm_barrier();
        }

        //=========================================================== 1807 ~ 3247
        //# St = k @ Qt
        matrix<float, kv_step, q_step> St;
        auto St2 = St.format<float, 2, REG_M*REG_N>();
        #pragma unroll
        for(int k = 0, ri = 0; k < head_size; k += REG_K, ri++) {
            matrix<half, 2, REG_M * REG_K> Kmat;
            cm_slm_block_read(slm_K, GENX_NONE, ri * Kmat.n_elems() * sizeof(half), Kmat.format<half>());
            //show(Kmat.format<half, 2*REG_M, REG_K>());
            if (k == 0) {
                St2[0] = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount, float>(0, rQ[ri].format<int32_t>(), Kmat[0].format<int32_t>());
                St2[1] = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount, float>(0, rQ[ri].format<int32_t>(), Kmat[1].format<int32_t>());
            } else {
                St2[0] = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount>(St2[0], rQ[ri].format<int32_t>(), Kmat[0].format<int32_t>());
                St2[1] = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount>(St2[1], rQ[ri].format<int32_t>(), Kmat[1].format<int32_t>());
            }
        }
        //show(St); return;

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

        matrix<half, REG_N, REG_K> P;
        Transpose2DMatrix(St, P);

        //show(cur_sum.format<float, 1, REG_N>()); return;
        //============================================================== 1074

        //============================================================== 666
        //if (kv_pos == 16) {show(P);return;}
        //auto clk0 = get_clock();
        auto P2 = P.format<half, num_P_tiles, REG_M * REG_K>();
        
        if (kv_pos == 0) {
            #pragma unroll
            for(int k = 0, ri = 0; k < head_size; k += REG_N, ri += num_P_tiles) {
                matrix<half, REG_K/2, REG_N*2> Vmat;
                // V has been VNNI-prepacked
                cm_slm_block_read(slm_V, GENX_NONE, REG_K*k*sizeof(half), Vmat.format<half>());
                
                //show(Vmat);

                #pragma unroll
                for(int p = 0; p < num_P_tiles; p++) {
                    rO[ri + p] = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount, float>(
                                    0,
                                    Vmat.format<int32_t>(),
                                    P2[p].format<int32_t>());
                    //show(rO[ri + p].format<float, REG_M, REG_N>());
                }
            }
        } else {
            #pragma unroll
            for(int k = 0, ri=0; k < head_size; k += REG_N, ri += num_P_tiles) {
                matrix<half, REG_K/2, REG_N*2> Vmat;
                // V has been VNNI-prepacked
                cm_slm_block_read(slm_V, GENX_NONE, REG_K*k*sizeof(half), Vmat.format<half>());

                //show(Vmat);
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
                                P2[p].format<int32_t>());
                    //if (kv_pos == args_verbose) show(rO[ri + p].format<float, REG_M, REG_N>());
                }
                // if (kv_pos == args_verbose) show(cur_O.format<float, 2*REG_M, REG_N>());
            }
            //if (kv_pos == args_verbose) {
            //    printf(">>>>> %d\n", 12345);
            //    return;
            //}
        }
        cur_max = new_max_t;
    }//# for(int kv_pos = 0; kv_pos < kv_len; kv_pos += kv_step) {

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

    constexpr uint K_SLM_SIZE = (kv_step * head_size * sizeof(half));
    constexpr uint V_SLM_SIZE = (kv_step * head_size * sizeof(half));
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
    auto q_offset = wg_local_id * q_step * head_size * sizeof(half);
    auto o_offset = wg_local_id * q_step * head_size * sizeof(float);

    //# debugging stage
    auto q_base = reinterpret_cast<svmptr_t>(query + ((batch*q_len + q_start)*num_heads + h)*head_size);
    auto k_base = reinterpret_cast<svmptr_t>(key + (batch*num_kv_heads*kv_len + hkv)*head_size);
    auto o_base = reinterpret_cast<svmptr_t>(output + ((batch * q_len + q_start)*num_heads + h)*head_size);
    auto v_base = reinterpret_cast<svmptr_t>(value + (batch*num_kv_heads*kv_len + hkv)*head_size);

    //# b2dQ reinterpret as 32bit(DWORD) for transposed load(combined with VNNI)

    //lsc::block_2d_desc<uint, 1, REG_N, REG_K/2> b2dQ(reinterpret_cast<uint*>(query + (batch*num_heads*q_len + h)*head_size), q_len - 1, head_size*sizeof(half) - 1, qo_pitch - 1, 0, 0);
    //lsc::block_2d_desc<half, 1, REG_M, REG_K> b2dK(key + (batch*num_kv_heads*kv_len + hkv)*head_size,   kv_len - 1, head_size*sizeof(half) - 1, kv_pitch - 1, 0, 0);
    //lsc::block_2d_desc<half, 2, REG_K, REG_N> b2dV(value + (batch*num_kv_heads*kv_len + hkv)*head_size, kv_len - 1, head_size*sizeof(half) - 1, kv_pitch - 1, 0, 0);
    //lsc::block_2d_desc<half, 1, REG_M, REG_N> b2dO(output + (batch*num_heads*q_len + h)*head_size,   q_len - 1, head_size*sizeof(half) - 1, qo_pitch - 1, 0, 0);
    int kv_stop = kv_len;
    if constexpr (causal_mask) {
        kv_stop = (q_group_id + 1) * local_size * q_step;
        if (kv_stop > kv_len) kv_stop = kv_len;
    }
    //# load Qt into register & pack as VNNI & store to SLM (as dpas-B tile)
    sdpa_kernel<causal_mask, WG_SIZE_HINT>(   slm_K,
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
        sdpa_kernel<causal_mask, WG_SIZE_HINT>(   slm_K,
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