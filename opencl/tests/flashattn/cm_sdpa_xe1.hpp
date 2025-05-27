#include "cm_sdpa_common.hpp"

static_assert(REG_N == 8);

template<bool USE_CAUSAL_MASK>
void row_kernel(
    uint slm_K,
    uint slm_V,
    int local_id,
    int q_start,
    int kv_stop,
    int head_base_id,
    int kv_len,
    svmptr_t q_base [[type("svmptr_t")]],
    svmptr_t k_base [[type("svmptr_t")]],
    svmptr_t v_base [[type("svmptr_t")]],
    svmptr_t o_base [[type("svmptr_t")]],
    svmptr_t mask_base [[type("svmptr_t")]]) {

    uint qo_pitch = num_heads * head_size * sizeof(half);
    uint kv_pitch = num_kv_heads * head_size * sizeof(half);
    vector<float, q_step> cur_max;
    vector<float, q_step> cur_sum;

    const int local_size = cm_linear_local_size();

    cur_max = -1e9;
    cur_sum = 0;

    //# load Qt into register & pack as VNNI & store to SLM (as dpas-B tile)
    matrix<half, head_size/REG_K, REG_K*REG_N> rQ;
    {
        matrix<uint, REG_N, REG_K/2> Qmat;

        #pragma unroll
        for(int k = 0, ri = 0; k < head_size/2; k += REG_K/2, ri++, q_base += 16*sizeof(half)) {
            //# DWORD transposed load == (transposed + VNNI) load
            svm_read_2d(Qmat, q_base, qo_pitch);
            Transpose_8x8(Qmat, rQ[ri].format<uint, REG_K/2, REG_N>());
            rQ[ri].format<half>() = cm_mul<half>(rQ[ri].format<half>(), (half)scale_factor);

            //b2dQ.set_block_x(k);
            //cm_load<lsc::Transpose>(rQ[ri].format<uint>(), b2dQ.set_block_y(q_start));

            // show(rQ[ri].format<half, REG_K/2, REG_N*2>()); //# vnni
        }
    }

    matrix <float, head_size/REG_K*2, REG_M*REG_N> rO;

    int causal_left = q_start;
    for(int kv_pos = 0; kv_pos < kv_stop; kv_pos += kv_step,
            k_base += kv_step * kv_pitch,
            v_base += kv_step * kv_pitch) {
        {
            if (kv_pos > 0) cm_barrier();

            if (local_size == 1) {
                matrix<half, 2*REG_M, REG_K> temp;
                for(int k = 0; k < head_size; k += REG_K) {
                    //b2dK.set_block_x(k);
                    //cm_load<lsc::Normal>(temp0.format<half>(), b2dK.set_block_y(kv_pos));
                    //cm_prefetch(b2dK.set_block_y(kv_pos + kv_step));

                    svm_read_2d(temp, k_base + k*sizeof(half), kv_pitch);

                    // show(temp);

                    uint offset = k * 2 * REG_M * sizeof(half);
                    cm_slm_block_write(slm_K, offset, temp.format<half>());
                }

                matrix<half, REG_K, REG_N> temp2;
                matrix<half, REG_K/2, REG_N*2> temp_vnni;
                //b2dV.set_block_y(kv_pos);
                for(int k = 0; k < head_size; k += REG_N) {
                    //cm_load<lsc::VNNI>(temp2.format<half>(), b2dV.set_block_x(k).set_block_y(kv_pos));
                    //cm_prefetch(b2dV.set_block_y(kv_pos + kv_step));

                    svm_read_2d(temp2, v_base + k*sizeof(half), kv_pitch);

                    temp_vnni.select<REG_K/2, 1, REG_N, 2>(0, 0) = temp2.select<REG_K/2, 2, REG_N, 1>(0, 0);
                    temp_vnni.select<REG_K/2, 1, REG_N, 2>(0, 1) = temp2.select<REG_K/2, 2, REG_N, 1>(1, 0);

                    // show(temp_vnni);
                    cm_slm_block_write(slm_V, k * REG_K * sizeof(half), temp_vnni.format<half>());
                }
            } else {
                if (local_id < local_size/2) {
                    matrix<half, 2*REG_M, REG_K> temp;
                    for(int k = REG_K * local_id; k < head_size; k += REG_K*(local_size/2)) {
                        //b2dK.set_block_x(k);
                        //cm_load<lsc::Normal>(temp.format<half>(), b2dK.set_block_y(kv_pos));
                        //cm_prefetch(b2dK.set_block_y(kv_pos + kv_step));

                        svm_read_2d(temp, k_base + k*sizeof(half), kv_pitch);
                        // show(temp);
                        cm_slm_block_write(slm_K, k * 2 * REG_M * sizeof(half), temp.format<half>());
                    }
                } else {
                    matrix<half, REG_K, 2*REG_N> temp2;
                    matrix<half, REG_K/2, REG_N*2> temp_vnni;
                    matrix<half, REG_K/2, REG_N*2> temp_vnni2;
                    //b2dV.set_block_y(kv_pos);

                    static_assert((head_size % (2*REG_N)) == 0);

                    for(int k = 2*REG_N*(local_id-local_size/2); k < head_size; k += 2*REG_N*(local_size/2)) {
                        //cm_load<lsc::VNNI>(temp2.format<half>(), b2dV.set_block_x(k).set_block_y(kv_pos));
                        //cm_prefetch(b2dV.set_block_y(kv_pos + kv_step));

                        svm_read_2d(temp2, v_base + k*sizeof(half), kv_pitch);

                        temp_vnni.select<REG_K/2, 1, REG_N, 2>(0, 0) = temp2.select<REG_K/2, 2, REG_N, 1>(0, 0);
                        temp_vnni.select<REG_K/2, 1, REG_N, 2>(0, 1) = temp2.select<REG_K/2, 2, REG_N, 1>(1, 0);
                        // show(temp_vnni);
                        cm_slm_block_write(slm_V, k * REG_K * sizeof(half), temp_vnni.format<half>());

                        temp_vnni2.select<REG_K/2, 1, REG_N, 2>(0, 0) = temp2.select<REG_K/2, 2, REG_N, 1>(0, REG_N);
                        temp_vnni2.select<REG_K/2, 1, REG_N, 2>(0, 1) = temp2.select<REG_K/2, 2, REG_N, 1>(1, REG_N);
                        cm_slm_block_write(slm_V, (k + REG_N) * REG_K * sizeof(half), temp_vnni2.format<half>());
                    }
                }
            }
            cm_barrier();
        }

        //=========================================================== 1807 ~ 3247
        //# St = k @ Qt
        matrix<float, 2*REG_M, REG_N> St;
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
        //show(St);

        if constexpr (USE_CAUSAL_MASK) {
            auto cmask_off = kv_step - causal_left;
            if (cmask_off > 0) {
                //# full-attention: skip loading causal mask
                if (cmask_off > 2*kv_step) cmask_off = 2*kv_step;
                matrix<half, St.n_rows(), REG_N> temp;
                cm_svm_block_read(reinterpret_cast<svmptr_t>(mask_base + cmask_off * REG_N  * sizeof(half)), temp.format<half>());
                St = cm_add<float>(St, temp);
            }

            causal_left -= kv_step;
        } else {
            matrix<half, REG_M, REG_N + REG_N> Maskmat;
            //b2dMask.set_block_x(kv_pos);
            //cm_load<lsc::Normal>(Maskmat[0].format<half>(), b2dMask.set_block_y(q_start));
            //cm_load<lsc::Normal>(Maskmat[1].format<half>(), b2dMask.set_block_y(q_start + REG_M));
            svm_read_2d(Maskmat, mask_base + kv_pos * sizeof(half), kv_len*sizeof(half));

            matrix<float, 2*REG_M, REG_N> MaskT;
            Transpose_8x8(Maskmat.select<REG_M, 1, REG_N, 1>(0,0), MaskT.select<REG_M, 1, REG_N, 1>(0,0));
            Transpose_8x8(Maskmat.select<REG_M, 1, REG_N, 1>(0,REG_N), MaskT.select<REG_M, 1, REG_N, 1>(REG_M,0));

            //show(MaskT);
            //St = cm_mul<float>(St, (float)scale_factor);  // convert scale_factor into (float), or it will be promoted to double
            St = cm_add<float>(St, MaskT);
        }

        // show(St);

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

        //show(cur_sum.format<float, 1, REG_N>());

        // [2*REG_M, REG_N] => [REG_M, REG_K = REG_N + REG_N]
        matrix<half, REG_N, REG_K> P; // REG_K = REG_N + REG_N
        Transpose_8x8(St.select<REG_M, 1, REG_N, 1>(0,0), P.select<REG_M, 1, REG_N, 1>(0,0));
        Transpose_8x8(St.select<REG_M, 1, REG_N, 1>(REG_M,0), P.select<REG_M, 1, REG_N, 1>(0,REG_N));

        //show(P);return;

        if (kv_pos == 0) {
            matrix<half, REG_K/2, REG_N*2> Vmat;
            #pragma unroll
            for(int k = 0, ri = 0; k < head_size; k += REG_N, ri++) {
                // V has been VNNI-prepacked
                cm_slm_block_read(slm_V, GENX_NONE, REG_K*k*sizeof(half), Vmat.format<half>());
                rO[ri] = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount, float>(0, Vmat.format<int32_t>(), P.format<int32_t>());
            }
        } else {
            matrix<half, REG_K/2, REG_N*2> Vmat;
            #pragma unroll
            for(int k = 0, ri=0; k < head_size; k += REG_N, ri++) {
                // V has been VNNI-prepacked
                cm_slm_block_read(slm_V, GENX_NONE, REG_K*k*sizeof(half), Vmat.format<half>());

                //# compensate cur_O
                //  matrix <float, head_size/REG_K*2, REG_M*REG_N> rO;
                auto cO = rO[ri].format<float, REG_M, REG_N>();
                #pragma unroll
                for(int r = 0; r < REG_M; r++)
                    cO.row(r) = cm_mul<float>(cO.row(r), max_comp[r]);

                //# show(cur_O.format<float, 2*REG_M, REG_N>()); return;
                
                rO[ri] = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount>(
                                rO[ri].format<float>(),
                                Vmat.format<int32_t>(),
                                P.format<int32_t>());
                //show(rO[ri].format<float, REG_M, REG_N>());
                // if (kv_pos == args_verbose) show(cur_O.format<float, 2*REG_M, REG_N>());
            }
            // if (kv_pos == args_verbose) return;
        }
        cur_max = new_max_t;
    }//# for(int kv_pos = 0; kv_pos < kv_len; kv_pos += kv_step) {

    //# save cur_O/cur_sum.transpose(0, 1)
    matrix<float, REG_M, REG_N> cur_O;
    matrix<half, REG_M, REG_N> cur_O_f16;
    cur_sum = cm_inv(cur_sum);
    #pragma unroll
    for(int k = 0, ri=0; k < head_size; k += REG_N, ri++) {
        auto cO = rO[ri].format<float, REG_M, REG_N>();
        for(int r = 0; r < cO.n_rows(); r++) {
            cur_O_f16[r] = cm_mul<float>(cO[r], cur_sum[r]);
        }

        // if (i == args_verbose) show(cur_O_f16);
        svm_write_2d(cur_O_f16, o_base + k*sizeof(half), qo_pitch);

        // cm_store(b2dO.set_block_x(k).set_block_y(q_start), cur_O_f16.format<half, 2, REG_M*REG_N>()[0]);
        // cm_store(b2dO.set_block_x(k).set_block_y(q_start + REG_M), cur_O_f16.format<half, 2, REG_M*REG_N>()[1]);
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
    half* mask [[type("svmptr_t")]],
    __global uint64_t* cminfo [[type("svmptr_t")]]
    ) {
    CMTracer_begin(&cminfo);
    //# query [batch, q_len, num_heads, S]
    //#   key [batch, kv_len, num_heads, S]
    //# value [batch, kv_len, num_heads, S]
    //# to load Q

    constexpr uint K_SLM_SIZE = (kv_step * head_size * sizeof(half));
    constexpr uint V_SLM_SIZE = (kv_step * head_size * sizeof(half));

    cm_slm_init(K_SLM_SIZE + V_SLM_SIZE);

    auto slm_K = cm_slm_alloc(K_SLM_SIZE);
    auto slm_V = cm_slm_alloc(V_SLM_SIZE);

    auto batch = cm_group_id(dim_batch);
    auto h = head_base_id + cm_group_id(dim_head);
    auto hkv = h / (num_heads/num_kv_heads);
    auto local_id = cm_local_id(dim_q_batch);
    int q_group_id = cm_group_id(dim_q_batch);
    const int local_size = cm_linear_local_size();
    int kv_stop = kv_len;
#if causal_mask
    kv_stop = (q_group_id + 1) * local_size * q_step;
    if (kv_stop > kv_len) kv_stop = kv_len;
#endif
    {
        auto q_start = (q_group_id * local_size + local_id) * q_step;
        auto q_base = reinterpret_cast<svmptr_t>(query + ((batch*q_len + q_start)*num_heads + h)*head_size);
        auto k_base = reinterpret_cast<svmptr_t>(key + (batch*num_kv_heads*kv_len + hkv)*head_size);
        auto v_base = reinterpret_cast<svmptr_t>(value + (batch*num_kv_heads*kv_len + hkv)*head_size);
        auto o_base = reinterpret_cast<svmptr_t>(output + ((batch * q_len + q_start)*num_heads + h)*head_size);
        svmptr_t mask_base;
        if (causal_mask)
            mask_base = reinterpret_cast<svmptr_t>(mask);
        else
            mask_base = reinterpret_cast<svmptr_t>(mask + (batch * q_len + q_start) * kv_len);

        row_kernel<causal_mask>(
            slm_K,
            slm_V,
            local_id,
            q_start,
            kv_stop,
            head_base_id,
            kv_len,
            q_base,
            k_base,
            v_base,
            o_base,
            mask_base);
    }

    {
        auto q_start = (q_len - q_group_id * local_size * q_step) - (local_size * q_step) + local_id*q_step;
        auto q_base = reinterpret_cast<svmptr_t>(query + ((batch*q_len + q_start)*num_heads + h)*head_size);
        auto k_base = reinterpret_cast<svmptr_t>(key + (batch*num_kv_heads*kv_len + hkv)*head_size);
        auto v_base = reinterpret_cast<svmptr_t>(value + (batch*num_kv_heads*kv_len + hkv)*head_size);
        auto o_base = reinterpret_cast<svmptr_t>(output + ((batch * q_len + q_start)*num_heads + h)*head_size);
        svmptr_t mask_base;
        if (causal_mask)
            mask_base = reinterpret_cast<svmptr_t>(mask);
        else
            mask_base = reinterpret_cast<svmptr_t>(mask + (batch * q_len + q_start) * kv_len);

    #if causal_mask
        kv_stop = (q_len - q_group_id * local_size * q_step);
        if (kv_stop > kv_len) kv_stop = kv_len;
    #endif
        row_kernel<causal_mask>(
            slm_K,
            slm_V,
            local_id,
            q_start,
            kv_stop,
            head_base_id,
            kv_len,
            q_base,
            k_base,
            v_base,
            o_base,
            mask_base);
    }

    CMTracer_end(&cminfo);
}