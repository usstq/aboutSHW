
#ifdef CM_HAS_LSC_UNTYPED_2D

#define cm_load_normal cm_load<lsc::Normal>
#define cm_load_transpose cm_load<lsc::Transpose>
#define cm_load_vnni cm_load<lsc::VNNI>
#define cm_store_normal cm_store

#else

template <typename T = int, unsigned NBlocks = 1, unsigned BlockH = 1, unsigned BlockW = 1>
inline void cm_load_normal(vector_ref<T, NBlocks*BlockH*BlockW> Res, const lsc::block_2d_desc<T, NBlocks, BlockH, BlockW> &Desc, int16_t Pred = 1) {
    static_assert(NBlocks == 1);
    auto pitch = Desc.get_pitch() + 1;
    auto base = reinterpret_cast<svmptr_t>(Desc.get_base() + Desc.get_block_y()*pitch + Desc.get_block_x() * sizeof(T));
    #pragma unroll
    for(int i = 0; i < BlockH; i++) {
        cm_svm_block_read(base + i * pitch, Res.select<BlockW, 1>(i*BlockW));
    }
}

template <typename T = int, unsigned NBlocks = 1, unsigned BlockH = 1, unsigned BlockW = 1>
inline void cm_load_transpose(vector_ref<T, NBlocks*BlockW*BlockH> Res, const lsc::block_2d_desc<T, NBlocks, BlockH, BlockW> &Desc, int16_t Pred = 1) {
    static_assert(NBlocks == 1);
    auto pitch = Desc.get_pitch() + 1;
    auto base = reinterpret_cast<svmptr_t>(Desc.get_base() + Desc.get_block_y()*pitch + Desc.get_block_x() * sizeof(T));
    matrix<T, BlockH, BlockW> temp;
    #pragma unroll
    for(int i = 0; i < BlockH; i++) {
        cm_svm_block_read(base + i * pitch, temp[i]);
    }
    Transpose2DMatrix(temp, Res.format<T, BlockW, BlockH>());
}

// in VNNI case, NBlocks is increasing along X dimension (increase cache-line usage)
template <typename T = int, unsigned NBlocks = 1, unsigned BlockH = 1, unsigned BlockW = 1>
inline void cm_load_vnni(vector_ref<T, NBlocks*BlockW*BlockH> Res, const lsc::block_2d_desc<T, NBlocks, BlockH, BlockW> &Desc, int16_t Pred = 1) {
    static_assert(NBlocks == 1 || NBlocks == 2);
    // each block must be a full XMX B matrix
    static_assert(BlockH == REG_K);
    static_assert(BlockW == REG_N);
    auto pitch = Desc.get_pitch() + 1;
    auto base = reinterpret_cast<svmptr_t>(Desc.get_base() + Desc.get_block_y()*pitch + Desc.get_block_x() * sizeof(T));
    matrix<T, BlockH, NBlocks * BlockW> temp;
    #pragma unroll
    for(int i = 0; i < BlockH; i++) {
        cm_svm_block_read(base + i * pitch, temp[i]);
    }

    auto out_vnni = Res.format<T, NBlocks * (BlockH/2), 2*BlockW>();
    #pragma unroll
    for(int i = 0; i < NBlocks; i ++) {
        out_vnni.select<BlockH/2, 1, BlockW, 2>(i*(BlockH/2), 0) = temp.select<BlockH/2, 2, BlockW, 1>(0, i*BlockW);
        out_vnni.select<BlockH/2, 1, BlockW, 2>(i*(BlockH/2), 1) = temp.select<BlockH/2, 2, BlockW, 1>(1, i*BlockW);
    }
}

template <typename T = int, unsigned NBlocks = 1, unsigned BlockH = 1, unsigned BlockW = 1>
inline void cm_store_normal(const lsc::block_2d_desc<T, NBlocks, BlockH, BlockW> &Desc, vector_ref<T, NBlocks*BlockW*BlockH> Res) {
    static_assert(NBlocks == 1);
    auto pitch = Desc.get_pitch() + 1;
    auto base = reinterpret_cast<svmptr_t>(Desc.get_base() + Desc.get_block_y()*pitch + Desc.get_block_x() * sizeof(T));
    #pragma unroll
    for(int i = 0; i < BlockH; i++) {
        cm_svm_block_write(base + i * pitch, Res.select<BlockW, 1>(i*BlockW));
    }
}
#endif

static_assert(q_step == REG_N);

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
    //CMTracer_begin(&cminfo);
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

    vector<float, q_step> cur_max;
    vector<float, q_step> cur_sum;

    cur_max = -1e9;
    cur_sum = 0;

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
    lsc::block_2d_desc<half, 1, REG_N, REG_K> b2dMask(mask + batch * q_len * kv_len, q_len - 1, kv_len*sizeof(half) - 1, kv_len*sizeof(half) - 1, 0, 0);

    //# b2dQ reinterpret as 32bit(DWORD) for transposed load(combined with VNNI)
    uint qo_pitch = num_heads * head_size * sizeof(half);
    uint kv_pitch = num_kv_heads * head_size * sizeof(half);
    lsc::block_2d_desc<uint, 1, REG_N, REG_K/2> b2dQ(reinterpret_cast<uint*>(query + (batch*num_heads*q_len + h)*head_size), q_len - 1, head_size*sizeof(half) - 1, qo_pitch - 1, 0, 0);
    lsc::block_2d_desc<half, 1, REG_M, REG_K> b2dK(key + (batch*num_kv_heads*kv_len + hkv)*head_size,   kv_len - 1, head_size*sizeof(half) - 1, kv_pitch - 1, 0, 0);
    lsc::block_2d_desc<half, 2, REG_K, REG_N> b2dV(value + (batch*num_kv_heads*kv_len + hkv)*head_size, kv_len - 1, head_size*sizeof(half) - 1, kv_pitch - 1, 0, 0);
    lsc::block_2d_desc<half, 1, REG_M, REG_N> b2dO(output + (batch*num_heads*q_len + h)*head_size,   q_len - 1, head_size*sizeof(half) - 1, qo_pitch - 1, 0, 0);

    //# load Qt into register & pack as VNNI & store to SLM (as dpas-B tile)

    matrix<half, head_size/REG_K, REG_K*REG_N> rQ;

    {
        //matrix<uint, REG_K/2, REG_N> Qmat;
        #pragma unroll
        for(int k = 0, ri = 0; k < head_size/2; k += REG_K/2, ri++) {
            b2dQ.set_block_x(k);

            //# DWORD transposed load == (transposed + VNNI) load
            cm_load_transpose(rQ[ri].format<uint>(), b2dQ.set_block_y(q_start));

            //show(rQ[ri].format<half, REG_K/2, REG_N*2>());
        }
    }

    //int kv_stop = (cm_group_id(2) + 1) * local_size * q_step;
    int kv_stop = kv_len;
#if causal_mask
    kv_stop = (q_group_id + 1) * local_size * q_step;
    if (kv_stop > kv_len) kv_stop = kv_len;
#endif

    constexpr int num_P_tiles = REG_N / REG_M;
    matrix <float, head_size/REG_N*num_P_tiles, REG_M*REG_N> rO;
    int causal_left = q_start;
    for(int kv_pos = 0; kv_pos < kv_stop; kv_pos += kv_step) {
        // if (args_verbose >= 0) printf("======== %d =========\n", kv_pos);
        //===========================================================
        //# load K into SLM as dpas-A tile (shared by all hw within same WG)
        //# load V into SLM as dpas-B tile (shared by all hw within same WG)
        {
            //# 1849 ~ 3259
            if (kv_pos > 0) cm_barrier();
            // auto clk0 = get_clock();
            if (local_size == 1) {
                matrix<half, REG_M, REG_K> temp0;
                matrix<half, REG_M, REG_K> temp1;
                for(int k = 0; k < head_size; k += REG_K) {
                    b2dK.set_block_x(k);
                    cm_load_normal(temp0.format<half>(), b2dK.set_block_y(kv_pos));
                    cm_load_normal(temp1.format<half>(), b2dK.set_block_y(kv_pos + REG_M));

                    //cm_prefetch(b2dK.set_block_y(kv_pos + kv_step));
                    //cm_prefetch(b2dK.set_block_y(kv_pos + kv_step + REG_M));

                    //show(temp0);                    show(temp1);                    return;
                    
                    uint offset = k * 2 * REG_M * sizeof(half);
                    cm_slm_block_write(slm_K, offset, temp0.format<half>());
                    offset += REG_M * REG_K * sizeof(half);
                    cm_slm_block_write(slm_K, offset, temp1.format<half>());
                }
                matrix<half, REG_K, 2*REG_N> temp2;
                b2dV.set_block_y(kv_pos);
                for(int k = 0; k < head_size; k += 2*REG_N) {
                    cm_load_vnni(temp2.format<half>(), b2dV.set_block_x(k).set_block_y(kv_pos));
                    //cm_prefetch(b2dV.set_block_y(kv_pos + kv_step));
                    //show(temp2);
                    cm_slm_block_write(slm_V, k * REG_K * sizeof(half), temp2.format<half>());
                }
            } else {
                if (wg_local_id < local_size/2) {
                    matrix<half, REG_M, REG_K> temp0;
                    matrix<half, REG_M, REG_K> temp1;
                    for(int k = REG_K*wg_local_id; k < head_size; k += REG_K*(local_size/2)) {
                        b2dK.set_block_x(k);
                        cm_load_normal(temp0.format<half>(), b2dK.set_block_y(kv_pos));
                        cm_load_normal(temp1.format<half>(), b2dK.set_block_y(kv_pos + REG_M));

                        //cm_prefetch(b2dK.set_block_y(kv_pos + kv_step));
                        //cm_prefetch(b2dK.set_block_y(kv_pos + kv_step + REG_M));

                        uint offset = k * 2 * REG_M * sizeof(half);
                        cm_slm_block_write(slm_K, offset, temp0.format<half>());
                        offset += REG_M * REG_K * sizeof(half);
                        cm_slm_block_write(slm_K, offset, temp1.format<half>());
                    }
                } else {
                    matrix<half, REG_K, 2*REG_N> temp2;
                    b2dV.set_block_y(kv_pos);
                    for(int k = 2*REG_N*(wg_local_id-local_size/2); k < head_size; k += 2*REG_N*(local_size/2)) {
                        cm_load_vnni(temp2.format<half>(), b2dV.set_block_x(k).set_block_y(kv_pos));
                        //cm_prefetch(b2dV.set_block_y(kv_pos + kv_step));

                        cm_slm_block_write(slm_V, k * REG_K * sizeof(half), temp2.format<half>());
                    }
                }
            }
            // printf(" diff= %lu\n", get_clock() - clk0);

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
        //show(St); return;
        //=========================================================== 361
        if constexpr (causal_mask) {
            auto cmask_off = kv_step - causal_left;
            St = cm_mul<float>(St, (float)scale_factor);  // convert scale_factor into (float), or it will be promoted to double
            if (cmask_off > 0) {
                //# full-attention: skip loading causal mask
                if (cmask_off > 2*kv_step) cmask_off = 2*kv_step;
                matrix<half, St.n_rows(), REG_N> temp;
                cm_svm_block_read(reinterpret_cast<svmptr_t>(mask + cmask_off * REG_N), temp.format<half>());
                St = cm_add<float>(St, temp);
            }
            causal_left -= kv_step;
        } else {
            matrix<half, REG_N, REG_K> Maskmat;
            cm_load_normal(Maskmat.format<half>(), b2dMask.set_block_x(kv_pos).set_block_y(q_start));

            matrix<float, REG_K, REG_N> MaskT;
            Transpose2DMatrix(Maskmat, MaskT);
            // show(MaskT); return;
            St = cm_mul<float>(St, (float)scale_factor);  // convert scale_factor into (float), or it will be promoted to double
            St = cm_add<float>(St, MaskT);
            //show(St); return;
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
                }
                //show(rO[ri].format<float, REG_M, REG_N>());

                // if (kv_pos == args_verbose) show(cur_O.format<float, 2*REG_M, REG_N>());
            }
            //if (kv_pos == args_verbose) return;
            // if (kv_pos == args_verbose) return;
        }
        //============================================================== 1168

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
            for(int r = 0; r < cO.n_rows(); r++) {
                cur_O_f16[r + p*REG_M] = cm_mul<float>(cO[r], cur_sum[r + p*REG_M]);
            }
        }

        // if (i == args_verbose) show(cur_O_f16);
        #pragma unroll
        for(int p = 0; p < num_P_tiles; p++) {
            cm_store_normal(b2dO.set_block_x(k).set_block_y(q_start + p*REG_M),
                     cur_O_f16.format<half, num_P_tiles, REG_M*REG_N>()[p]);
        }
        //cm_store(b2dO.set_block_x(k).set_block_y(q_start + REG_M), cur_O_f16.format<half, 2, REG_M*REG_N>()[1]);
    }
    // if (i == args_verbose) return;
    //CMTracer_end(&cminfo);
}
