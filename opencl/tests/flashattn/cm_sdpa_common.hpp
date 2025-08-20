//# CM kernel for flash attn, reference
#include <cm/cm.h>
#include <cm/cmtl.h>

//# CM-compiler is C++17
static_assert(__cplusplus >= 201703L);
//# static_assert(__cplusplus >= 202002L);
//# static_assert(__cplusplus >= 202302L);

#define SystolicDepth 8
#define RepeatCount 8
#define VNNI_WIDTH 2
#define REG_K (SystolicDepth * VNNI_WIDTH)
#define REG_M RepeatCount
//REG_N
//  Xe1:  8
//  Xe2: 16
#define REG_N (CM_GRF_WIDTH/32)

#define kv_step  REG_K
#define q_step   REG_N

constexpr float scale_factor = CMFLA_SCALE_FACTOR;

static_assert(q_step == 16 || q_step == 8);
static_assert(kv_step == 16);
static_assert(CM_HAS_DPAS);

template<typename T, int M, int N>
void show(const matrix<T, M, N> mat) {
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

template <typename T1, typename T2>
CM_INLINE void Transpose_16x16(matrix_ref<T1, 16, 16> in,
                               matrix_ref<T2, 16, 16> out) {
  matrix<T2, 16, 16> bBuf;
  bBuf.row(0) = in.template select<4, 1, 4, 4>(0, 0);   // 0,4,8,c
  bBuf.row(1) = in.template select<4, 1, 4, 4>(4, 0);   // 0,4,8,c
  bBuf.row(2) = in.template select<4, 1, 4, 4>(8, 0);   // 0,4,8,c
  bBuf.row(3) = in.template select<4, 1, 4, 4>(12, 0);  // 0,4,8,c
  bBuf.row(4) = in.template select<4, 1, 4, 4>(0, 1);   // 1,5,9,d
  bBuf.row(5) = in.template select<4, 1, 4, 4>(4, 1);   // 1,5,9,d
  bBuf.row(6) = in.template select<4, 1, 4, 4>(8, 1);   // 1,5,9,d
  bBuf.row(7) = in.template select<4, 1, 4, 4>(12, 1);  // 1,5,9,d
  bBuf.row(8) = in.template select<4, 1, 4, 4>(0, 2);   // 2,6,a,e
  bBuf.row(9) = in.template select<4, 1, 4, 4>(4, 2);   // 2,6,a,e
  bBuf.row(10) = in.template select<4, 1, 4, 4>(8, 2);  // 2,6,a,e
  bBuf.row(11) = in.template select<4, 1, 4, 4>(12, 2); // 2,6,a,e
  bBuf.row(12) = in.template select<4, 1, 4, 4>(0, 3);  // 3,7,b,f
  bBuf.row(13) = in.template select<4, 1, 4, 4>(4, 3);  // 3,7,b,f
  bBuf.row(14) = in.template select<4, 1, 4, 4>(8, 3);  // 3,7,b,f
  bBuf.row(15) = in.template select<4, 1, 4, 4>(12, 3); // 3,7,b,f

  out.row(0) = bBuf.template select<4, 1, 4, 4>(0, 0);   // 0
  out.row(1) = bBuf.template select<4, 1, 4, 4>(4, 0);   // 1
  out.row(2) = bBuf.template select<4, 1, 4, 4>(8, 0);   // 2
  out.row(3) = bBuf.template select<4, 1, 4, 4>(12, 0);  // 3
  out.row(4) = bBuf.template select<4, 1, 4, 4>(0, 1);   // 4
  out.row(5) = bBuf.template select<4, 1, 4, 4>(4, 1);   // 5
  out.row(6) = bBuf.template select<4, 1, 4, 4>(8, 1);   // 6
  out.row(7) = bBuf.template select<4, 1, 4, 4>(12, 1);  // 7
  out.row(8) = bBuf.template select<4, 1, 4, 4>(0, 2);   // 8
  out.row(9) = bBuf.template select<4, 1, 4, 4>(4, 2);   // 9
  out.row(10) = bBuf.template select<4, 1, 4, 4>(8, 2);  // a
  out.row(11) = bBuf.template select<4, 1, 4, 4>(12, 2); // b
  out.row(12) = bBuf.template select<4, 1, 4, 4>(0, 3);  // c
  out.row(13) = bBuf.template select<4, 1, 4, 4>(4, 3);  // d
  out.row(14) = bBuf.template select<4, 1, 4, 4>(8, 3);  // e
  out.row(15) = bBuf.template select<4, 1, 4, 4>(12, 3); // f
}

template <typename T1, typename T2>
CM_INLINE void Transpose_8x8(matrix_ref<T1, 8, 8> in, matrix_ref<T2, 8, 8> out) {
  matrix<T2, 8, 8> temp;
  temp.row(0) = in.template select<2, 1, 4, 2>(0, 0);
  temp.row(1) = in.template select<2, 1, 4, 2>(2, 0);
  temp.row(2) = in.template select<2, 1, 4, 2>(4, 0);
  temp.row(3) = in.template select<2, 1, 4, 2>(6, 0);
  temp.row(4) = in.template select<2, 1, 4, 2>(0, 1);
  temp.row(5) = in.template select<2, 1, 4, 2>(2, 1);
  temp.row(6) = in.template select<2, 1, 4, 2>(4, 1);
  temp.row(7) = in.template select<2, 1, 4, 2>(6, 1);

  out.row(0) = temp.template select<4, 1, 2, 4>(0, 0);
  out.row(2) = temp.template select<4, 1, 2, 4>(0, 1);
  out.row(4) = temp.template select<4, 1, 2, 4>(0, 2);
  out.row(6) = temp.template select<4, 1, 2, 4>(0, 3);
  out.row(1) = temp.template select<4, 1, 2, 4>(4, 0);
  out.row(3) = temp.template select<4, 1, 2, 4>(4, 1);
  out.row(5) = temp.template select<4, 1, 2, 4>(4, 2);
  out.row(7) = temp.template select<4, 1, 2, 4>(4, 3);
}

// function templates cannot be partially specialized; use overloading to achieve the same effect
template <typename T1, typename T2>
inline void Transpose2DMatrix(matrix_ref<T1, 8, 8> in, matrix_ref<T2, 8, 8> out) {
    Transpose_8x8(in, out);
}
template <typename T1, typename T2>
inline void Transpose2DMatrix(matrix_ref<T1, 16, 16> in, matrix_ref<T2, 16, 16> out) {
    Transpose_16x16(in, out);
}
template <typename T1, typename T2>
inline void Transpose2DMatrix(matrix_ref<T1, 16, 8> in, matrix_ref<T2, 8, 16> out) {
    Transpose_8x8(in.select<8, 1, 8, 1>(0,0), out.select<8, 1, 8, 1>(0,0));
    Transpose_8x8(in.select<8, 1, 8, 1>(8,0), out.select<8, 1, 8, 1>(0,8));
}
template <typename T1, typename T2>
inline void Transpose2DMatrix(matrix_ref<T1, 8, 16> in, matrix_ref<T2, 16, 8> out) {
    Transpose_8x8(in.select<8, 1, 8, 1>(0,0), out.select<8, 1, 8, 1>(0,0));
    Transpose_8x8(in.select<8, 1, 8, 1>(0,8), out.select<8, 1, 8, 1>(8,0));
}

template <int n_stride, typename T, int M, int N>
CM_INLINE void slm_read_2d(matrix_ref<T, M, N> out, uint slm, int offset) {
    #pragma unroll
    for(int i = 0; i < out.n_rows(); i++) {
        cm_slm_block_read(slm, GENX_DWALIGNED, offset + i*n_stride*sizeof(T), out.row(i));
    }
}

template <typename T, int M, int N>
CM_INLINE void svm_read_2d(matrix_ref<T, M, N> out, svmptr_t base, uint pitch) {
    #pragma unroll
    for(int i = 0; i < out.n_rows(); i++) {
        cm_svm_block_read(base + i * pitch, out[i]);
    }
}

template <int M, int N>
CM_INLINE void cm_load_2d(matrix_ref<uint, M, N> out, SurfaceIndex base, uint offset, uint pitch) {
    #pragma unroll
    for(int i = 0; i < out.n_rows(); i++) {
        out.row(i).format<uint>() = cm_load<uint, N>(base, offset + i * pitch);
    }
}

template <int M, int N>
CM_INLINE void cm_load_2d(matrix_ref<half, M, N> out, SurfaceIndex base, uint offset, uint pitch) {
    #pragma unroll
    for(int i = 0; i < out.n_rows(); i++) {
        out.row(i).format<uint>() = cm_load<uint, N/2>(base, offset + i * pitch);
    }
}

template <int M, int N>
CM_INLINE void cm_store_2d(matrix_ref<half, M, N> out, SurfaceIndex base, uint offset, uint pitch) {
    #pragma unroll
    for(int i = 0; i < out.n_rows(); i++) {
        cm_store<uint, N/2>(base, offset + i * pitch, out.row(i).format<uint>());
    }
}

template <typename T, int M, int N>
CM_INLINE void svm_read_2d(matrix_ref<T, M, N> out, svmptr_t base, vector_ref<uint, M> offsets) {
    #pragma unroll
    for(int i = 0; i < out.n_rows(); i++) {
        cm_svm_block_read(base + offsets[i], out[i]);
    }
}

template <typename T, int M, int N>
CM_INLINE void svm_read_2d(matrix_ref<T, M, N> out, svmptr_t base, uint pitch, int n_rows) {
    #pragma unroll
    for(int i = 0; i < out.n_rows(); i++, base += pitch, n_rows--) {
        if (n_rows > 0) cm_svm_block_read(base, out[i]);
    }
}

template <typename T, int M, int N>
CM_INLINE void svm_write_2d(matrix_ref<T, M, N> out, svmptr_t base, uint pitch) {
    #pragma unroll
    for(int i = 0; i < out.n_rows(); i++, base += pitch) {
        cm_svm_block_write(base, out[i]);
    }
}

template <typename T, int M, int N>
CM_INLINE void svm_write_2d(matrix_ref<T, M, N> out, svmptr_t base, uint pitch, int n_rows) {
    #pragma unroll
    for(int i = 0; i < out.n_rows(); i++, base += pitch) {
        if (i < n_rows) cm_svm_block_write(base, out[i]);
    }
}

CM_INLINE uint64_t get_clock() {
    auto clk = cm_clock();
    return ((uint64_t)clk[1]) << 32 | clk[0];
}


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

template<int num_P_tiles = REG_N/REG_M, int num_rO_tiles>
inline void ugemm_PV0(uint slm_V, matrix_ref<half, REG_N, REG_K> P, matrix_ref<float, num_rO_tiles, REG_M*REG_N> rO, uint slm_offset = 0) {
    constexpr int _head_size = num_rO_tiles*REG_N/num_P_tiles;

    auto P2 = P.format<half, num_P_tiles, REG_M * REG_K>();
    #pragma unroll
    for(int k = 0, ri = 0; k < _head_size; k += REG_N, ri += num_P_tiles) {
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

template<int num_P_tiles = REG_N/REG_M, int num_rO_tiles>
inline void ugemm_PV1(uint slm_V, matrix_ref<half, REG_N, REG_K> P, vector_ref<float, REG_N> max_comp,
                      matrix_ref<float, num_rO_tiles, REG_M*REG_N> rO, uint slm_offset = 0) {
    constexpr int _head_size = num_rO_tiles*REG_N/num_P_tiles;
    auto P2 = P.format<half, num_P_tiles, REG_M * REG_K>();
    #pragma unroll
    for(int k = 0, ri=0; k < _head_size; k += REG_N, ri += num_P_tiles) {
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

#ifdef CM_HAS_LSC_UNTYPED_2D
    #define cm_load_normal cm_load<lsc::Normal>
    #define cm_load_transpose cm_load<lsc::Transpose>
    #define cm_load_vnni cm_load<lsc::VNNI>
    #define cm_store_normal cm_store
#else
    // simulation of LSC API using SVM API
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



// //===============================================================================================
// template <int i, int N, int M>
// constexpr void apply_causal_mask(matrix_ref<float, N, M> St) {
//     if constexpr (i < N) {
//         St.row(i).select<i, 1>(0) = -3.4e38f;
//         apply_causal_mask<i + 1>(St);
//     }
// }


//===============================================================================================
template <int row, int maskcnt, int N, int M>
constexpr void apply_causal_mask(matrix_ref<float, N, M> St) {
    if constexpr (row < N) {
        if constexpr(maskcnt < M) {
            St.row(row).select<maskcnt, 1>(0) = -3.4e38f;
        } else {
            St.row(row) = -3.4e38f;

        }
        apply_causal_mask<row + 1, maskcnt + 1>(St);
    }
}

template <int N, int M>
void apply_causal_mask_with_row(matrix_ref<float, N, M> St , const int row) {
    if (row == 1) {
            apply_causal_mask<1, 1>(St);
    } else if (row == 2) {
            apply_causal_mask<2, 1>(St);
    }  else if(row == 3) {
            apply_causal_mask<3, 1>(St);
    }  else if(row == 4) {
            apply_causal_mask<4, 1>(St);
    }  else if(row == 5) {
            apply_causal_mask<5, 1>(St);
    }  else if(row == 6) {
            apply_causal_mask<6, 1>(St);
    }  else if(row == 7) {
            apply_causal_mask<7, 1>(St);
    }  else if(row == 8) {
            apply_causal_mask<8, 1>(St);
    }  else if(row == 9) {
            apply_causal_mask<9, 1>(St);
    }  else if(row == 10) {
            apply_causal_mask<10, 1>(St);
    }  else if(row == 11) {
            apply_causal_mask<11, 1>(St);
    }  else if(row == 12) {
            apply_causal_mask<12, 1>(St);
    }  else if(row == 13) {
            apply_causal_mask<13, 1>(St);
    }  else if(row == 14) {
            apply_causal_mask<14, 1>(St);
    }  else if(row == 15) {
            apply_causal_mask<15, 1>(St);
    }
}


template <int N, int M>
void apply_causal_mask_with_maskcnt(matrix_ref<float, N, M> St , const int mask_cnt_base) {
    if (mask_cnt_base == 1) {
            apply_causal_mask<0, 1>(St);
    } else if (mask_cnt_base == 2) {
            apply_causal_mask<0, 2>(St);
    }  else if(mask_cnt_base == 3) {
            apply_causal_mask<0, 3>(St);
    }  else if(mask_cnt_base == 4) {
            apply_causal_mask<0, 4>(St);
    }  else if(mask_cnt_base == 5) {
            apply_causal_mask<0, 5>(St);
    }  else if(mask_cnt_base == 6) {
            apply_causal_mask<0, 6>(St);
    }  else if(mask_cnt_base == 7) {
            apply_causal_mask<0, 7>(St);
    }  else if(mask_cnt_base == 8) {
            apply_causal_mask<0, 8>(St);
    }  else if(mask_cnt_base == 9) {
            apply_causal_mask<0, 9>(St);
    }  else if(mask_cnt_base == 10) {
            apply_causal_mask<0, 10>(St);
    }  else if(mask_cnt_base == 11) {
            apply_causal_mask<0, 11>(St);
    }  else if(mask_cnt_base == 12) {
            apply_causal_mask<0, 12>(St);
    }  else if(mask_cnt_base == 13) {
            apply_causal_mask<0, 13>(St);
    }  else if(mask_cnt_base == 14) {
            apply_causal_mask<0, 14>(St);
    }  else if(mask_cnt_base == 15) {
            apply_causal_mask<0, 15>(St);
    }
}

template<bool use_causal_mask, int num_heads, int num_kv_heads, int head_size, int is_qkv_fused, int wg_local_size>
void sdpa_kernel_lsc_prefetch(
    int wg_local_id,
    int q_start,
    int kv_stop, //
    int q_len, //q_step
    int kv_len, //not used for now
    svmptr_t q_base [[type("svmptr_t")]],
    svmptr_t k_base [[type("svmptr_t")]],
    svmptr_t v_base [[type("svmptr_t")]],
    svmptr_t o_base [[type("svmptr_t")]],
    int32_t past_lens,
    int32_t* block_indices [[type("svmptr_t")]],
    int32_t block_indices_begins) {
    constexpr uint o_pitch = (num_heads * head_size * sizeof(half));
    constexpr uint q_pitch = is_qkv_fused ? ((num_heads + num_kv_heads*2) * head_size * sizeof(half)) : o_pitch;
    // constexpr uint k_pitch = is_qkv_fused ? q_pitch : (num_kv_heads * head_size * sizeof(half));
    // constexpr uint v_pitch = is_qkv_fused ? q_pitch : (num_kv_heads * head_size * sizeof(half));
    //[block_num, kv_heads, block_size, head_size]
    constexpr uint k_pitch =  head_size * sizeof(half);
    constexpr uint v_pitch = k_pitch;
    bool debug_data=false;
    // if (cm_local_id(2) == 0 && (cm_group_id(2) == 0))
    //     debug_data=true;

    vector<float, q_step> cur_max;
    vector<float, q_step> cur_sum;

    cur_max = -3e38f;
    cur_sum = 0;
    constexpr int num_P_tiles = REG_N / REG_M;
    matrix<half, head_size/REG_K, REG_K*REG_N> rQ;
    matrix <float, head_size/REG_N*num_P_tiles, REG_M*REG_N> rO;

    auto q_tokens_left = q_len;// - q_start;
    static_assert(q_step == REG_N);
    static_assert(kv_step == REG_K);

    if (q_tokens_left < 0) q_tokens_left = 0;
    if (q_tokens_left > q_step) q_tokens_left = q_step;

    if (q_tokens_left > 0) {
        lsc::block_2d_desc<uint, 1, REG_N, REG_K/2> b2dQ(reinterpret_cast<uint*>(q_base), q_tokens_left - 1, head_size*sizeof(half) - 1, q_pitch - 1, 0, 0);
        #pragma unroll
        for(int k = 0, ri = 0; k < head_size/2; k += REG_K/2, ri++) {
            if (debug_data) {
                cm_load<lsc::Normal>(rQ[ri].format<uint>(), b2dQ.set_block_x(k));
                printf("qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq:wg:%d\n", cm_group_id(2));
                show(rQ.format<half, 16, head_size>());
            }
            cm_load<lsc::Transpose>(rQ[ri].format<uint>(), b2dQ.set_block_x(k));
            rQ[ri].format<half>() = cm_mul<half>(rQ[ri].format<half>(), (half)scale_factor);

        }
    }

    lsc::block_2d_desc<half, 1, kv_step, REG_K> b2dK(k_base, CMPA_BLOCK_SZ - 1, head_size*sizeof(half) - 1, k_pitch - 1, 0, 0);
    lsc::block_2d_desc<half, 1, REG_K, REG_N> b2dV(v_base, CMPA_BLOCK_SZ - 1, head_size*sizeof(half) - 1, v_pitch - 1, 0, 0);

    static_assert(wg_local_size == 16);
    lsc::block_2d_desc<half, 1, kv_step/wg_local_size, REG_K> prefetch_K(k_base, CMPA_BLOCK_SZ - 1, head_size*sizeof(half) - 1, k_pitch - 1, 0, 0);
    lsc::block_2d_desc<half, 1, REG_K/wg_local_size, REG_N> prefetch_V(v_base, CMPA_BLOCK_SZ - 1, head_size*sizeof(half) - 1, v_pitch - 1, 0, 0);
    constexpr int blk_stride = CMFLA_NUM_KV_HEADS*CMFLA_HEAD_SIZE*CMPA_BLOCK_SZ;
    int causal_left = q_start+past_lens;

    for(int kv_pos = 0; kv_pos < kv_stop; kv_pos += kv_step) {
        auto cur_block_id = block_indices[kv_pos / CMPA_BLOCK_SZ + block_indices_begins];
        uint32_t prefetch_kv_pos = (kv_pos+kv_step) >= kv_stop ?  kv_pos : (kv_pos+kv_step);

        auto prefetch_block_id = block_indices[prefetch_kv_pos / CMPA_BLOCK_SZ + block_indices_begins];
        //# St = k @ Qt
        matrix<float, kv_step, q_step> St; // = ugemm_KQ(slm_K, rQ, slm_offset);
        {
            constexpr int num_K = kv_step/REG_M;
            auto St2 = St.format<float, num_K, REG_M*REG_N>();

            matrix<half, num_K, REG_M * REG_K> Kmat;
            //cm_slm_block_read(slm_K, GENX_NONE, slm_offset, Kmat.format<half>());

            prefetch_K.set_base_ptr((reinterpret_cast<half*>(k_base)+prefetch_block_id*blk_stride));
            prefetch_K.set_block_y((prefetch_kv_pos + wg_local_id) % CMPA_BLOCK_SZ);
            cm_prefetch<CacheHint::Cached, CacheHint::Cached>(prefetch_K.set_block_x(0));

            b2dK.set_base_ptr((reinterpret_cast<half*>(k_base)+cur_block_id*blk_stride));
            b2dK.set_block_y(kv_pos%CMPA_BLOCK_SZ);
            cm_load<lsc::Normal>(Kmat.format<half>(), b2dK.set_block_x(0));
            if (debug_data) {
                    printf("kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk:wg:%d\n", cm_group_id(2));
                    show(Kmat.format<half, 16, 16>());
                }
            #pragma unroll
            for(int k = 0; k < num_K; k++)
                St2.row(k) = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount, float>(
                                0,
                                rQ[0].format<int32_t>(),
                                Kmat[k].format<int32_t>());

            #pragma unroll
            for(int ri = 1; ri < head_size/REG_K; ri++) {
                //cm_slm_block_read(slm_K, GENX_NONE, slm_offset + ri * Kmat.n_elems() * sizeof(half), Kmat.format<half>());
                cm_prefetch<CacheHint::Cached, CacheHint::Cached>(prefetch_K.set_block_x(ri*REG_K));
                cm_load<lsc::Normal>(Kmat.format<half>(), b2dK.set_block_x(ri*REG_K));
                if (debug_data) {
                    printf("kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk:wg%d\n", cm_group_id(2));
                    show(Kmat.format<half, 16, 16>());
                }

                #pragma unroll
                for(int k = 0; k < num_K; k++) {
                    St2.row(k) = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount, float>(
                        St2.row(k),
                        rQ[ri].format<int32_t>(),
                        Kmat[k].format<int32_t>());
                }
            }
        }
        if (debug_data) {
                    printf("\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\score0:wg:%d\n", cm_group_id(2));
                    show(St.format<float, 16, 16>());
                    printf("causal_left:%d use_causal_mask%d\n", causal_left, use_causal_mask);

        }

        if constexpr (use_causal_mask) {

            // since kv_step == q_step == 16, causal_left is n*kv_step
            if (causal_left >= 0  && causal_left <= 14) {
                // printf("\\\\\\\\\\causal_left: %d, q_start:%d, past_lens:%d, q_len:%d\n", causal_left, q_start, past_lens, q_tokens_left);
                apply_causal_mask_with_row(St , (causal_left+1));
                //apply_causal_mask<3, 1>(St);
                //apply_causal_mask(St , (causal_left+1));
                // St.row(3).select<1, 1>(0) = -3.4e38f;
                // St.row(4).select<2, 1>(0) = -3.4e38f;
                // St.row(5).select<3, 1>(0) = -3.4e38f;
                // St.row(6).select<4, 1>(0) = -3.4e38f;
                // St.row(7).select<5, 1>(0) = -3.4e38f;
                // St.row(8).select<6, 1>(0) = -3.4e38f;
                // St.row(9).select<7, 1>(0) = -3.4e38f;
                // St.row(10).select<8, 1>(0) = -3.4e38f;
                // St.row(11).select<9, 1>(0) = -3.4e38f;
                // St.row(12).select<10, 1>(0) = -3.4e38f;
                // St.row(13).select<11, 1>(0) = -3.4e38f;
                // St.row(14).select<12, 1>(0) = -3.4e38f;
                // St.row(15).select<13, 1>(0) = -3.4e38f;

            } else if (causal_left < 0 && causal_left > -16) {
                apply_causal_mask_with_maskcnt(St , (0-causal_left));
            } else if (causal_left <= -16) {
                St = -3.4e38f;
            }
            causal_left -= kv_step;

        } else {
            int kv_tokens = kv_stop - kv_pos;
            // LSC ensures no overflow-access, but mask off k-tails attn-score is still required
            for(int p = kv_tokens; p < kv_step; p++) St[p] = -3.4e38f;
        }

        if (debug_data) {
                    printf("\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\score1 :wg:%d\n", cm_group_id(2));
                    show(St.format<float, 16, 16>());
        }
        //show(St);
        auto max_comp = online_softmax_update(St, cur_max, cur_sum);

        matrix<half, REG_N, REG_K> P;
        Transpose2DMatrix(St, P);

        prefetch_V.set_base_ptr((reinterpret_cast<half*>(v_base)+prefetch_block_id*blk_stride));
        prefetch_V.set_block_y((prefetch_kv_pos + wg_local_id) % CMPA_BLOCK_SZ);

        b2dV.set_base_ptr((reinterpret_cast<half*>(v_base)+cur_block_id*blk_stride));
        b2dV.set_block_y(kv_pos%CMPA_BLOCK_SZ);
        if (kv_pos == 0) {
            // ugemm_PV0(slm_V, P, rO, slm_offset);
            auto P2 = P.format<half, num_P_tiles, REG_M * REG_K>();
            #pragma unroll
            for(int k = 0, ri = 0; k < head_size; k += REG_N, ri += num_P_tiles) {
                matrix<half, REG_K/2, REG_N*2> Vmat;
                cm_prefetch<CacheHint::Cached, CacheHint::Cached>(prefetch_V.set_block_x(k));

                if (debug_data) {
                    cm_load<lsc::Normal>(Vmat.format<half>(), b2dV.set_block_x(k));
                    printf("vvvvvvvvvvvvvvvvvvvvvvvv :wg%d\n", cm_group_id(2));
                    show(Vmat.format<half, 16, 16>());
                }
                cm_load<lsc::VNNI>(Vmat.format<half>(), b2dV.set_block_x(k));
                #pragma unroll
                for(int p = 0; p < num_P_tiles; p++) {
                    rO[ri + p] = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount, float>(
                                    0,
                                    Vmat.format<int32_t>(),
                                    P2.row(p).format<int32_t>());
                }
            }
        }
        else {
            //ugemm_PV1(slm_V, P, max_comp, rO, slm_offset);
            auto P2 = P.format<half, num_P_tiles, REG_M * REG_K>();
            #pragma unroll
            for(int k = 0, ri=0; k < head_size; k += REG_N, ri += num_P_tiles) {
                matrix<half, REG_K/2, REG_N*2> Vmat;

                cm_prefetch<CacheHint::Cached, CacheHint::Cached>(prefetch_V.set_block_x(k));
                if (debug_data) {
                    cm_load<lsc::Normal>(Vmat.format<half>(), b2dV.set_block_x(k));
                    printf("vvvvvvvvvvvvvvvvvvvvvvvv :wg%d\n", cm_group_id(2));
                    show(Vmat.format<half, 16, 16>());
                }
                cm_load<lsc::VNNI>(Vmat.format<half>(), b2dV.set_block_x(k));

                //# compensate cur_O
                //  matrix <float, head_size/REG_K*2, REG_M*REG_N> rO;
                #pragma unroll
                for(int p = 0; p < num_P_tiles; p++) {
                    auto cO = rO[ri + p].format<float, REG_M, REG_N>();
                    #pragma unroll
                    for(int r = 0; r < REG_M; r++)
                        cO.row(r) = cm_mul<float>(cO.row(r), max_comp[r + p*REG_M]);
                }

                #pragma unroll
                for(int p = 0; p < num_P_tiles; p++) {
                    rO[ri + p] = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount>(
                                rO[ri + p].format<float>(),
                                Vmat.format<int32_t>(),
                                P2.row(p).format<int32_t>());
                }
            }
        }
    }
    if (q_tokens_left == 0) return;

    //# save cur_O/cur_sum.transpose(0, 1)
    matrix<half, num_P_tiles*REG_M, REG_N> cur_O_f16;
    cur_sum = cm_inv(cur_sum);

    lsc::block_2d_desc<half, 1, REG_M, REG_N> b2dO(o_base, q_tokens_left - 1, head_size*sizeof(half) - 1, o_pitch - 1, 0, 0);

    #pragma unroll
    for(int k = 0, ri=0; k < head_size; k += REG_N, ri += num_P_tiles) {
        #pragma unroll
        for(int p = 0; p < num_P_tiles; p++) {
            auto cO = rO[ri + p].format<float, REG_M, REG_N>();
            #pragma unroll
            for(int r = 0; r < cO.n_rows(); r++) {
                cur_O_f16[r + p*REG_M] = cm_mul<float>(cO.row(r), cur_sum[r + p*REG_M]);

            }
        }
        if (debug_data) {
            printf("oooooooooooooooooooooooooooooooo :wg%d\n", cm_group_id(2));
            show(cur_O_f16.format<half, 16, 16>());
        }
        b2dO.set_block_x(k);
        cm_store(b2dO.set_block_y(0), cur_O_f16.format<half, num_P_tiles, REG_M * REG_N>().row(0));
        cm_store(b2dO.set_block_y(REG_M), cur_O_f16.format<half, num_P_tiles, REG_M * REG_N>().row(1));
    }
}
