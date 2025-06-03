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

template <typename T, int M, int N>
CM_INLINE void svm_read_2d(matrix_ref<T, M, N> out, svmptr_t base, uint pitch) {
    #pragma unroll
    for(int i = 0; i < out.n_rows(); i++) {
        cm_svm_block_read(base + i * pitch, out[i]);
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
