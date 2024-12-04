#include <cstdlib>
#include <memory>
#include <stdexcept>

#define JIT_DEBUG 1
#include "../include/jit.h"
#include "omp.h"


class GemmRegBlocking : public jit_generator {
public:
    const int m_rows;
    const int m_cols;
    const bool m_preload_b;
    const int m_prefetch_B_adv;
    GemmRegBlocking(int rows, int cols, int prefetch_B_adv = 0) : m_rows(rows), m_cols(cols), m_preload_b(rows >= cols), m_prefetch_B_adv(prefetch_B_adv) {
        if (m_preload_b) {
            ASSERT(rows * cols + cols + 1 <= 16);
        } else {
            ASSERT(rows * cols + rows + 1 <= 16);
        }
        create_kernel("GemmRegBlocking");
    }

    struct CallArgs {
        const float* A;
        int64_t A_stride;  // stride in number of bytes
        const float* B;
        int64_t B_stride;  // stride in number of bytes
        float* C;
        int64_t C_stride;  // stride in number of bytes
        int64_t K;
        int64_t accumulate;
    };

    Xbyak::Ymm ymmC(int row, int col) {
        return Xbyak::Ymm(row * m_cols + col);
    }
    Xbyak::Ymm ymmB(int col) {
        if (m_preload_b)
            return Xbyak::Ymm(m_rows * m_cols + col);
        else
            return Xbyak::Ymm(m_rows * m_cols);
    }
    Xbyak::Ymm ymmA(int row) {
        if (m_preload_b)
            return Xbyak::Ymm(m_rows * m_cols + m_cols);
        else
            return Xbyak::Ymm(m_rows * m_cols + 1 + row);
    }

    void generate() override {
        auto A_ptr = abi_param2;
        auto A_stride = abi_param3;
        auto B_ptr = abi_param4;
        auto B_stride = abi_param5;

        mov(A_ptr, ptr[abi_param1 + offsetof(CallArgs, A)]);
        mov(A_stride, ptr[abi_param1 + offsetof(CallArgs, A_stride)]);
        mov(B_ptr, ptr[abi_param1 + offsetof(CallArgs, B)]);
        mov(B_stride, ptr[abi_param1 + offsetof(CallArgs, B_stride)]);

        // initilaize C
        {
            Xbyak::Label skip_load;
            auto reg_tmp = rax;
            for (int r = 0; r < m_rows; r++)
                for (int c = 0; c < m_cols; c++) {
                    auto ymm = ymmC(r, c);
                    vxorps(ymm, ymm, ymm);
                }

            mov(reg_tmp, ptr[abi_param1 + offsetof(CallArgs, accumulate)]);
            and_(reg_tmp, 1);
            jz(skip_load);
            {
                auto dst_ptr = r10;
                auto dst_stride = r11;
                mov(dst_ptr, ptr[abi_param1 + offsetof(CallArgs, C)]);
                mov(dst_stride, ptr[abi_param1 + offsetof(CallArgs, C_stride)]);

                // load subC[m_rows, m_cols]
                for (int r = 0; r < m_rows; r++) {
                    for (int c = 0; c < m_cols; c++) {
                        vmovups(ymmC(r, c), ptr[dst_ptr + c * 32]);
                    }
                    add(dst_ptr, dst_stride);
                }
            }
            L(skip_load);
        }

        // loop over K
        //            B:    1 x cols regs
        // A : 1 regs C: rows x cols regs
        {
            Xbyak::Label loop_over_k;
            auto reg_k = r10;
            auto A_ptr3 = r11;

            auto loadA = [&](int r) {
                switch (r) {
                case 0:
                    vbroadcastss(ymmA(r), ptr[A_ptr]);
                    break;
                case 1:
                    vbroadcastss(ymmA(r), ptr[A_ptr + A_stride]);
                    break;
                case 2:
                    vbroadcastss(ymmA(r), ptr[A_ptr + 2 * A_stride]);
                    break;
                case 3:
                    vbroadcastss(ymmA(r), ptr[A_ptr3]);
                    break;
                case 4:
                    vbroadcastss(ymmA(r), ptr[A_ptr3 + A_stride]);
                    break;
                case 5:
                    vbroadcastss(ymmA(r), ptr[A_ptr3 + 2 * A_stride]);
                    break;
                default:
                    throw std::runtime_error("number of reg-blocking rows is not supported");
                }
            };

            if (m_rows > 3) {
                lea(A_ptr3, ptr[A_ptr + 2 * A_stride]);
                lea(A_ptr3, ptr[A_ptr3 + A_stride]);
            }
            mov(reg_k, ptr[abi_param1 + offsetof(CallArgs, K)]);

            align(64, false);
            L(loop_over_k);
            if (m_preload_b) {
                // preload B regs
                for (int c = 0; c < m_cols; c++)
                    vmovups(ymmB(c), ptr[B_ptr + c * 32]);

                if (m_prefetch_B_adv > 0)
                    prefetcht0(ptr[B_ptr + m_prefetch_B_adv]);

                lea(B_ptr, ptr[B_ptr + B_stride]);
                for (int r = 0; r < m_rows; r++) {
                    loadA(r);
                    for (int c = 0; c < m_cols; c++)
                        vfmadd231ps(ymmC(r, c), ymmA(r), ymmB(c));
                }

                lea(A_ptr, ptr[A_ptr + 4]);
                if (m_rows > 3)
                    lea(A_ptr3, ptr[A_ptr3 + 4]);
            } else {
                // preload A regs
                for (int r = 0; r < m_rows; r++)
                    loadA(r);

                for (int c = 0; c < m_cols; c++) {
                    vmovups(ymmB(c), ptr[B_ptr + c * 32]);
                    for (int r = 0; r < m_rows; r++)
                        vfmadd231ps(ymmC(r, c), ymmA(r), ymmB(c));
                }

                lea(B_ptr, ptr[B_ptr + B_stride]);
                lea(A_ptr, ptr[A_ptr + 4]);
                if (m_rows > 3)
                    lea(A_ptr3, ptr[A_ptr3 + 4]);
            }
            dec(reg_k);
            jnz(loop_over_k, T_NEAR);
        }

        // save C
        {
            auto dst_ptr = r10;
            auto dst_stride = r11;
            mov(dst_ptr, ptr[abi_param1 + offsetof(CallArgs, C)]);
            mov(dst_stride, ptr[abi_param1 + offsetof(CallArgs, C_stride)]);
            for (int r = 0; r < m_rows; r++) {
                for (int c = 0; c < m_cols; c++) {
                    vmovups(ptr[dst_ptr + c * 32], ymmC(r, c));
                }
                add(dst_ptr, dst_stride);
            }
        }
        ret();
    }
};