#!/usr/bin/python3
from clops import cl
import numpy as np
import sys

def test_mm(M, K, N, K_group_size, w_dtype):
    np.random.seed(0)
    A = np.random.randint(-1,2,[M, K]).astype(np.float16)
    tA = cl.tensor(A)
    tC = cl.tensor(np.zeros([M, N], dtype=np.float16))

    if w_dtype == cl.onednn_dtype.f16:
        B = np.random.randint(-1,2,[K, N]).astype(np.float16)
        C = A @ B
        print("ref is calculated!")

        tB = cl.tensor(B)
        with_wc_scales = False
        with_wc_zp = False
        with_silu = False
        mm = cl.onednn_mm(cl.onednn_dtype.f16, w_dtype, K, N, 0,
                            with_wc_scales,
                            with_wc_zp,
                            with_silu)
        wmem = mm.get_wmem(cl.onednn_dtype.f16, tB, cl.tensor(), cl.tensor())


    if w_dtype == cl.onednn_dtype.s8:
        assert (K % K_group_size) == 0
        K_groups = K // K_group_size
        B_q = np.random.randint(-1,2,[K, N]).astype(np.int8)
        B_scale = np.random.randint(-4,5,[K_groups, N]).astype(np.float32)
        B_zp = np.random.randint(-1,2,[K_groups, N]).astype(np.int8)

        B = (B_q.astype(np.float32) - B_zp.astype(np.float32).repeat(K_group_size, axis=0)) * B_scale.repeat(K_group_size, axis=0)

        C = A @ B
        print("ref is calculated!")

        tBq = cl.tensor(B_q)
        tBs = cl.tensor(B_scale)
        tBz = cl.tensor(B_zp)

        with_wc_scales = True
        with_wc_zp = True
        with_silu = False
        mm = cl.onednn_mm(cl.onednn_dtype.f16, w_dtype, K, N, K_group_size,
                            with_wc_scales,
                            with_wc_zp,
                            with_silu)
        wmem = mm.get_wmem(cl.onednn_dtype.s8, tBq, tBs, tBz)


    if w_dtype == cl.onednn_dtype.s4:
        assert (K % K_group_size) == 0
        K_groups = K // K_group_size
        B_q = np.random.randint(-1,2,[K, N]).astype(np.int8)
        B_scale = np.random.randint(-4,5,[K_groups, N]).astype(np.float32)
        B_zp = np.random.randint(-1,2,[K_groups, N]).astype(np.int8)

        B = (B_q.astype(np.float32) - B_zp.astype(np.float32).repeat(K_group_size, axis=0)) * B_scale.repeat(K_group_size, axis=0)

        C = A @ B
        print("ref is calculated!")

        tBq = cl.tensor(B_q)
        tBs = cl.tensor(B_scale)
        tBz = cl.tensor(B_zp)

        with_wc_scales = True
        with_wc_zp = True
        with_silu = False
        mm = cl.onednn_mm(cl.onednn_dtype.f16, w_dtype, K, N, K_group_size,
                            with_wc_scales,
                            with_wc_zp,
                            with_silu)
        wmem = mm.get_wmem(cl.onednn_dtype.s8, tBq, tBs, tBz)

    mm.forward(tA, wmem, tC)

    cl.finish()

    C1 = tC.numpy()

    if not np.allclose(C, C1):
        print(C)
        print(C1)

#test_mm(M = 1, K = 768, N = 2048, K_group_size = 0, w_dtype = cl.onednn_dtype.f16)
#test_mm(M = 1, K = 768, N = 2048, K_group_size = 32, w_dtype = cl.onednn_dtype.s8)
test_mm(M = 1, K = 768, N = 2048, K_group_size = 32, w_dtype = cl.onednn_dtype.s4)
