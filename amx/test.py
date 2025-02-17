import csrc
import numpy as np
import time

def test_amx_repack_B():
    np.random.seed(0)
    src = np.random.randint(low=-100, high=100, size=(160, 160)).astype(np.float32)
    dst = csrc.test_amx_repack_B(src)
    if (dst != np.transpose(src[:16, :16])).any():
        print(src)
        print(dst)
        assert False, "amx_repack_B failed!"

def test():
    M = 256
    K = 4096
    N0 = 4096*2
    N1 = 4096
    N2 = 4096
    layers = 100
    np.random.seed(0)

    W0 = np.random.randint(low=-1, high=2, size=(N0, K)).astype(np.int8) # -1, 0, 1
    W1 = np.random.randint(low=-1, high=2, size=(N1, K)).astype(np.int8) # -1, 0, 1
    W2 = np.random.randint(low=-1, high=2, size=(N2, K)).astype(np.int8) # -1, 0, 1

    t0 = time.time()
    qkv_projs = [csrc.AMXQKVLinear(W0, W1, W2) for _ in range(layers)]
    print(f"AMXQKVLinear constructor {layers} layers tooks {(time.time() - t0)*1e3:.2f} ms")

    X = np.random.randint(low=-1, high=2, size=(M, K)).astype(np.int8) # -1, 0, 1

    Y0 = X.astype(np.int32) @ W0.transpose().astype(np.int32)
    Y1 = X.astype(np.int32) @ W1.transpose().astype(np.int32)
    Y2 = X.astype(np.int32) @ W2.transpose().astype(np.int32)

    Z0 = np.zeros([M, N0], dtype=np.int32)
    Z1 = np.zeros([M, N1], dtype=np.int32)
    Z2 = np.zeros([M, N2], dtype=np.int32)
    for r in range(5):
        t0 = time.time()
        for qkv in qkv_projs:
            qkv.forward(X, Z0, Z1, Z2)
        print(f"[{r}] : {((time.time() - t0)/layers)*1e3:.2f} ms")

    if not np.allclose(Y0, Z0):
        print(Y0)
        print(Z0)
        print(np.where(Y0 != Z0))
        assert False
    if not np.allclose(Y1, Z1):
        print(Y1)
        print(Z1)
        assert False
    if not np.allclose(Y2, Z2):
        print(Y2)
        print(Z2)
        assert False

test_amx_repack_B()
test()

#