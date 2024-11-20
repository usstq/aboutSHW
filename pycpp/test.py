import pycpp
import torch
import time
src = r'''
#include "common.hpp"

{
    int nthr = omp_get_max_threads();
    #pragma omp parallel
    {
        int ithr = omp_get_thread_num();
        int64_t n0 = 0, n1 = N;
        splitter(N, nthr, ithr, n0, n1);
    }
}

extern "C" void Linear_quant_I4(int ithr, int nthr, std::vector<KArg>& kargs) {
    auto N = kargs[0].i;
    auto K = kargs[1].i;
}
'''
#kernel = pycpp.kernels(src)

print(dir(pycpp))
M = 400
K = 4096
N = 4800

print(torch.backends.mkldnn.is_available())
print(*torch.__config__.show().split("\n"), sep="\n")


from torch.utils import mkldnn as mkldnn_utils;
with torch.no_grad():
    device = torch.device("cpu")
    A = torch.randint(-8, 8, [M, K]).float().to(device)
    B = torch.randint(-8, 8, [K, N]).float().to(device)
    C = torch.randint(-8, 8, [M, N]).float().to(device)

    REPEAT = 3
    LAYERS = 30

    net = []
    for i in range(LAYERS):
        node = torch.nn.Linear(K, N)
        #node = mkldnn_utils.to_mkldnn(node)
        net.append(node)
   
    #batch = A.to_mkldnn()
    batch = A
    print("============== torch.nn.Linear ============ ")
    for r in range(REPEAT):
        t0 = time.time()
        for n in net:
            n(batch)
        t1 = time.time()
        print(f"{(t1-t0)/len(net)*1e3:.3f} ms/linear")


    print("============== torch.matmul ============ ")
    net = []
    for i in range(LAYERS):
        w = torch.randint(-8, 8, [K, N]).float().to(device)
        net.append(w)

    for r in range(REPEAT):
        t0 = time.time()
        for wei in net:
            torch.matmul(A, wei)
        t1 = time.time()
        print(f"{(t1-t0)/len(net)*1e3:.3f} ms/linear")

    print("============== my gemm ============ ")
    src = A.numpy()
    dst = C.numpy()
    for r in range(REPEAT):
        t0 = time.time()
        for wei in net:
            pycpp.gemm(src, wei.numpy(), dst)
        t1 = time.time()
        print(f"{(t1-t0)/len(net)*1e3:.3f} ms/linear")
        
    ref = torch.matmul(A, B)


#pycpp.gemm()