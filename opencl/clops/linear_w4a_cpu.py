cpp_kernel_sources = r'''
#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <cstring>
union KArg {
    int64_t i;
    float f;
    void* p;
    float* pf32;
    uint8_t* pu8;
    int8_t* pi8;
    uint16_t* pu16;
};

#define CLAMP(x, a, b) (x)<(a)?(a):((x)>(b)?(b):(x))

// https://stackoverflow.com/a/21371401/9292588
#if USE_DEBUG
#define DEBUG0(...) std::cout << "===" << __LINE__ << ":" << std::endl;
#define DEBUG1(x) std::cout << "===" << __LINE__ << ":" << #x << "=" << x << std::endl;
#define DEBUG2(x1, x2) std::cout << "===" << __LINE__ << ":" << #x1 << "=" << x1 << "," << #x2 << "=" << x2 << std::endl;
#define DEBUG3(x1, x2, x3) std::cout << "===" << __LINE__ << ":" << #x1 << "=" << x1 << "," << #x2 << "=" << x2 << "," << #x3 << "=" << x3 << std::endl;
#define DEBUG4(x1, x2, x3, x4) std::cout << "===" << __LINE__ << ":" << #x1 << "=" << x1 << "," << #x2 << "=" << x2 << "," << #x3 << "=" << x3 << "," << #x4 << "=" << x4 << std::endl;
#define DEBUG5(x1, x2, x3, x4, x5) std::cout << "===" << __LINE__ << ":" << #x1 << "=" << x1 << "," << #x2 << "=" << x2 << "," << #x3 << "=" << x3 << "," << #x4 << "=" << x4 <<  "," << #x5 << "=" << x5 << std::endl;
#define DEBUG6(x1, x2, x3, x4, x5, x6) std::cout << "===" << __LINE__ << ":" << #x1 << "=" << x1 << "," << #x2 << "=" << x2 << "," << #x3 << "=" << x3 << "," << #x4 << "=" << x4 <<  "," << #x5 << "=" << x5 << "," << #x6 << "=" << x6 << std::endl;

#define GET_MACRO(_0, _1, _2, _3, _4, _5, _6, NAME, ...) NAME
#define DEBUG(...) GET_MACRO(_0 __VA_OPT__(,) __VA_ARGS__,  DEBUG6, DEBUG5, DEBUG4, DEBUG3, DEBUG2, DEBUG1, DEBUG0)(__VA_ARGS__)
#else
#define DEBUG(...)
#endif

template <typename T, typename Q>
inline void splitter(const T& n, const Q& team, const Q& tid, T& n_start, T& n_end) {
    if (team <= 1 || n == 0) {
        n_start = 0;
        n_end = n;
    } else {
        T n1 = (n + (T)team - 1) / (T)team;
        T n2 = n1 - 1;
        T T1 = n - n2 * (T)team;
        n_end = (T)tid < T1 ? n1 : n2;
        n_start = (T)tid <= T1 ? tid * n1 : T1 * n1 + ((T)tid - T1) * n2;
    }

    n_end += n_start;
}

extern "C" void Linear_quant_I4(int ithr, int nthr, std::vector<KArg>& kargs) {
    auto N = kargs[0].i;
    auto K = kargs[1].i;

    int64_t n0 = 0, n1 = N;
    splitter(N, nthr, ithr, n0, n1);

    DEBUG(N, K, GROUP_SIZE, kargs.size());
    DEBUG(n0, n1);
    
    for(int64_t n = n0; n < n1; n++) {
        float* src = kargs[2].pf32 + n*K;
        uint8_t* dst = kargs[3].pu8 + n*K/2;
        auto* scales = kargs[4].pf32 + n*(K/GROUP_SIZE);
        int8_t* zps = kargs[5].pi8 + n*(K/GROUP_SIZE);

        for(int64_t k = 0; k < K; k += GROUP_SIZE) {
            float vmin = src[k];
            float vmax = src[k];
            for(int64_t ki = 0; ki < GROUP_SIZE; ki++) {
                vmin = std::min(vmin, src[k + ki]);
                vmax = std::max(vmax, src[k + ki]);
            }

            // zero-point zp is choosen as interger
            //  vmin = (0 + zp)*s
            //  vmax = (15 + zp)*s
            //
            int8_t zp = std::roundf(15.0f * vmin/(vmax - vmin));
            float s = (vmax - vmin)/15.0f;
            float rs = 1.0f/s;

            *scales++ = s;
            *zps++ = zp;

            DEBUG(n, k, vmin, vmax, s, (int)zp);

            for(int64_t ki = 0; ki < GROUP_SIZE; ki +=2) {
                int8_t q0 = std::roundf(src[k + ki] * rs) - zp;
                int8_t q1 = std::roundf(src[k + ki + 1] * rs) - zp;
                uint8_t v0 = CLAMP(q0, 0, 15);
                uint8_t v1 = CLAMP(q1, 0, 15);

                auto rq0 = ((int)v0 + zp)*s;
                auto rq1 = ((int)v1 + zp)*s;

                DEBUG((int)v0, rq0);
                DEBUG((int)v1, rq1);
                *dst++ = (v1 << 4) | v0;
            }
        }
    }
    DEBUG();
}

extern "C" void Linear_woq_I4(int ithr, int nthr, std::vector<KArg>& kargs) {
    auto M = kargs[0].i;
    auto N = kargs[1].i;
    auto K = kargs[2].i;

    int64_t n0 = 0, n1 = N;
    splitter(N, nthr, ithr, n0, n1);

    //DEBUG(M, N, K);
    for(int64_t m = 0; m < M; m++) {
        for(int64_t n = n0; n < n1; n++) {
            float* src = kargs[3].pf32 + m*K;
            float* dst = kargs[4].pf32;
            uint8_t* wei = kargs[5].pu8 + n*(K/2);
            auto* scales = kargs[6].pf32 + n*(K/GROUP_SIZE);
            int8_t* zps = kargs[7].pi8 + n*(K/GROUP_SIZE);

            float sum = 0;
            for(int64_t k = 0; k < K; k += GROUP_SIZE) {
                auto s = *scales++;
                auto zp = *zps++;
                float part_sum = 0;
                //DEBUG(m, n, k, s, (int)zp);
                for(int64_t ki = 0; ki < GROUP_SIZE; ki +=2) {
                    uint8_t w = *wei++;
                    float w0 = (int)(w & 0xF) + zp;
                    float w1 = (int)(w >> 4) + zp;
                    //DEBUG((int)w, w0*s, w1*s);
                    part_sum += w0 * src[k + ki];
                    part_sum += w1 * src[k + ki + 1];
                }
                sum += part_sum * s;
            }
            dst[m*N + n] = sum;
        }
    }
    //DEBUG();
}
'''
from . import cl
import numpy as np
from .utils import *


GROUP_SIZE = 128
USE_DEBUG=0
cpp_kernels = cl.cpp_kernels(cpp_kernel_sources, f"-DGROUP_SIZE={GROUP_SIZE} -DUSE_DEBUG={USE_DEBUG}")

# weight-only-quantization to INT4
class Linear_w4a_cpu:
    # weight: [N, K], 
    def __init__(self, weight, bias=None):
        assert(bias is None)

        # quantize weight into groupped INT4 format:(sym)
        N, K = weight.shape
        assert((K % GROUP_SIZE) == 0)   # 2 element packed into a INT8

        self.weight_i4 = np.ndarray([N, K//2], np.uint8)   # two int4 packed into a uint8
        self.scales = np.ndarray([N, K//GROUP_SIZE], np.float32) # scales
        self.zps = np.ndarray([N, K//GROUP_SIZE], np.int8)       # zero-points
        self.N = N
        self.K = K

        weight_src = weight.float().detach().numpy()
        cpp_kernels.call("Linear_quant_I4", N, K, weight_src, self.weight_i4, self.scales, self.zps)
        #print("==============") print(weight.transpose(1,0)[:8,:8])
        #print("==============") print(np.where(np.isnan(self.weight_i4.numpy())))

        #print("weight_src=\n", weight_src)
        #print("weight_i4=\n", self.weight_i4)
        #print("scales=", self.scales)
        #print("zps=", self.zps)

    def __call__(self, input):
        # shape inference
        i_shape = input.shape
        o_shape = list(i_shape)
        o_shape[-1] = self.N
        
        output = np.ndarray(o_shape, np.float32)

        M = input.numel // self.K
        #print(M, self.K, self.N,  input.shape, self.weight.shape, output.shape)

        src = input.numpy().astype(np.float32)

        cpp_kernels.call("Linear_woq_I4", M, self.N, self.K, src, output, self.weight_i4, self.scales, self.zps)

        return cl.tensor(output.astype(np.float16))


if __name__ == "__main__":
    def test_acc(shape, Bcnt = 0):
        M, K, N = shape

        RA = 1
        RB = 1
        torch.manual_seed(0)
        A = torch.randint(low=-RA, high=RA+1, size=[M, K], dtype=torch.half)
        B = torch.randint(low=-RB, high=RB+1, size=[N, K], dtype=torch.half)/RB/2

        #A[:, 0] *= 5
        ref = torch.matmul(A, B.transpose(1,0)).numpy()

        Bsize = K*N*2
        if (Bcnt <= 0): Bcnt = int(500e6)//(Bsize)
        linears = [Linear_w4a_cpu(B) for _ in range(Bcnt)]

        input = to_cl(A)
        for l in linears:
            output = l(input)

        res = output.numpy()
        print("==============")
        print(A)
        print("==============")
        print(ref)
        print("==============")
        print(res)
        compare(ref, res, atol=0.1, rtol=0.1)

    #test_acc([1, GROUP_SIZE, 2], 1)
    #import sys;    sys.exit(0)

    for b in [7, 1]:
        test_acc([b, 4096, 12288], 10)
        test_acc([b, 4096, 4096], 10)
        test_acc([b, 4096, 11008], 10)
        test_acc([b, 11008, 4096], 10)
        test_acc([b, 4096, 32000], 10)

    import sys
    sys.exit(0)

