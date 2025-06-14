from . import cl
import numpy as np
from .utils import *

cl_kernel_source = r'''
//#
//# GWS/LWS: [K//QUANT_GROUP_SIZE, N//2], [1, 1]
struct quant_param {
    half scale;
    char zp;  
};

struct quant_param quantize_i4_group(__global half * src, __global uchar * dst, bool with_zp)
{
    half vmin = src[0];
    half vmax = src[0];
    for(int i = 1; i < QUANT_GROUP_SIZE; i++) {
        vmin = min(src[i], vmin);
        vmax = max(src[i], vmax);
    }

    char zp;
    half s;
    if (with_zp) {
        // zero-point zp is choosen as interger
        //   vmax =  (7 + zp)*s
        //   vmin = (-8 + zp)*s
        if (vmax == vmin) {
            zp = 0;
            s = fabs(vmax);
        } else {
            // with limitation that zp is also int4, the scale must :
            //    1. shrink the max(abs(vmax), abs(vmin)) to be within [-8-8, 7+7]
            //    2. shrink the (vmax - vmin) to be <= 15
            s = (vmax - vmin)/15.5f;

            // increase s to fit rules 1
            half absmax = fabs(vmax);
            half absmin = fabs(vmin);
            if (absmax > absmin) {
                // assert(vmax > 0)
                if (vmax > s*14.0f) s = vmax/14.0f;
                zp = 7 - vmax/s;
            } else {
                // assert(vmin < 0)
                if (vmin < s*(-16.0f)) s = vmin/(-16.0f);
                zp = -8 - vmin/s;
            }
        }
    } else {
        // no zero-point
        zp = 0;

        // scales can be negative number, so we can always map the max to -8
        if (vmin >= 0) {
            // both >=0, map -8 to vmax
            s = -vmax / 8.0f;
        } else if (vmax <= 0) {
            // both <0, map -8 to vmin
            s = -vmin / 8.0f;
        } else {
            // vmax > 0 & vmin < 0
            if (fabs(vmin) >= fabs(vmax)) {
                //maps vmin to -8 (using positive scales)
                s = -vmin / 8.0f;
            } else {
                //maps vmax to -8 (using negative scales)
                s = -vmax / 8.0f;
            }
        }
    }
    zp = clamp(zp, (char)-8, (char)7);

    half rs = 1.0f/s;
    for(int i = 0; i < QUANT_GROUP_SIZE; i += 2) {
        uchar v0 = as_uchar(clamp((char)round(src[i] * rs + zp), (char)-8, (char)7));
        uchar v1 = as_uchar(clamp((char)round(src[i + 1] * rs + zp), (char)-8, (char)7));

        *dst = (v1 << 4) | (v0 & 0xF);
        dst++;

        //# update src as fake-quanted value
        src[i] = (as_char(v0) - zp) * s;
        src[i+1] = (as_char(v1) - zp) * s;
    }
    struct quant_param ret = {s, zp};
    return ret;
}

__kernel void quant_I4(__global half * _src,
                       __global uchar * _dst,
                       __global half * _scales,
                       __global char * _zps,
                       int N, int K) {
    int kg = get_global_id(0);
    int ng = get_global_id(1);
    int k = kg * QUANT_GROUP_SIZE;
    int n = ng * 2;

    bool with_zp = (_zps != NULL);
    struct quant_param p0 = quantize_i4_group(_src + (n + 0) * K + k, _dst +  (n + 0) * (K/2) + (k/2), with_zp);
    struct quant_param p1 = quantize_i4_group(_src + (n + 1) * K + k, _dst +  (n + 1) * (K/2) + (k/2), with_zp);

    _scales[kg*N + n + 0] = p0.scale;
    _scales[kg*N + n + 1] = p1.scale;
    if (with_zp) {
        _zps[kg*N + ng] = (p1.zp << 4) | ( p0.zp & 0xF);
    }
}
'''


# quantize :
#  weight half [N, K] => weight i4 [N, K] / scales half [K_groups, N] / zp i4 [K_groups, N]
# 
def unpack_i4(ti4):
    s = ti4.shape
    row_size = s[-1]
    batch_size = ti4.size // row_size
    
    ti4 = ti4.reshape([batch_size, row_size])
    ti8 = np.zeros([batch_size, row_size * 2], dtype=np.int8)
    for b in range(batch_size):
        for i in range(row_size):
            ti8[b, 2*i + 0] = (ti4[b, i] << 4) >> 4
            ti8[b, 2*i + 1] = (ti4[b, i] >> 4)
    return ti8

def quantize_weight_to_i4(weight, QUANT_GROUP_SIZE, with_zero_point):
    N, K = weight.shape
    assert (K % QUANT_GROUP_SIZE) == 0
    assert (N % 2) == 0

    weight_raw = weight.half().detach().numpy()
    weight_half = cl.tensor(weight_raw)
    weight_i4 = cl.tensor([N, K//2], np.dtype(np.int8))
    # scales/zp are stored so a cache-line can be shared by a quantization-group
    scales = cl.tensor([K//QUANT_GROUP_SIZE, N], np.dtype(np.float16))
    zps = cl.tensor([K//QUANT_GROUP_SIZE, N//2], np.dtype(np.int8))
    cl_kernels = kernel_cache(cl_kernel_source, options=(f"-D{QUANT_GROUP_SIZE=}"))
    cl_kernels.enqueue("quant_I4", [K//QUANT_GROUP_SIZE, N//2], [1, 1],
                            weight_half,
                            weight_i4,
                            scales,
                            zps if with_zero_point else None,
                            N, K)
    if 0:
        print("========== wi4\n", weight_i4.numpy())
        wi8 = unpack_i4(weight_i4.numpy()).reshape(N, K//QUANT_GROUP_SIZE, QUANT_GROUP_SIZE)
        s = scales.numpy().transpose().reshape(N, K//QUANT_GROUP_SIZE, 1)
        z = unpack_i4(zps.numpy()).transpose().reshape(N, K//QUANT_GROUP_SIZE, 1)
        deq = ((wi8.astype(np.float16) - z.astype(np.float16)) * s.astype(np.float16)).reshape(N, K)
        print("========== weight_raw")
        print(weight_raw)
        print("========== weight_dequantized")
        print(weight_half.numpy())
        print("========== wi8")
        print(wi8)
        print("========== s")
        print(s)
        print("========== z")
        print(z)
        print(deq)
        print("========== abs(diff): max, mean")
        diff = np.abs(deq - weight_raw)
        print(diff.max(), diff.mean())
        
    return weight_i4, scales, zps, weight_half.numpy()

# creator function cache act as kernel factory cache
from functools import cache, lru_cache
@cache
def create_onednn_matmul(a_dtype, w_dtype, bias_dtype, M, K, N,
                         wei_quant_k_group_size,
                         act_quant_k_group_size,
                         pos_ops_name):
    mm = cl.onednn_matmul(a_dtype, w_dtype, bias_dtype, M, K, N, -1)
    if wei_quant_k_group_size:
        mm.w_scale(wei_quant_k_group_size)
        if w_dtype == cl.onednn_dtype.u4:
            mm.w_zp(wei_quant_k_group_size)
    if act_quant_k_group_size:
        mm.a_scale(act_quant_k_group_size)
    if pos_ops_name == "silu_binmul":
        mm.post_op_silu()
        mm.post_op_bin_mul(False)
    mm.fpmath_f16()
    mm.create()
    return mm

cl_dyn_quant_src = r'''
__kernel void quant_I8_per_row(__global half * _src,
                                __global char * _dst,
                                __global half * scales,
                                int N) {
    int n = get_global_id(0);
    __global half * src = _src + n * K;
    __global char * dst = _dst + n * K;

    half absmax = 0;
    for(int k = 0; k < K; k++) {
        absmax = fmax(absmax, fabs(src[k]));
    }
    half s = 127.0f/absmax;
    for(int k = 0; k < K; k++) {
        dst[k] = src[k] * s;
    }
    scales[n] = absmax/127.0f;
}
'''

class Linear_onednn:
    def __init__(self, weight, bias = None, w_dtype = cl.onednn_dtype.s4, dq_per_token = False):
        self.linears = {}

        self.N, self.K = weight.shape # weight: [N, K]
        self.weight = to_cl(weight.half())
        self.bias_dtype = cl.onednn_dtype.f16 if bias is not None else cl.onednn_dtype.undef
        self.bias = to_cl(bias.half()) if bias is not None else cl.tensor()
        self.post_ops_name = ""
        self.src_quant_group_size = 0
        self.wei_quant_group_size = 0
        self.with_zero_point = False
        self.w_dtype = w_dtype
        self.scales = cl.tensor()
        self.zps = cl.tensor()
        if self.w_dtype == cl.onednn_dtype.f16:
            pass
        elif self.w_dtype == cl.onednn_dtype.s4:
            self.wei_quant_group_size = 128
            self.with_zero_point = False
            self.weight, self.scales, self.zps, self.wei_deq = quantize_weight_to_i4(weight, self.wei_quant_group_size, self.with_zero_point)                
        else:
            assert 0, f"unsuuported weight-quantization dtype {self.w_dtype}" 

        self.dq_per_token = dq_per_token
        if self.dq_per_token:
            self.src_quant_group_size = self.K
        self.cl_kernels = kernel_cache(cl_dyn_quant_src, options=(f"-DK={self.K}"))

        if not self.with_zero_point:
            self.zps = cl.tensor()

    def __call__(self, input):
        # shape inference
        i_shape = input.shape
        o_shape = list(i_shape)
        o_shape[-1] = self.N
        output = cl.tensor(o_shape, input.dtype)

        M = input.numel // self.K
        if M not in self.linears:
            # there is a internal onednn kernel cache inside cl
            self.linears[M] = create_onednn_matmul(cl.onednn_dtype.s8 if self.dq_per_token else cl.onednn_dtype.f16,
                                                   self.w_dtype, self.bias_dtype, M, self.K, self.N,
                                                   self.wei_quant_group_size,
                                                   self.src_quant_group_size,
                                                   self.post_ops_name)
        if self.dq_per_token:
            input_i8 = cl.tensor(input.shape, np.dtype(np.int8))
            input_sc = cl.tensor([M], np.dtype(np.float16))
            self.cl_kernels.enqueue("quant_I8_per_row", [M], [1], input, input_i8, input_sc, M)
            src = (input_i8, input_sc)
        else:
            src = (input,)

        self.linears[M].forward(output, src, [self.weight, self.scales, self.zps], self.bias, cl.tensor())
        return output

if __name__ == "__main__":
    cl.profiling(True)
    if 0:
        torch.manual_seed(0)
        A = torch.randint(-8, 8, [2, 128]).half() / 8
        A[0, 0] = A[0, 1]
        A = -A
        wi4, s, zp = quantize_weight_to_i4(A, 128)

        assert 0
    
    def test_acc(shape, Bcnt = 0):
        M, K, N = shape
        A = torch.randint(-8, 8, [M, K]).half()/8
        B = torch.randint(-8, 8, [N, K]).half()/8
        bias = torch.randint(-8, 8, [1, N]).half()/8
        ref = (torch.matmul(A, B.transpose(1,0)) + bias).numpy()

        Bsize = K*N*2 + M*K*2 + M*N*2
        if (Bcnt <= 0): Bcnt = int(500e6)//(Bsize)
        linears = [Linear_onednn(B, bias) for _ in range(Bcnt)]

        ref2 = (torch.matmul(A, torch.from_numpy(linears[0].wei_deq).transpose(1,0)) + bias).numpy()

        input = to_cl(A)
        for l in linears:
            output = l(input)

        durs = cl.finish()
        '''
        for ns in durs:
            print(f"{shape} {Bsize*1e-6:.3f} MB {ns*1e-6:.3f} ms, BW: { Bsize/ns : .2f} GB/s")
        if (len(durs)):
            mean_ns = sum(durs)/len(durs)
            print(f"  {mean_ns*1e-6: .3f} ms   BW: { K*N*2/mean_ns : .2f} GB/s")
        '''
        res = output.numpy()
        print(">>>>>>>>>>>>>> compare with ref2")
        compare(ref2, res)
        #print(">>>>>>>>>>>>>> compare with ref")
        #compare(ref, res)

    #test_acc([1, 128*4, 4096], 1)
    test_acc([24, 3584, 4608], 1)
