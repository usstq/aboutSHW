cl_kernel_sources = r'''

// global_id : [Hq + Hk, q_len, batch_size]
//
// inv_freq : [half_rotary_dim]
//
//    input : [batch_size, q_len, (Hq + Hk + Hv)*S)]
__kernel void ROPE_ref(__global half * input,
                    __global float * inv_freq,
                    int half_rotary_dim,
                    int batch_size,
                    int q_len,
                    int head_cnt_qkv,
                    int head_size,
                    int pos0) {
    int h = get_global_id(0);
    int q_offset = get_global_id(1);
    int batch = get_global_id(2);
    float position_idx = pos0 + q_offset;

    __global half * src = input + ((batch * q_len + q_offset) * head_cnt_qkv +  h) * head_size;

    for(int i0=0; i0 < half_rotary_dim; i0++) {
        int i1 = i0 + half_rotary_dim;
        float xita = position_idx * inv_freq[i0];
        float cx = cos(xita);
        float sx = sin(xita);
        float x0 = src[i0];
        float x1 = src[i1];
        src[i0] = cx * x0 - sx * x1;
        src[i1] = sx * x0 + cx * x1;
    }
}

// global_id : [Hq + Hk, q_len, batch_size]
//
// inv_freq : [half_rotary_dim]
//
//    input : [batch_size, q_len, (Hq + Hk + Hv)*S)]

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void ROPE(__global half * input,
                    __global float * inv_freq,
                    int half_rotary_dim,
                    int batch_size,
                    int q_len,
                    int head_cnt_qkv,
                    int head_cnt_qk,
                    int head_size,
                    int pos0) {
    
    int i = get_global_id(0);
    int dim = get_sub_group_id();
    int q_offset = get_global_id(1);
    int batch = get_global_id(2);
    int position_idx = pos0 + q_offset;

    __global half * src = input + ((batch * q_len + q_offset) * head_cnt_qkv) * head_size + dim*16;

    float xita = (float)position_idx * inv_freq[i];
    half cx = cos(xita);
    half sx = sin(xita);

    for(int h=0; h < head_cnt_qk; h++) {
        half x0 = as_half(intel_sub_group_block_read_us((const __global ushort*)(src)));
        half x1 = as_half(intel_sub_group_block_read_us((const __global ushort*)(src + half_rotary_dim)));

        half y0 = cx * x0 - sx * x1;
        half y1 = sx * x0 + cx * x1;
        intel_sub_group_block_write_us((__global ushort*)(src), as_ushort(y0));
        intel_sub_group_block_write_us((__global ushort*)(src + half_rotary_dim), as_ushort(y1));
        src += head_size;
    }
}

'''

from . import cl
import numpy as np
from .utils import *

class ROPE:
    def __init__(self, inv_freq, rotary_dim, head_cnt_q, head_cnt_k, head_size):
        self.inv_freq = to_cl(inv_freq)
        self.half_rotary_dim = rotary_dim//2
        self.head_cnt_q = head_cnt_q
        self.head_cnt_k = head_cnt_k
        self.head_cnt_qkv = head_cnt_q + 2*head_cnt_k
        self.head_size = head_size
        self.kernels = kernel_cache(cl_kernel_sources, "", "./dump/")
        self.position_length = 0
    
    def __call__(self, qkv, position_id_base):
        '''
         no shape change, inplace on VRAM
         input : [batch_size, q_len, (Hq + Hk + Hv) * head_size)]
        '''
        batch_size, q_len, S = qkv.shape
        assert S == (self.head_cnt_qkv * self.head_size)

        if 1:
            self.kernels.enqueue("ROPE",
                                [self.half_rotary_dim, q_len, batch_size],
                                [self.half_rotary_dim, 1, 1],
                                qkv,
                                self.inv_freq,
                                self.half_rotary_dim,
                                batch_size,
                                q_len,
                                self.head_cnt_qkv,
                                self.head_cnt_q + self.head_cnt_k,
                                self.head_size,
                                position_id_base)
        else:
            self.kernels.enqueue("ROPE_ref",
                                [self.head_cnt_q + self.head_cnt_k, q_len, batch_size],
                                [1, 1],
                                qkv,
                                self.inv_freq,
                                self.half_rotary_dim,
                                batch_size,
                                q_len,
                                self.head_cnt_qkv,
                                self.head_size,
                                position_id_base)
        return qkv
        

if __name__ == "__main__":
    cl.profiling(True)
    # batch_size, max_kv_len = 16, 1024 
    # qkv=[16, 1024, 1152] float16  position_id_base=0
    inv_freq = torch.tensor([1.0000e+00, 6.4938e-01, 4.2170e-01, 2.7384e-01, 1.7783e-01, 1.1548e-01,
        7.4989e-02, 4.8697e-02, 3.1623e-02, 2.0535e-02, 1.3335e-02, 8.6596e-03,
        5.6234e-03, 3.6517e-03, 2.3714e-03, 1.5399e-03, 1.0000e-03, 6.4938e-04,
        4.2170e-04, 2.7384e-04, 1.7783e-04, 1.1548e-04, 7.4989e-05, 4.8697e-05,
        3.1623e-05, 2.0535e-05, 1.3335e-05, 8.6596e-06, 5.6234e-06, 3.6517e-06,
        2.3714e-06, 1.5399e-06], dtype=torch.float32)
    
    rotary_dim=64
    head_cnt_q=14
    head_cnt_k=2
    head_size=64
    
    qkv = to_cl(torch.ones(16, 1024, 1152, dtype=torch.float16))

    rope = ROPE(inv_freq, rotary_dim, head_cnt_q, head_cnt_k, head_size)
    
    for i in range(10):
        qkv = rope(qkv, i)

    Bsize = qkv.numel * 2
    durs = cl.finish()
    for ns in durs:
        print(f"{Bsize*1e-6:.3f} MB {ns*1e-6:.3f} ms, BW: { Bsize/ns : .2f} GB/s")
