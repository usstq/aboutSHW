from . import cl
import numpy as np
from .utils import *
import clops

MAX_KV_LEN = 256*8

TARGET_SEQ_LEN_BLOCK_SIZE=16
SG_SCALE_FACTOR=1
SUBGROUP_SIZE=16
Hq=28
Hk=7
HEAD_SIZE=128

B=1
Lq=8416
Lk=8416

assert(Hq % Hk == 0);     # implied
assert(HEAD_SIZE % SUBGROUP_SIZE == 0);     # implied
assert(TARGET_SEQ_LEN_BLOCK_SIZE == SUBGROUP_SIZE);   # implied

class SDPA_opt:
    def __init__(self, multi_query : bool = False):
        self.multi_query = multi_query
        
        if self.multi_query:
            cl_source_file = "cl_kernels/sdpa_multiquery.cl"
        else:
            cl_source_file = "cl_kernels/sdpa.cl"

        with open(cl_source_file, "r") as file:
            # Read the entire file content into a string
            cl_kernel_sources = file.read()
        print(cl_kernel_sources[:100])
        self.cl_kernels = kernel_cache(cl_kernel_sources)

    def __call__(self, shape_info_input, query_input, key_input, value_input, attn_mask_input, scale_input):
        # shape inference       
        exp_sums = to_cl(np.zeros([4], dtype=np.float32))
        max_logits = to_cl(np.zeros([4], dtype=np.float32))
        tmp_out = to_cl(np.zeros([2], dtype=np.float16))

        output = to_cl(torch.zeros(B, Hq, Lq, HEAD_SIZE))

        if self.multi_query:
            GWS = [B*Hq, int(Lq/TARGET_SEQ_LEN_BLOCK_SIZE), HEAD_SIZE*SG_SCALE_FACTOR]
            LWS = [Hq//Hk, 1, HEAD_SIZE*SG_SCALE_FACTOR]
        else:
            GWS = [B*Hq, int(Lq/TARGET_SEQ_LEN_BLOCK_SIZE), HEAD_SIZE*SG_SCALE_FACTOR]
            LWS = [1, 1, HEAD_SIZE*SG_SCALE_FACTOR]            

        print(f"GWS={GWS}, LWS={LWS}")
        self.cl_kernels.enqueue("sdpa_opt_multi_tokens_6761455398808095608_0_0__sa", GWS, LWS,
                            shape_info_input, query_input, key_input, value_input, attn_mask_input, scale_input, output,
                            exp_sums, max_logits, tmp_out)

        return output

if __name__ == "__main__":
    import sys
    cl.profiling(True)

    #qkv [B, Lq, (Hq + Hk + Hv) * S)], attn_mask [B, Lk] -> output [B, Hq, Lq, S]
    def MHA_torch_impl(qkv : torch.Tensor, attention_mask : torch.Tensor, scale : torch.Tensor):
        ref_mha = clops.MHA_cpu(Hq, Hk, HEAD_SIZE, MAX_KV_LEN)               
        output = ref_mha(qkv, attention_mask)  # B, Lq, Hq*S
        output = to_torch(output).view(B, Lq, Hq, HEAD_SIZE).transpose(1, 2)
        return output.numpy()
    
    def MHA_cl_impl(qkv : torch.Tensor, attention_mask : torch.Tensor, scale : torch.Tensor):
        mha_gpu = clops.MHA(Hq, Hk, HEAD_SIZE, MAX_KV_LEN, False, kv_block=32)
        output = mha_gpu(to_cl(qkv), attention_mask)
        output = to_torch(output).view(B, Lq, Hq, HEAD_SIZE).transpose(1, 2)
        return output.numpy()
    
    def opt_impl(qkv : torch.Tensor, attention_mask : torch.Tensor, scale : torch.Tensor, multi_query : bool = False):
        bsz, q_len, _ = qkv.size()
        q_size = Hq * HEAD_SIZE
        kv_size = Hk * HEAD_SIZE
        query = qkv[:,:,:q_size].view(bsz, q_len, Hq, HEAD_SIZE).transpose(1, 2).contiguous()
        key = qkv[:,:,q_size:q_size+kv_size].view(bsz, q_len, Hk, HEAD_SIZE).transpose(1, 2).contiguous()
        value = qkv[:,:,q_size+kv_size:].view(bsz, q_len, Hk, HEAD_SIZE).transpose(1, 2).contiguous()
        attn_mask = torch.broadcast_to(attention_mask, [B, Lq, Lk])[:, None, :, :].contiguous()
        print("shapes : ", query.shape, key.shape, value.shape)

        query_input = to_cl(query)
        key_input = to_cl(key)
        value_input = to_cl(value)

        scale_input = to_cl(scale)
        attn_mask_input = to_cl(attn_mask)

        sdpa = SDPA_opt(multi_query)
        shape_info = [
            # // input0 query
            B, Hq, 1, 1, 1, 1, Lq, HEAD_SIZE,
            # // input1 key
            B, Hk, 1, 1, 1, 1, Lk, HEAD_SIZE, 0, 0,
            # // input2 value
            B, Hk, 1, 1, 1, 1, Lk, HEAD_SIZE, 0, 0,
            #  input3 attn_mask
            1, 1, 1, 1, 1, 1, Lq, Lk,
            #  input4 scale
            #  output
            B, Hq, 1, 1, 1, 1, Lq, HEAD_SIZE
        ]
        print(f"len(shape_info)={len(shape_info)}, shape_info={shape_info}")
        shape_info_input = to_cl(torch.tensor(shape_info).int())
        output = sdpa(shape_info_input, query_input, key_input, value_input, attn_mask_input, scale_input)

        duration = cl.finish()
        return output.numpy(), duration

    def test_acc(Bcnt = 0):              
        # reference torch impl
        qkv = torch.randn([B, Lk, (Hq + Hk + Hk) * HEAD_SIZE], dtype=torch.float16)
        attention_mask = torch.randn([B, Lk], dtype=torch.float16)
        scale = torch.ones([1], dtype=torch.float16)
        
        # ref = MHA_cl_impl(qkv, attention_mask, scale)
        ref, durs = opt_impl(qkv, attention_mask, scale, False)
        for ns in durs:
            print(f'{ref.shape=}, {ns*1e-6:.3f} ms')
        
        # cl impl
        opt, durs = opt_impl(qkv, attention_mask, scale, True)
        # opt = MHA_cl_impl(qkv, attention_mask, scale)
        for ns in durs:
            print(f'{opt.shape=}, {ns*1e-6:.3f} ms')

        try:
            if not np.allclose(ref, opt, atol=0.01, rtol=0.01, equal_nan=True):
                # print(f'{ref=}\n{opt=}')
                pos = np.where(np.abs(ref - opt) > 0.01)
                print(f'failed at shape = {opt.shape}')
                # print(f'pos = {pos}')
                # print(f'ref_val = {ref[pos]}\nopt_val={opt[pos]}')
                raise Exception("failed.")
            print('done.')
        except Exception as inst:
            print('failed.')

    test_acc(); sys.exit(0)
