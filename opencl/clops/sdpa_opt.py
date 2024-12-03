from . import cl
import math
from .utils import *
import clops

class SDPA_opt:
    def __init__(self, Hq, Hk, HEAD_SIZE, is_optimized):
        self.is_optimized = is_optimized
        
        self.SG_SCALE_FACTOR=1
        self.SUBGROUP_SIZE=16
        SEQ_LEN_PARTITION_SIZE=(HEAD_SIZE*self.SG_SCALE_FACTOR)
        self.TARGET_SEQ_LEN_BLOCK_SIZE=16
        
        assert(Hq % Hk == 0);     # implied
        assert(HEAD_SIZE % self.SUBGROUP_SIZE == 0);     # implied
        assert(self.TARGET_SEQ_LEN_BLOCK_SIZE == self.SUBGROUP_SIZE);   # implied
        
        if self.is_optimized:
            cl_source_file = "cl_kernels/sdpa_opt.cl"
            self.kernel_name = 'sdpa_opt_multi_tokens'
        else:
            cl_source_file = "cl_kernels/sdpa.cl"
            self.kernel_name = 'sdpa_opt_multi_tokens_6761455398808095608_0_0__sa'

        with open(cl_source_file, "r") as file:
            # Read the entire file content into a string
            cl_kernel_sources = file.read()
        # print(cl_kernel_sources[:100])
        options = f'-DHEAD_SIZE={HEAD_SIZE} -DNUM_HEADS={Hq} -DNUM_KV_HEADS={Hk} \
                    -DSG_SCALE_FACTOR={self.SG_SCALE_FACTOR} -DSEQ_LEN_PARTITION_SIZE={SEQ_LEN_PARTITION_SIZE} -cl-mad-enable'
        self.cl_kernels = kernel_cache(cl_kernel_sources, options)

    def __call__(self, shape_info_input, query_input, key_input, value_input, attn_mask_input, scale_input):
        B, L, Hq, Hk, HEAD_SIZE = shape_info_input[0], shape_info_input[6], shape_info_input[1], shape_info_input[9], shape_info_input[7]

        shape_info_input = to_cl(torch.tensor(shape_info_input).int())
    
        # shape inference       
        exp_sums = to_cl(np.zeros([4], dtype=np.float32))
        max_logits = to_cl(np.zeros([4], dtype=np.float32))
        tmp_out = to_cl(np.zeros([2], dtype=np.float16))

        output = to_cl(torch.zeros(B, Hq, L, HEAD_SIZE, dtype=torch.float16))
        # output = cl.tensor([B, Hq, L, HEAD_SIZE], query_input.dtype)

        # if self.is_optimized:
        #     GWS = [HEAD_SIZE*self.SG_SCALE_FACTOR, B*Hq, int(L/self.TARGET_SEQ_LEN_BLOCK_SIZE)]
        #     LWS = [HEAD_SIZE*self.SG_SCALE_FACTOR, Hq//Hk, 1]
        # else:
        GWS = [B*Hq, math.ceil(L/self.TARGET_SEQ_LEN_BLOCK_SIZE), HEAD_SIZE*self.SG_SCALE_FACTOR]
        LWS = [1, 1, HEAD_SIZE*self.SG_SCALE_FACTOR]            

        print(f"GWS={GWS}, LWS={LWS}")
        print(self.cl_kernels.info(self.kernel_name, LWS, self.SUBGROUP_SIZE))

        self.cl_kernels.enqueue(self.kernel_name, GWS, LWS,
                            shape_info_input, query_input, key_input, value_input, attn_mask_input, scale_input, output, exp_sums, max_logits, tmp_out)

        return output

if __name__ == "__main__":
    import sys
    import numpy as np
    import math
    cl.profiling(True)
    np.set_printoptions(precision=3, suppress=True)

    MAX_KV_LEN = 1024*9
    #qkv [B, L, (Hq + Hk + Hv) * S)], attn_mask [B, L] -> output [B, Hq, L, S]
    def MHA_torch_impl(qkv : torch.Tensor, attention_mask : torch.Tensor, scale : torch.Tensor):
        B, L, _ = qkv.size()
        ref_mha = clops.MHA_cpu(Hq, Hk, HEAD_SIZE, MAX_KV_LEN)               
        output = ref_mha(qkv, attention_mask)  # B, L, Hq*S
        output = to_torch(output).view(B, L, Hq, HEAD_SIZE).transpose(1, 2)
        return output.numpy()
    
    def MHA_cl_impl(qkv : torch.Tensor, attention_mask : torch.Tensor, scale : torch.Tensor, Hq, Hk, HEAD_SIZE):
        B, L, _ = qkv.size()
        mha_gpu = clops.MHA(Hq, Hk, HEAD_SIZE, MAX_KV_LEN, False, kv_block=32)
        output = mha_gpu(to_cl(qkv), attention_mask)
        output = to_torch(output).view(B, L, Hq, HEAD_SIZE).transpose(1, 2)
        durations = cl.finish()
        return output.numpy(), durations
    
    def sdpa_impl(q : torch.Tensor, k : torch.Tensor, v : torch.Tensor, attention_mask : torch.Tensor, scale : torch.Tensor, Hq, Hk, HEAD_SIZE, is_optimized):
        print(f'{q.size()=}\n{k.size()=}\n{v.size()=}')
        B, L, _, _ = q.size()
        query = q.view(B, L, Hq, HEAD_SIZE).transpose(1, 2).contiguous()
        key = k.view(B, L, Hk, HEAD_SIZE).transpose(1, 2).contiguous()
        value = v.view(B, L, Hk, HEAD_SIZE).transpose(1, 2).contiguous()

        attn_mask = torch.broadcast_to(attention_mask, [B, L, L])[:, None, :, :].contiguous()
        # attn_mask = torch.tril(b, diagonal=1)
        # print("shapes : ", query.shape, key.shape, value.shape)

        # print(f'{query=}\n{key=}\n{value=}')

        query_input = to_cl(query)
        key_input = to_cl(key)
        value_input = to_cl(value)

        scale_input = to_cl(scale)
        attn_mask_input = to_cl(attn_mask)

        shape_info = [
            # // input0 query
            B, Hq, 1, 1, 1, 1, L, HEAD_SIZE,
            # // input1 key
            B, Hk, 1, 1, 1, 1, L, HEAD_SIZE, 0, 0,
            # // input2 value
            B, Hk, 1, 1, 1, 1, L, HEAD_SIZE, 0, 0,
            #  input3 attn_mask
            1, 1, 1, 1, 1, 1, L, L,
            #  input4 scale
            #  output
            B, Hq, 1, 1, 1, 1, L, HEAD_SIZE
        ]
        # print(f"len(shape_info)={len(shape_info)}, shape_info={shape_info}")

        sdpa = SDPA_opt(Hq, Hk, HEAD_SIZE, is_optimized)
        output = sdpa(shape_info, query_input, key_input, value_input, attn_mask_input, scale_input)

        output = to_torch(output)
        durations = cl.finish()
        return output.numpy(), durations

    def test_acc(B, Hq, Hk, HEAD_SIZE, L, use_randn = True):
        # reference torch impl
        # qkv = torch.randn([B, L, (Hq + Hk + Hk) * HEAD_SIZE], dtype=torch.float16)
        attention_mask = torch.zeros([B, L], dtype=torch.float16)
        # const float head_size_scaler = 1.0f / sqrt(convert_float(S));
        scale = torch.ones([1], dtype=torch.float16) * math.sqrt(HEAD_SIZE)
        print(f'====================={scale=}, {scale.dtype=}')
        
        if use_randn:
            # with open('q.npy', 'rb') as f:
            #     q = np.load(f)
            # with open('k.npy', 'rb') as f:
            #     k = np.load(f)
            # with open('v.npy', 'rb') as f:
            #     v = np.load(f)
            # q = torch.from_numpy(q)
            # k = torch.from_numpy(k)
            # v = torch.from_numpy(v)
            # q = torch.ones([B, L, Hq, HEAD_SIZE], dtype=torch.float16)*torch.randn([1], dtype=torch.float16)
            # k = torch.ones([B, L, Hk, HEAD_SIZE], dtype=torch.float16)*torch.randn([1], dtype=torch.float16)
            q = torch.randn([B, L, Hq, HEAD_SIZE], dtype=torch.float16)
            k = torch.randn([B, L, Hk, HEAD_SIZE], dtype=torch.float16)
            v = torch.randn([B, L, Hk, HEAD_SIZE], dtype=torch.float16)
            # np.save("q.npy", q)
            # np.save("k.npy", k)
            # np.save("v.npy", v)
        else:
            # with open('q_samenumber.npy', 'rb') as f:
            #     q = np.load(f)
            # with open('k_samenumber.npy', 'rb') as f:
            #     k = np.load(f)
            # with open('v_samenumber.npy', 'rb') as f:
            #     v = np.load(f)
            # q = torch.from_numpy(q)
            # k = torch.from_numpy(k)
            # v = torch.from_numpy(v)
            q = torch.ones([B, L, Hq, HEAD_SIZE], dtype=torch.float16)*torch.randn([1], dtype=torch.float16)
            k = torch.ones([B, L, Hk, HEAD_SIZE], dtype=torch.float16)*torch.randn([1], dtype=torch.float16)
            v = torch.ones([B, L, Hk, HEAD_SIZE], dtype=torch.float16)*torch.randn([1], dtype=torch.float16)
            # np.save("q_samenumber.npy", q)
            # np.save("k_samenumber.npy", k)
            # np.save("v_samenumber.npy", v)
            print(f'{Colors.CYAN} q k v shape = {q.shape=} {k.shape=} {v.shape=}.{Colors.END}')
        qkv = torch.cat((q, k, v), 2)
        qkv = torch.reshape(qkv, (B, L, (Hq + Hk + Hk) * HEAD_SIZE))
        
        # ref0, durs = MHA_cl_impl(qkv.clone(), attention_mask.clone(), scale.clone(), Hq, Hk, HEAD_SIZE)
        # print(f'{ref0=}\n')
        # for i, ns in enumerate(durs):
        #     print(f'{Colors.CYAN}{ref0.shape=}, {i}/{len(durs)} {ns*1e-6:.3f} ms {Colors.END}')

        ref, durs = sdpa_impl(q.clone(), k.clone(), v.clone(), attention_mask.clone(), scale.clone(), Hq, Hk, HEAD_SIZE, False)
        # print(f'{ref=}\n')
        for i, ns in enumerate(durs):
            print(f'{Colors.BLUE}{ref.shape=}, {i}/{len(durs)} {ns*1e-6:.3f} ms {Colors.END}')

        opt, durs = sdpa_impl(q.clone(), k.clone(), v.clone(), attention_mask.clone(), scale.clone(), Hq, Hk, HEAD_SIZE, True)
        # print(f'{opt=}\n')
        for i, ns in enumerate(durs):
            print(f'{Colors.BLUE}{opt.shape=}, {i}/{len(durs)} {ns*1e-6:.3f} ms {Colors.END}')
            
        # print(f'all zeros? {np.all(ref == 0)} {np.all(opt == 0)}')

        try:
            if not np.allclose(ref, opt, atol=0.01, rtol=0.01, equal_nan=True):
                pos = np.where(np.abs(ref - opt) > 0.01)
                # print(f"{pos=} {pos[2].shape=} {pos[3].shape=}")
                # print(f'ref_val = {ref[pos]}\nopt_val={opt[pos]}\n')
                raise Exception("failed.")
            print(f'{Colors.GREEN} PASS at shape = {opt.shape}.{Colors.END}')
        except Exception as inst:
            print(f'{Colors.RED} FAIL at shape = {opt.shape}.{Colors.END}')

    # "B, Hq, Hk, HEAD_SIZE, L"
    test_acc(1, 28, 7, 128, 8410)   # tail
    test_acc(1, 24, 6, 128, 2134)   # tail
    # test_acc(1, 28, 7, 128, 64*128)
    # test_acc(1, 24, 6, 128, 16*128)
    # test_acc(1, 1, 1, 128, 16*512)
    # test_acc(1, 1, 1, 128, 16*2)
    # test_acc(1, 1, 1, 128, 16+1)
    # for k in range(1, 10):
    #     test_acc(1, 1, 1, 128, 16*k+1)
    sys.exit(0)
