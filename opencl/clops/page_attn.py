from . import cl
import math
from .utils import *
import clops

class SDPA_opt:
    def __init__(self, Hq, Hk, HEAD_SIZE, is_optimized):
        self.is_optimized = is_optimized
        
        self.SG_SCALE_FACTOR=2
        self.SUBGROUP_SIZE=16
        SEQ_LEN_PARTITION_SIZE=(HEAD_SIZE*self.SG_SCALE_FACTOR)
        self.TARGET_SEQ_LEN_BLOCK_SIZE=16
        
        assert(Hq % Hk == 0);     # implied
        assert(HEAD_SIZE % self.SUBGROUP_SIZE == 0);     # implied
        assert(self.TARGET_SEQ_LEN_BLOCK_SIZE == self.SUBGROUP_SIZE);   # implied
        
        if self.is_optimized:
            cl_source_file = "cl_kernels/page_attn_opt.cl"
            self.kernel_name = 'pa_sdpa_opt_multi_tokens_2'
        else:
            cl_source_file = "cl_kernels/page_attn.cl"
            self.kernel_name = 'pa_sdpa_opt_multi_tokens'

        with open(cl_source_file, "r") as file:
            # Read the entire file content into a string
            cl_kernel_sources = file.read()
        # print(cl_kernel_sources[:100])
        options = f'-DHEAD_SIZE={HEAD_SIZE} -DNUM_HEADS={Hq} -DNUM_KV_HEADS={Hk} \
                    -DSG_SCALE_FACTOR={self.SG_SCALE_FACTOR} -DSEQ_LEN_PARTITION_SIZE={SEQ_LEN_PARTITION_SIZE} -cl-mad-enable'
        self.cl_kernels = kernel_cache(cl_kernel_sources, options)

    def __call__(self, shape_info_input, query_input, key_input, value_input, scale_input):
        B, Hq, L, HEAD_SIZE = query_input.size()

        # if self.is_optimized:
        #     GWS = [HEAD_SIZE*self.SG_SCALE_FACTOR, B*Hq, int(L/self.TARGET_SEQ_LEN_BLOCK_SIZE)]
        #     LWS = [HEAD_SIZE*self.SG_SCALE_FACTOR, Hq//Hk, 1]
        # else:
        GWS = [B*Hq, math.ceil(L/self.TARGET_SEQ_LEN_BLOCK_SIZE), HEAD_SIZE*self.SG_SCALE_FACTOR]
        LWS = [1, 1, HEAD_SIZE*self.SG_SCALE_FACTOR]

        print(f"GWS={GWS}, LWS={LWS}")
        print(self.cl_kernels.info(self.kernel_name, LWS, self.SUBGROUP_SIZE))
        
        shape_info_input = to_cl(torch.tensor(shape_info_input).int())
    
        # shape inference       
        subsequence_begins = torch.zeros(2, dtype=torch.int32)
        subsequence_begins[0] = 0
        subsequence_begins[1] = L
        blocked_indexes_start = torch.zeros(math.ceil(L/self.TARGET_SEQ_LEN_BLOCK_SIZE), dtype=torch.int32)
        blocked_indexes_end = torch.zeros(math.ceil(L/self.TARGET_SEQ_LEN_BLOCK_SIZE), dtype=torch.int32)
        for i in range(math.ceil(L/self.TARGET_SEQ_LEN_BLOCK_SIZE)):
            blocked_indexes_start[i] = 16 * i
            blocked_indexes_end[i] = 16 * (i + 1)
        gws_seq_indexes_correspondence = torch.zeros(math.ceil(L/self.TARGET_SEQ_LEN_BLOCK_SIZE), dtype=torch.int32)

        output = to_cl(torch.zeros(B, Hq, L, HEAD_SIZE, dtype=torch.float16))

        self.cl_kernels.enqueue(self.kernel_name, GWS, LWS,
                            shape_info_input, to_cl(query_input), to_cl(key_input), to_cl(value_input), to_cl(subsequence_begins), output,
                            to_cl(blocked_indexes_start), to_cl(blocked_indexes_end), to_cl(gws_seq_indexes_correspondence))

        return output

if __name__ == "__main__":
    import sys
    import numpy as np
    import math
    cl.profiling(True)
    np.set_printoptions(precision=3, suppress=True)
   
    def sdpa_impl(q : torch.Tensor, k : torch.Tensor, v : torch.Tensor, scale : torch.Tensor, Hq, Hk, HEAD_SIZE, is_optimized):
        # print(f'{q.size()=}\n{k.size()=}\n{v.size()=}')
        B, Lq, _, _ = q.size()
        _, Lk, _, _ = k.size()
        query = q.view(B, Lq, Hq, HEAD_SIZE).transpose(1, 2).contiguous()
        key = k.view(B, Lk, Hk, HEAD_SIZE).transpose(1, 2).contiguous()
        value = v.view(B, Lk, Hk, HEAD_SIZE).transpose(1, 2).contiguous()
        # print(f'{query=}\n{key=}\n{value=}')
       
        batch_size_in_tokens = Lq

        shape_info = [
            # // input0 query
            batch_size_in_tokens, Hq*HEAD_SIZE, 1, 1, 1, 1, 1, 1,
            # // input1 key
            batch_size_in_tokens, Hk*HEAD_SIZE, 1, 1, 1, 1, 1, 1,
            # // input2 value [16 - 25]
            batch_size_in_tokens, Hk*HEAD_SIZE, 1, 1, 1, 1, 1, 1, 0, 0,
            #  input3 [26 - 33]
            327, Hq, 1, 1, 1, 1, HEAD_SIZE, 16,
            #  input4 [34 - 41]
            327, Hq, 1, 1, 1, 1, 16, HEAD_SIZE,
            #  input5
            1, 1, 1, 1, 1, 1, 1, 1,
            #  input6 [50 - 57]
            2, 1, 1, 1, 1, 1, 1, 1,
            #  input7 [58 - 65]
            16, 1, 1, 1, 1, 1, 1, 1,
            #  input8 [66 -73]
            2, 1, 1, 1, 1, 1, 1, 1,
            #  output [74 - 81]
            batch_size_in_tokens, Hq*HEAD_SIZE, 1, 1, 1, 1, 1, 1
        ]
        # print(f"len(shape_info)={len(shape_info)}, shape_info={shape_info}")

        sdpa = SDPA_opt(Hq, Hk, HEAD_SIZE, is_optimized)
        output = sdpa(shape_info, query, key, value, scale)

        output = to_torch(output)
        durations = cl.finish()
        return output.numpy(), durations

    def test_acc(B, Hq, Hk, HEAD_SIZE, Lq, Lk, use_randn = False):
        scale = torch.ones([1], dtype=torch.float16) / math.sqrt(HEAD_SIZE)
        # scale = torch.ones([1], dtype=torch.float16)
        print(f'====================={scale=}, {scale.dtype=}')
        
        if use_randn:
            with open('q.npy', 'rb') as f:
                q = np.load(f)
            with open('k.npy', 'rb') as f:
                k = np.load(f)
            with open('v.npy', 'rb') as f:
                v = np.load(f)
            q = torch.from_numpy(q)
            k = torch.from_numpy(k)
            v = torch.from_numpy(v)
            # q = torch.ones([B, L, Hq, HEAD_SIZE], dtype=torch.float16)*torch.randn([1], dtype=torch.float16)
            # k = torch.ones([B, L, Hk, HEAD_SIZE], dtype=torch.float16)*torch.randn([1], dtype=torch.float16)
            # q = torch.randn([B, Lq, Hq, HEAD_SIZE], dtype=torch.float16)
            # k = torch.randn([B, Lk, Hk, HEAD_SIZE], dtype=torch.float16)
            # v = torch.randn([B, Lk, Hk, HEAD_SIZE], dtype=torch.float16)
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
            q = torch.ones([B, Lq, Hq, HEAD_SIZE], dtype=torch.float16)
            k = torch.ones([B, Lk, Hk, HEAD_SIZE], dtype=torch.float16)
            v = torch.ones([B, Lk, Hk, HEAD_SIZE], dtype=torch.float16)
            # np.save("q_samenumber.npy", q)
            # np.save("k_samenumber.npy", k)
            # np.save("v_samenumber.npy", v)
            # print(f'{Colors.CYAN} q k v shape = {q.shape=} {k.shape=} {v.shape=}.{Colors.END}')

        ref, durs = sdpa_impl(q.clone(), k.clone(), v.clone(), scale.clone(), Hq, Hk, HEAD_SIZE, False)
        print(f'{ref=}\n')
        ref_t = 0
        for i, ns in enumerate(durs):
            print(f'{Colors.BLUE}{ref.shape=}, {i}/{len(durs)} {ns*1e-6:.3f} ms {Colors.END}')
            ref_t += ns

        opt, durs = sdpa_impl(q.clone(), k.clone(), v.clone(), scale.clone(), Hq, Hk, HEAD_SIZE, True)
        print(f'{opt=}\n')
        for i, ns in enumerate(durs):
            print(f'{Colors.BLUE}{opt.shape=}, {i}/{len(durs)} {ns*1e-6:.3f} ms, with improvment {(ref_t - ns)/ref_t*100:.2f}% {Colors.END}')
            
        # print(f'all zeros? {np.all(ref == 0)} {np.all(opt == 0)}')

        try:
            if not np.allclose(ref, opt, atol=0.01, rtol=0.01, equal_nan=True):
                pos = np.where(np.abs(ref - opt) > 0.01)
                print(f"{pos=} {pos[2].shape=} {pos[3].shape=}")
                if not pos[0].size > 0:
                    pos = np.where(np.isnan(opt))
                    # print(f"{pos=} {pos[2].shape=} {pos[3].shape=}")
                print(f'ref_val = {ref[pos]}\nopt_val={opt[pos]}\n')
                raise Exception("failed.")
            print(f'{Colors.GREEN} ref:opt PASS at shape = {opt.shape}.{Colors.END}')
        except Exception as inst:
            print(f'{Colors.RED} ref:opt FAIL at shape = {opt.shape}.{Colors.END}')

    # "B, Hq, Hk, HEAD_SIZE, Lq, Lk"
    for _ in range(1):
        test_acc(1, 40, 40, 128, 256, 256, True)   # tail
# vtune -collect gpu-hotspots -knob characterization-mode=overview -knob collect-memory-bandwidth=true -knob analyze-power-usage=false --
    sys.exit(0)
