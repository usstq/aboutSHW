import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)
torch.set_printoptions(linewidth=1024)

# prototyping CM kernels
from clops import cl
import numpy as np
import time

def check_close(input, other, atol=1e-3, rtol=1e-3):
    print(f"[check_close] {input.shape}{input.dtype} vs {other.shape}{other.dtype}")
    rtol_max = (((input - other).abs() - 1e-5)/other.abs())[other != 0].max()
    atol_max = (((input - other).abs()) - 1e-5*other.abs()).max()
    print(f"[check_close] rtol_max: {rtol_max}")
    print(f"[check_close] atol_max: {atol_max}")
    if not torch.allclose(input, other, atol=atol, rtol=rtol):
        close_check = torch.isclose(input, other, atol=atol, rtol=rtol)
        not_close_indices = torch.where(~close_check) # Invert the close check to find failures
        print(f"Not close indices: {not_close_indices}")
        print(f"    input_tensor: {input[not_close_indices]}")
        print(f"    other_tensor: {other[not_close_indices]}")
        assert 0

def get_cm_grf_width():
    cm_kernels = cl.kernels(r'''
    extern "C" _GENX_MAIN_ void cm_get_grf_width(int * info [[type("svmptr_t")]]) {
        info[0] = CM_GRF_WIDTH;
    }''', f"-cmc")
    t_info = cl.tensor([2], np.dtype(np.int32))
    cm_kernels.enqueue("cm_get_grf_width", [1], [1], t_info)
    return t_info.numpy()[0]

CM_GRF_WIDTH = get_cm_grf_width()

seq_len = 8192
head_num = 28
head_sz = 128
MAX_GRF_NUM = 64
assert head_sz%32 == 0
#CM_GRF_WIDTH is 256 or 512, headsize%32 can ensure the (head_sz*16)%CM_GRF_WIDTH == 0
GRFs_per_token=head_sz*16//CM_GRF_WIDTH
#total
token_blk = MAX_GRF_NUM//GRFs_per_token
local_sz=64
token_per_wg=local_sz*token_blk
low=-2
high = 3

q_factor = torch.randint(high, high+3, [seq_len, head_num, head_sz]).to(dtype=torch.float16)
q = torch.randint(low, high, [seq_len, head_num, head_sz]).to(dtype=torch.float16) / q_factor
qint8 = torch.randint(low, high, [seq_len, head_num, head_sz]).to(dtype=torch.int8)
out = torch.zeros(seq_len, head_num, head_sz*2).to(dtype=torch.int8)
cl.profiling(True)



src = r'''

#pyeval f"#define TOKEN_NUM {token_blk}"
#pyeval f"#define HEAD_SZ {head_sz}"
#pyeval f"#define HEAD_NUM {head_num}"
#pyeval f"#define SEQ_LEN {seq_len}"


template<typename T, int M, int N>
void show(const matrix<T, M, N> mat) {
    printf("Matrix [%d, %d]:\n", M, N);
    for(int m = 0; m < M; m ++) {
        printf("\t[");
        for(int n = 0; n < N; n ++) {
            printf("%4d,", mat[m][n]);
        }
        printf("],\n");
    }
    printf("]\n");
}

template<typename T, int M, int N>
void show_float(const matrix<T, M, N> mat) {
    printf("Matrix [%d, %d]:\n", M, N);
    for(int m = 0; m < M; m ++) {
        printf("\t[");
        for(int n = 0; n < N; n ++) {
            printf("%8.4f,", mat[m][n]);
        }
        printf("],\n");
    }
    printf("]\n");
}

extern "C" _GENX_MAIN_ void cm_test2d(SurfaceIndex base [[type("buffer_t")]]) {
    auto id = cm_group_id(0)*cm_local_size(0) + cm_linear_local_id();
    auto pitch = HEAD_SZ*sizeof(half);
    auto offset = id*TOKEN_NUM*pitch;
    if (id*TOKEN_NUM >= HEAD_NUM*SEQ_LEN)
        return;

    auto remaining_tokens = (HEAD_NUM*SEQ_LEN-id*TOKEN_NUM) >= TOKEN_NUM ? TOKEN_NUM : (HEAD_NUM*SEQ_LEN-id*TOKEN_NUM);

    matrix<half, TOKEN_NUM, HEAD_SZ> tokens;
    auto quan_tokens = tokens.format<int8_t,TOKEN_NUM*2,  HEAD_SZ>();


#if 0
    if (remaining_tokens == TOKEN_NUM) {
        #pragma unroll
        for(int i= 0;i<TOKEN_NUM;i++,offset+=pitch) {
            tokens.row(i).format<uint32_t>() = cm_load<uint, HEAD_SZ/2>(base, offset);
            //half max=cm_reduced_max<half>(cm_abs(tokens[i]));
            //quan_tokens[i] =  cm_mul<int8_t>(tokens[i], float(127.0)/float(max));
            //if (offset != 9999999999)
                //continue;
            cm_store<uint32_t, HEAD_SZ/4>(base, offset, quan_tokens[i].format<uint32_t>());
        }
    } else {
        for(int i= 0;i<remaining_tokens;i++,offset+=pitch) {
            tokens.row(i).format<uint32_t>() = cm_load<uint, HEAD_SZ/2>(base, offset);
            //half max=cm_reduced_max<half>(cm_abs(tokens[i]));
            //quan_tokens[i] =  cm_mul<int8_t>(tokens[i], float(127.0)/float(max));
            cm_store<uint32_t, HEAD_SZ/4>(base, offset, quan_tokens[i].format<uint32_t>());
        }
    }
#endif
}

'''

def pyeval(src):

    result_src = ""
    for line in src.splitlines():
        if line.startswith("#pyeval"):
            new_line = eval(line[8:])
            result_src += new_line + "\n"
        else:
            result_src += line + "\n"
    return result_src


for seq in range(seq_len):
    for h in range(head_num):
        token=q[seq, h, :]
        max=torch.amax(token.abs(), dim=0, keepdim=True)
        qtoken=(token/max*127.0).to(dtype=torch.int8)
        qint8[seq,h,:]=qtoken

all_layers=[]
mem_size=0
while mem_size < 4e9:
    all_layers.append([
        cl.tensor(q.detach().numpy())
    ])
    mem_size += q.numel() * q.element_size()
t_Alist = [cl.tensor(q.to(torch.float16).detach().numpy()) for _ in range(50)]

print(f'GRFs_per_token:{GRFs_per_token}, head_num:{head_num}, seq_len:{seq_len}, token_per_wg:{token_per_wg}, token_blk:{token_blk}')

cm_kernels = cl.kernels(pyeval(src), f"-cmc -mdump_asm -g2 ")

lws = [local_sz]
tokens_align_up=(head_num*seq_len+token_per_wg-1)//token_per_wg*local_sz
gws = [tokens_align_up]
print(f'GWS:{gws}, LWS:{lws}')
for i in range(50):
    cm_kernels.enqueue("cm_test2d", gws, lws, t_Alist[i])
lat=cl.finish()

rdbytes=seq_len*head_num*head_sz*2
ns=sum(lat[5:])/len(lat[5:])
print(f'avg latency:{ns*1e-3:.2f} us, read:{rdbytes/ns:.2f} GB/S, write:{rdbytes/2/ns:.2f} GB/S')

out=t_Alist[0].numpy()[:,:,0:head_sz//2].view(np.int8).reshape((seq_len, head_num, head_sz))
# check_close(qint8,torch.from_numpy(out))





cl.finish()
