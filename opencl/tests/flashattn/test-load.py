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


seq_len = 8192
head_num = 28
kvhead_num = 4
head_sz = 128

token_blk = 1
local_sz=64
token_per_wg=local_sz*token_blk
low=-2
high = 3

q_factor = torch.randint(high, high+3, [seq_len, head_num, head_sz]).to(dtype=torch.float16)
kv_factor = torch.randint(high, high+3, [seq_len, kvhead_num, head_sz]).to(dtype=torch.float16)

q = torch.randint(low, high, [seq_len, head_num, head_sz]).to(dtype=torch.float16) / q_factor
k = torch.randint(low, high, [seq_len, kvhead_num, head_sz]).to(dtype=torch.float16) / kv_factor
v = torch.randint(low, high, [seq_len, kvhead_num, head_sz]).to(dtype=torch.float16) / kv_factor

qint8_ref = torch.randint(low, high, [seq_len, head_num, head_sz]).to(dtype=torch.int8)
kint8_ref = torch.randint(low, high, [seq_len, kvhead_num, head_sz]).to(dtype=torch.int8)

qscale_ref = torch.zeros(seq_len, head_num).to(dtype=torch.float32)
kscale_ref = torch.zeros(seq_len, kvhead_num).to(dtype=torch.float32)

scale_out = torch.zeros((seq_len*head_num + token_blk -1)//token_blk*token_blk).to(dtype=torch.float32)

cl.profiling(True)

src = r'''

#pyeval f"#define TOKEN_NUM {token_blk}"
#pyeval f"#define HEAD_SZ {head_sz}"
#pyeval f"#define HEAD_NUM {head_num}"
#pyeval f"#define KVHEAD_NUM {kvhead_num}"

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

extern "C" _GENX_MAIN_ void cm_test2d(SurfaceIndex base [[type("buffer_t")]], SurfaceIndex scale [[type("buffer_t")]]) {
    auto id = cm_group_id(0)*cm_local_size(0) + cm_linear_local_id();
    if (id*TOKEN_NUM >= HEAD_NUM*SEQ_LEN)
        return;
    auto head = id % HEAD_NUM;
    auto seq = id / HEAD_NUM;
    auto pitch = HEAD_SZ*sizeof(half);
    auto offset = (seq * (HEAD_NUM + KVHEAD_NUM + KVHEAD_NUM) + head)*pitch;
    auto scale_off = id*TOKEN_NUM*sizeof(float);


    auto remaining_tokens = (HEAD_NUM*SEQ_LEN-id*TOKEN_NUM) >= TOKEN_NUM ? TOKEN_NUM : (HEAD_NUM*SEQ_LEN-id*TOKEN_NUM);

    vector<half, HEAD_SZ> token;
    vector<float, TOKEN_NUM> scaleV;

    auto quan_token = token.format<int8_t,2, HEAD_SZ>().row(0);

    if (remaining_tokens == TOKEN_NUM) {
        #pragma unroll
        for(int i= 0;i<TOKEN_NUM;i++,offset+=pitch) {
            token.format<uint32_t>() = cm_load<uint, HEAD_SZ/2>(base, offset);
            half max=cm_reduced_max<half>(cm_abs(token));
            quan_token =  cm_mul<int8_t>(token, (float)(127.0)/(float)(max));
            cm_store<uint32_t, HEAD_SZ/4>(base, offset, quan_token.format<uint32_t>());
            scaleV[i] = (float)(max)/127.0;
        }
    } else {
        for(int i= 0;i<remaining_tokens;i++,offset+=pitch) {
            token.format<uint32_t>() = cm_load<uint, HEAD_SZ/2>(base, offset);
            //half max=cm_reduced_max<half>(cm_abs(token));
            //quan_token =  cm_mul<int8_t>(token, float(127.0)/float(max));
            cm_store<uint32_t, HEAD_SZ/4>(base, offset, quan_token.format<uint32_t>());
            //scaleV[i] = (float)(max)/127.0;
        }

    }
    cm_store<uint32_t, TOKEN_NUM>(scale, scale_off, scaleV.format<uint32_t>());

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
        hkv = h // (head_num//kvhead_num)
        qtoken=q[seq, h, :]
        qmax=torch.amax(qtoken.abs(), dim=0, keepdim=True)
        qtoken_int8=(qtoken/qmax*127.0).to(dtype=torch.int8)
        qint8_ref[seq,h,:]=qtoken_int8
        qscale_ref[seq, h]=float(qmax)/127.0

        ktoken=k[seq,hkv,:]
        kmax=torch.amax(ktoken.abs(), dim=0, keepdim=True)
        ktoken_int8=(ktoken/kmax*127.0).to(dtype=torch.int8)
        kint8_ref[seq,hkv,:]=ktoken_int8
        kscale_ref[seq, hkv]=float(kmax)/127.0


all_layers=[]
mem_size=0
while mem_size < 4e9:
    all_layers.append([
        cl.tensor(q.detach().numpy())
    ])
    mem_size += q.numel() * q.element_size()
qkv =  torch.cat((q,k,v), 1)
t_qkvlist = [cl.tensor(qkv.to(torch.float16).detach().numpy()) for _ in range(50)]
t_qscaleList = [cl.tensor(scale_out.detach().numpy()) for _ in range(50)]

print(f'head_num:{head_num}, seq_len:{seq_len}, token_per_wg:{token_per_wg}, token_blk:{token_blk}')

cm_kernels = cl.kernels(pyeval(src), f"-cmc -mdump_asm -g2 ")

lws = [local_sz]
tokens_align_up=(head_num*seq_len+token_per_wg-1)//token_per_wg*local_sz
gws = [tokens_align_up]
print(f'GWS:{gws}, LWS:{lws}')
for i in range(50):
    cm_kernels.enqueue("cm_test2d", gws, lws, t_qkvlist[i], t_qscaleList[i])
lat=cl.finish()

rdbytes=seq_len*head_num*head_sz*2
ns=sum(lat[5:])/len(lat[5:])
print(f'avg latency:{ns*1e-3:.2f} us, read:{rdbytes/ns:.2f} GB/S, write:{(rdbytes/2+head_num*seq_len*4)/ns:.2f} GB/S')

qint8_out=t_qkvlist[0].numpy()[:,0:head_num,0:head_sz//2].view(np.int8).reshape((seq_len, head_num, head_sz))
qscale_out=t_qscaleList[0].numpy()[0:seq_len*head_num].view(np.float32).reshape((seq_len, head_num))

check_close(qint8_ref,torch.from_numpy(qint8_out))
# check_close(qscale_ref,torch.from_numpy(qscale_out))


cl.finish()
