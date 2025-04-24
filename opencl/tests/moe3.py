#!/usr/bin/python3
from clops import cl
import numpy as np
import sys
import time, sys
np.set_printoptions(linewidth=1024)


def pack_i8_to_i4(B_q):
    assert B_q.ndim == 2
    K = B_q.shape[0]
    N = B_q.shape[1]
    B_q4 = np.zeros([K, N//2], dtype=np.int8)
    for k in range(K):
        for n in range(N//2):
            even = (B_q[k,2*n]) & 0xF
            odd = (B_q[k,2*n+1]) & 0xF
            B_q4[k,n] = even | (odd << 4)
    return B_q4

cm_src = r'''
#define GEN_KERNEL

#undef CM_DEBUG

#include <cm/cm.h>

#ifdef CMRT_EMU
# include <shim_support.h>
#endif  //  CMRT_EMU

typedef half fp16;

#ifndef BTI
#define BUFFERINDEX(x) int * x [[type("svmptr_t")]]
#define LSC_LOAD(DATA_TYPE, VS, L1, L3) cm_ptr_load<DATA_TYPE,VS,DataSize::Default,CacheHint::L1,CacheHint::L3>
#define LSC_STORE(DATA_TYPE, VS, L1, L3) cm_ptr_store<DATA_TYPE,VS,DataSize::Default,CacheHint::L1,CacheHint::L3>
#define LSC_LOAD_GATHER(DATA_TYPE, VS, L1, L3,channels) cm_ptr_load<DATA_TYPE,VS,DataSize::Default,CacheHint::L1,CacheHint::L3, channels>
//#define LSC_STORE_2D(DATA_TYPE, h, w, L1, L3) cm_store<DATA_TYPE,h, w, CacheHint::L1,CacheHint::L3>
#define LSC_STORE_2D(DATA_TYPE, h, w, L1, L3) cm_ptr_store<DATA_TYPE,h, w, CacheHint::L1,CacheHint::L3>
#define LSC_LOAD_0(DATA_TYPE, VS)  cm_ptr_load<DATA_TYPE,VS,DataSize::Default,CacheHint::Cached,CacheHint::Cached>
#else
#define BUFFERINDEX(x) SurfaceIndex x [[type("buffer_t")]]
#define LSC_LOAD(DATA_TYPE, VS, L1, L3) cm_load<DATA_TYPE,VS,DataSize::Default,CacheHint::L1,CacheHint::L3>
#define LSC_STORE(DATA_TYPE, VS, L1, L3) cm_store<DATA_TYPE,VS,DataSize::Default,CacheHint::L1,CacheHint::L3>
#define LSC_LOAD_GATHER(DATA_TYPE, VS, L1, L3,channels) cm_load<DATA_TYPE,VS,DataSize::Default,CacheHint::L1,CacheHint::L3, channels>
#define LSC_STORE_2D(DATA_TYPE, h, w, L1, L3) cm_store<DATA_TYPE,h, w, CacheHint::L1,CacheHint::L3>
#endif


constexpr int QK = 32; // compatible
constexpr int SBS = 8; // compatible
constexpr int BLOCK_SIZE = QK / 2;
constexpr int SCALE_SIZE = sizeof(fp16);
#define NUM_EXPERTS             8
#define NUM_MOE_MLP_LAYER       128

inline auto load_qblocks(const uint8_t* weight, const uint8_t* scale)
{
    vector<uint8_t, BLOCK_SIZE* SBS> ybytes;
    ybytes.format<int>() = LSC_LOAD_0(int, (BLOCK_SIZE * SBS / 4))((int*)weight, 0);

    vector<fp16, SBS> scales;
    scales.format<int>() = LSC_LOAD_0(int, (SBS / 2))((int*)scale, 0);
    //vector<fp16, SBS> scales = LSC_LOAD(fp16, SBS, Cached, Cached)((fp16*)scale, 0);  compiler error


    vector<fp16, QK* SBS> yvs;
#pragma unroll
    for (int i = 0; i < SBS; ++i) {
        vector<uint8_t, QK> uyv;
        // uyv.select<QK / 2, 1>(0) = ybytes.template select<QK / 2, 1>(i * QK / 2) & (uint8_t)0xF;
        // uyv.select<QK / 2, 1>(QK / 2) = ybytes.template select<QK / 2, 1>(i * QK / 2) >> (uint8_t)4;
        uyv.select<QK / 2, 2>(0) = ybytes.template select<QK / 2, 1>(i * QK / 2) & (uint8_t)0xF; // compatible
        uyv.select<QK / 2, 2>(1) = ybytes.template select<QK / 2, 1>(i * QK / 2) >> (uint8_t)4; // compatible
        yvs.template select<QK, 1>(i * QK) = (uyv.format<int8_t>() - (int8_t)8) * scales[i];
    }
    return yvs;
}

inline auto load_qblock(const uint8_t* weight, const uint8_t* scale) {
    vector<uint8_t, BLOCK_SIZE> ybytes;
    ybytes.format<int>() = LSC_LOAD_0(int, (BLOCK_SIZE / 4))((int*)weight, 0);
    fp16 scales = *(const fp16*)scale;

    vector<uint8_t, QK> uyv;
    // uyv.select<QK / 2, 1>(0) = ybytes & (uint8_t)0xF;
    // uyv.select<QK / 2, 1>(QK / 2) = ybytes >> (uint8_t)4;
    uyv.select<QK / 2, 2>(0) = ybytes & (uint8_t)0xF; // compatible
    uyv.select<QK / 2, 2>(1) = ybytes >> (uint8_t)4; // compatible
    vector<fp16, QK> yv = (uyv.format<int8_t>() - (int8_t)8) * scales;

    return yv;
}
template <typename IT, const int VS, const int GS, const int ES>
void moe_forward_down_kernel(
    BUFFERINDEX(input),
    BUFFERINDEX(indexs),
    BUFFERINDEX(eweights),
    BUFFERINDEX(down_addrs),
    BUFFERINDEX(down_scales_addrs),
    BUFFERINDEX(output),
    const int num_experts,
    const int state_size,
    const int output_size,
    uint slmX
) {
    //cm_slm_init(GS * VS * sizeof(float));
    //uint slmX = cm_slm_alloc(GS * VS * sizeof(float));

    const int nb = state_size / QK;
    const int nsb = nb / SBS;

    const int eid = cm_group_id(0) * cm_local_size(0) + cm_local_id(0);
    const int tid = cm_local_id(1);
    const int vid = cm_group_id(1) * VS;

    vector<int, NUM_EXPERTS * 2> indexs_vec = LSC_LOAD_0(int, NUM_EXPERTS * 2)(indexs, 0);
    int index = indexs_vec.format<unsigned long long>()[eid];

    vector<IT, NUM_EXPERTS> eweights_vec;
    eweights_vec.format<int>() = LSC_LOAD_0(int, 4)(eweights, 0);   //8 fp16
    const IT eweight = eweights_vec[eid];

    int g_offset = index / 8;
    int l_offset = index % 8;
    vector<int, 16> gpu_down_addr = LSC_LOAD_0(int, 16)(down_addrs, g_offset * 64);
    const uint8_t* weight = (uint8_t*)gpu_down_addr.format<unsigned long long>()[l_offset];

    vector<int, 16> gpu_down_scales_addr = LSC_LOAD_0(int, 16)(down_scales_addrs, g_offset * 64);
    const uint8_t* scales = (const uint8_t*)gpu_down_scales_addr.format<unsigned long long>()[l_offset];


    //const IT* input = static_cast<const IT*>(input_ptr) + eid * state_size;
    int input_base_offset = eid * state_size * sizeof(IT);

    const uint8_t* weight_base = weight + nb * BLOCK_SIZE * vid;
    const uint8_t* scale_base = scales + nb * SCALE_SIZE * vid;

    vector<IT, VS* ES> accvs{};

    for (int s = tid; s < nsb; s += GS) {
        //8x32 fp16
        vector<IT, SBS* QK> xvs;
        xvs.format<int>().select<64, 1>(0) = LSC_LOAD_0(int, (SBS * QK / 4))((int*)input, input_base_offset + s * SBS * QK * sizeof(IT));
        xvs.format<int>().select<64, 1>(64) = LSC_LOAD_0(int, (SBS * QK / 4))((int*)input, input_base_offset + s * SBS * QK * sizeof(IT) + 256);

#pragma unroll
        for (int v = 0; v < VS; ++v) {
            vector<fp16, SBS* QK> yvs = load_qblocks(
                weight_base + v * nb * BLOCK_SIZE + s * SBS * BLOCK_SIZE,
                scale_base + v * nb * SCALE_SIZE + s * SBS * SCALE_SIZE
            );

#pragma unroll
            for (int i = 0; i < SBS * QK; i += ES) {
                accvs.template select<ES, 1>(v * ES) +=
                    xvs.template select<ES, 1>(i) *
                    yvs.template select<ES, 1>(i);
            }
        }
    }

    for (int b = nsb * SBS + tid; b < nb; b += GS) {
        vector<IT, QK> xv;
        xv.format<int>() = LSC_LOAD_0(int, (QK / 2))((int*)input, input_base_offset + b * QK * sizeof(IT));

#pragma unroll
        for (int v = 0; v < VS; ++v) {
            vector<fp16, QK> yv = load_qblock(
                weight_base + v * nb * BLOCK_SIZE + b * BLOCK_SIZE,
                scale_base + v * nb * SCALE_SIZE + b * SCALE_SIZE
            );

#pragma unroll
            for (int i = 0; i < QK; i += ES) {
                accvs.template select<ES, 1>(v * ES) +=
                    xv.template select<ES, 1>(i) *
                    yv.template select<ES, 1>(i);
            }
        }
    }

    vector<float, VS> accs;
#pragma unroll
    for (int v = 0; v < VS; ++v) {
        accs[v] = cm_sum<float>(accvs.template select<ES, 1>(v * ES));
    }

    cm_slm_block_write<float, VS >(slmX, tid * VS * sizeof(float), accs);

    cm_slm_fence(0x20);
    cm_barrier();

    if (tid == 0) {
#pragma unroll
        for (int i = 1; i < GS; ++i) {
            vector<float, VS> accs_slm;
            cm_slm_block_read<float, VS>(slmX, i * VS * sizeof(float), accs_slm);
            accs += accs_slm;
        }

        vector<IT, VS> accs_fp16 = accs * eweight;
        LSC_STORE(int, VS/2, WriteBack, WriteBack)((int*)output, (eid * output_size + vid) * sizeof(IT), accs_fp16.format<int>());//fp16
    }
}

_GENX_MAIN_ void moe_up_kernel(
    int* input, 			// shape: f16[1, 2048]
    int* indexs,			// shape: u64[8] ??? i32
    int* gate_addrs,		// shape: u64[128]
    int* up_addrs,			// shape: u64[128]
    int* gate_scales_addrs,	// shape: f16[768, 2048/32=64] 
    int* up_scales_addrs,	// shape: f16[768, 2048/32=64]
    int* output,			// shape: f16[8, 1, 768]
    const int num_experts,
    const int state_size,     // 2048
    const int output_size) {  // 768
#if 0
    moe_forward_up_kernel<fp16, 2U, 4U, 32U>(
    input, 			// shape: f16[1, 2048]
    indexs,			// shape: u64[8] ??? i32
    gate_addrs,		// shape: u64[128]
    up_addrs,			// shape: u64[128]
    gate_scales_addrs,	// shape: f16[768, 2048/32=64] 
    up_scales_addrs,	// shape: f16[768, 2048/32=64]
    output,			// shape: f16[8, 1, 768]
    num_experts,
    state_size,     // 2048
    output_size);   // 768
    #endif
}

_GENX_MAIN_ void moe_down_kernel(
    BUFFERINDEX(input),		// shape: f16[8, 1, 768]
    BUFFERINDEX(indexs),	// shape: i64[8]   ??? i32
    BUFFERINDEX(eweights),	// shape: f16[8]
    BUFFERINDEX(down_addrs),// shape: u64[128]
    BUFFERINDEX(down_scales_addrs),	// shape: u64[128]
    BUFFERINDEX(output),			// shape: f16[8, 2048]
    const int num_experts,			// const: 8
    const int state_size,			// const: 768
    const int output_size) {    	// const: 2048
    cm_slm_init(4 * 4 * sizeof(float));
    uint slmX = cm_slm_alloc(4 * 4 * sizeof(float));
    moe_forward_down_kernel<fp16, 4U, 4U, 32U>(
        input,		// shape: f16[8, 1, 768]
        indexs,	// shape: i64[8]   ??? i32
        eweights,	// shape: f16[8]
        down_addrs,// shape: u64[128]
        down_scales_addrs,	// shape: u64[128]
        output,			// shape: f16[8, 2048]
        num_experts,			// const: 8
        state_size,			// const: 768
        output_size,
        slmX);			// const: 2048
}

#ifdef CMRT_EMU
EXPORT_SIGNATURE(sgemm_kernel);
#endif  //  CMRT_EMU
'''

GROUP_SIZE = 128
INTERMEDIATE_SIZE = 768
HIDDEN_SIZE = 2048
MAX_TOPK = 8
SUBGROUP_SIZE = 16
SUBGROUP_SIZE = 16
SUBGROUP_NUM = 8*2
N_BLOCK = SUBGROUP_NUM
N_LAYERS = 100 # 24*8

import sys
K_group_size = 32

N_EXPERTS = 8
kernel_name = "mlp_down_n2" #sys.argv[1]
M, K, N = 8, INTERMEDIATE_SIZE, HIDDEN_SIZE


np.random.seed(0)
A = np.random.randint(-1,2,[M, K]).astype(np.float16)
tA = cl.tensor(A)
tC = cl.tensor(np.zeros([M, N], dtype=np.float16))
tP1 = cl.tensor()

assert (K % K_group_size) == 0
K_groups = K // K_group_size
B_q = np.random.randint(0,3,[K, N]).astype(np.int8)

B_scale = np.random.randint(-4,5,[K_groups, N]).astype(np.float16)

B_zp = np.random.randint(0,1,[K_groups, N]).astype(np.int8)
B = (B_q.astype(np.float32) - B_zp.astype(np.float32).repeat(K_group_size, axis=0)) * B_scale.repeat(K_group_size, axis=0)
C = A @ B
print("ref is calculated!")

# pack weight into 4bit
B_q4 = pack_i8_to_i4((B_q + 8).transpose().copy())

#B_q4 = pack_i8_to_i4(B_q.copy())
B_zp4 = pack_i8_to_i4(B_zp)

B_scale = B_scale.transpose().copy()

tBq4 = cl.tensor(B_q4)
tBs = cl.tensor(B_scale)
tBz = cl.tensor(B_zp4)

weight_tensors = [] # for holding reference counter
weight_ptrs = []    # [24, 128]
gate_addrs_ptrs = []
up_addrs_ptrs = []
down_weight_addrs_ptrs = []
down_scales_addrs_ptrs = []
down_zp_addrs_ptrs = []

for i in range(N_LAYERS):
    for k in range(N_EXPERTS):
        t_weight = cl.tensor(B_q4)
        t_scales =  cl.tensor(B_scale)
        t_zero_points = cl.tensor(B_zp4)
        weight_tensors.append([t_weight, t_scales, t_zero_points])
        weight_ptrs.append([t_weight.addr, t_scales.addr, t_zero_points.addr])
        down_weight_addrs_ptrs.append(t_weight.addr)
        down_scales_addrs_ptrs.append(t_scales.addr)
        down_zp_addrs_ptrs.append(t_zero_points.addr)

weight_ptrs = cl.tensor(np.array(weight_ptrs, dtype=np.uint64))

cm_kernels = cl.kernels(cm_src,
                        f" -cmc  -Qxcm_register_file_size=256", "./dump")
m=2048
n=768
k = 1
mProj = m
kProj = n
groupH = 8
groupV = mProj // 4
localH = 1
localV = 4
globalSize = [groupH * localH, groupV * localV]
localSize = [localH, localV]
all_index = []
index_np = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.uint64)
for i in range(N_LAYERS):
    all_index.append(cl.tensor(index_np + i*8))

# template
# _GENX_MAIN_ void moe_forward_down_kernel<fp16, 4U, 4U, 32U>(
#     BUFFERINDEX(input),		// shape: f16[8, 1, 768]
#     BUFFERINDEX(indexs),	// shape: i64[8]   ??? i32
#     BUFFERINDEX(eweights),	// shape: f16[8]
#     BUFFERINDEX(down_addrs),// shape: u64[128]
#     BUFFERINDEX(down_scales_addrs),	// shape: u64[128]
#     BUFFERINDEX(output),			// shape: f16[8, 2048]
#     const int num_experts,			// const: 8
#     const int state_size,			// const: 768
#     const int output_size			// const: 2048
# );
eweights = cl.tensor(np.ones(N_EXPERTS, np.float16))
down_weight_ptrs = cl.tensor(np.array(down_weight_addrs_ptrs, dtype=np.uint64))
down_scales_ptrs = cl.tensor(np.array(down_scales_addrs_ptrs, dtype=np.uint64))

cm_kernels.enqueue("moe_down_kernel", globalSize, localSize,
                    tA,         #     int* input, 			// shape: f16[8, 1, 768]
                    all_index[0],      #     int* indexs,			// shape: u64[8] ??? i32
                    eweights,   #     BUFFERINDEX(eweights),	// shape: f16[8]
                    down_weight_ptrs, #     BUFFERINDEX(down_addrs),// shape: u64[128]
                    down_scales_ptrs, #     BUFFERINDEX(down_scales_addrs),	// shape: u64[128]
                    tC,         #     int* output,			// shape: f16[8, 1, 2048]
                    N_EXPERTS,  #     const int num_experts,
                    INTERMEDIATE_SIZE,#     const int state_size,     // 2048
                    HIDDEN_SIZE#     const int output_size);   // 768
                  )
cl.finish()

C1 = tC.numpy()


cl.profiling(True)
for r in range(0, 100):
    #cl.finish()
    for i in range(N_LAYERS):
        # ocl_kernels.enqueue("mlp_down_ref", [1, HIDDEN_SIZE],[1, 32],
        #                             tBq4, tBs, tBz, tA, tC)
        # ocl_kernels.enqueue("mlp_down_n64", [1, HIDDEN_SIZE//2, SUBGROUP_NUM],[1, SUBGROUP_SIZE, SUBGROUP_NUM],
        #                                 tBq4, tBs, tBz, tA, tC)
        cm_kernels.enqueue("moe_down_kernel", globalSize, localSize,
                    tA,         #     int* input, 			// shape: f16[8, 1, 768]
                    all_index[i],      #     int* indexs,			// shape: u64[8] ??? i32
                    eweights,   #     BUFFERINDEX(eweights),	// shape: f16[8]
                    down_weight_ptrs, #     BUFFERINDEX(down_addrs),// shape: u64[128]
                    down_scales_ptrs, #     BUFFERINDEX(down_scales_addrs),	// shape: u64[128]
                    tC,         #     int* output,			// shape: f16[8, 1, 2048]
                    N_EXPERTS,  #     const int num_experts,
                    INTERMEDIATE_SIZE,#     const int state_size,     // 2048
                    HIDDEN_SIZE#     const int output_size);   // 768
                  )
    durs = cl.finish()
    Bsize = N_EXPERTS*N*K//2
    mean_ns = sum(durs)/len(durs)

    print(f" {kernel_name} {K=} {N=} {Bsize*1e-6:.3f} MB {mean_ns*1e-6: .3f} ms   BW: { Bsize/mean_ns : .2f} GB/s")


if not np.allclose(C, C1):
    print(f'{C[:, :8]}')
    print(f'{C1[:, :8]}')
    print(f'{C.shape=} {C1.shape=}')
else:
    print("================ PASSED ==================" , M, K, N)
print(f"INTERMEDIATE_SIZE={INTERMEDIATE_SIZE} HIDDEN_SIZE={HIDDEN_SIZE}")


