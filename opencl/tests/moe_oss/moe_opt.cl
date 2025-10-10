
#if 0
// Scatter
#define HIDDEN_SIZE
#define TYPE  // half/float
#define TYPE_SIZE // 2/4
// 
#endif


#define TYPE half
#define TYPE_SIZE 2


__kernel void softmax_topk(
    const __global TYPE* input, // [input_batch, sort_in_num]
    __global uint* output_index, // [input_batch, TOP_K]
    __global TYPE* output // [input_batch, TOP_K]
) {
    // gws [batch, sort_in_num]
    const uint batch = (uint)get_global_id(0);
    const uint sort_index = (uint)get_global_id(1);
    const uint sort_cnt = (uint)get_global_size(1);

    input += batch * sort_cnt + sort_index;

    uint sort_position = 0;

    __local TYPE local_input[VALUE_NUM];
    __local TYPE local_output[TOP_K];
    __local uint local_index[TOP_K];

    TYPE in_value = as_half(intel_sub_group_block_read_us((const __global ushort*)(input)));
    local_input[sort_index] = in_value;
    barrier(CLK_LOCAL_MEM_FENCE);

    __attribute__((opencl_unroll_hint(8)))
    for(uint i = 0; i < sort_index; i++) {
        TYPE value = local_input[i];
        if(value >= in_value) {
            sort_position++;
        }
    }

    __attribute__((opencl_unroll_hint(8)))
    for(uint i = sort_index; i < sort_cnt; i++) {
        TYPE value = local_input[i];
        if(value > in_value) {
            sort_position++;
        }
    }
    if (sort_position < TOP_K) {
        local_output[sort_position] = in_value;
        local_index[sort_position] = sort_index;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(sort_position == 0) {
        float softmax_total = 1.0;
        TYPE max_v = local_output[0];
        local_output[0] = 1;
        for(uint i = 1; i < TOP_K; i++) {
            local_output[i] = native_exp(local_output[i] - max_v);
            softmax_total += local_output[i];
        }
        output_index += batch * TOP_K;
        output += batch * TOP_K;

        for(uint i = 0; i < TOP_K; i++) {
            output[i] = local_output[i]/softmax_total;
            output_index[i] = local_index[i];
        }
    }
}

__kernel void gather_2d_ref(
    const __global TYPE* src_tok,
    const __global TYPE* src_rweight,
    __global int * tok_index,
    __global int * top_index,
    __global TYPE* dst_tok,
    __global TYPE* dst_rweight) {

    int k = get_global_id(0);
    int off = get_global_id(1);
    int tok_idx = tok_index[k];

    src_tok += tok_idx * HIDDEN_SIZE;
    dst_tok += k * HIDDEN_SIZE;

    #if TYPE_SIZE == 2
        ushort value = intel_sub_group_block_read_us((const __global ushort *)(src_tok + off));
        intel_sub_group_block_write_us((__global ushort *)(dst_tok + off), value);
    #elif TYPE_SIZE == 4
        uint value = intel_sub_group_block_read((const __global uint *)(src_tok + off));
        intel_sub_group_block_write((__global uint *)(dst_tok + off), value);
    #else
        dst_tok[off] = src_tok[off];
    #endif

    if (off == 0) {
        int top_idx = top_index[k];
        dst_rweight[k] = src_rweight[top_idx];
    }
}

__kernel void index_add_(const __global TYPE* src_tok,
    __global int * tok_index,
    __global TYPE* dst_tok) {

    int k = get_global_id(0);
    int off = get_global_id(1);
    int tok_idx = tok_index[k];

    src_tok += k * HIDDEN_SIZE;
    dst_tok += tok_idx * HIDDEN_SIZE;

    #if TYPE_SIZE == 2
        half src_value = as_half(intel_sub_group_block_read_us((const __global ushort *)(src_tok + off)));
        half dst_value = as_half(intel_sub_group_block_read_us((const __global ushort *)(dst_tok + off)));
        half value = dst_value + src_value;
        intel_sub_group_block_write_us((__global ushort *)(dst_tok + off), as_ushort(value));
    #elif TYPE_SIZE == 4
        float src_value = as_float(intel_sub_group_block_read((const __global uint *)(src_tok + off)));
        float dst_value = as_float(intel_sub_group_block_read((const __global uint *)(dst_tok + off)));
        float value = dst_value + src_value;
        intel_sub_group_block_write_us((__global ushort *)(dst_tok + off), as_uint(value));
    #else
        dst_tok[off] += src_tok[off];
    #endif
}

// gws[batch, intermediate_size]
// lws[1, 16]
__kernel void gate_up_post_proc(
    const __global half* gate,
    const __global half* up,
    __global half* output) {

    int m = get_global_id(0);
    int n = get_global_id(1);

    int offset = m * INTERMEDIATE_SIZE + n;

    const half oss_alpha = -1.702;
    const half oss_limit = 7.0;
    const half oss_neg_limit = -7.0;

    half src_gate = as_half(intel_sub_group_block_read_us((const __global ushort *)(gate + offset)));
    half src_up = as_half(intel_sub_group_block_read_us((const __global ushort *)(up + offset)));

    if(src_up < oss_neg_limit) {
        src_up = oss_neg_limit  ;
    } else if(src_up > oss_limit) {
        src_up = oss_limit;
    }

    if(src_gate > oss_limit) {
        src_gate = oss_limit;
    }

    half value = src_gate * ( 1.0 / (1.0 + native_exp(oss_alpha * src_gate))) * (src_up + 1.0h);
    intel_sub_group_block_write_us((__global ushort *)(output + offset), as_ushort(value));
}
