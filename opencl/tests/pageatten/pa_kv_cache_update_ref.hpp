
/*
 * Copyright (c) 2020-2025, Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#include <cm/cm.h>
#include <cm/cmtl.h>

#ifndef ATTR
#define ATTR [[type("svmptr_t")]]
#define ATTR_BUF [[type("buffer_t")]]
#endif

constexpr uint wg_size = WG_SIZE;

extern "C" _GENX_MAIN_ void pa_kv_cache_update(
    const half* key [[type("svmptr_t")]],
    const half* value [[type("svmptr_t")]],
    const int32_t* past_lens [[type("svmptr_t")]],
    const int32_t* block_indices [[type("svmptr_t")]],
    const int32_t* block_indices_begins [[type("svmptr_t")]],
    const int32_t* subsequence_begins [[type("svmptr_t")]],
    half* key_cache [[type("svmptr_t")]],
    half* value_cache [[type("svmptr_t")]],    
    uint32_t key_pitch,
    uint32_t value_pitch,
    uint32_t batch_size_in_sequences) {
    // # key:   [batch_size_in_tokens, num_kv_heads * k_head_size]
    // # value  [batch_size_in_tokens, num_kv_heads * v_head_size]
    // # key_cache:   [num_blocks, num_heads, block_size, k_head_size]
    // # value_cache: [num_blocks, num_heads, block_size, v_head_size]
    // 
    // # past_lens: [sequences_num]
    // # subsequence_begins: [sequences_num + 1]
    // # block_indices: [used_blocks_num]
    // # block_indices_begins: [sequences_num + 1]

    // wg_count = aligned_to(batch_size_in_tokens, wg_size) // wg_size
    // # GWS [1, num_heads, wg_count * wg_size]
    // # LWS [1, 1, wg_size]

    const auto head_idx = cm_group_id(1);
    const auto wg_id = cm_group_id(2);
    const auto wg_local_id = cm_local_id(2);
    const auto local_size = cm_local_size(2);

    // static_assert(local_size == wg_size);

    // const uint token_idx = wg_id * local_size + wg_local_id;
    const uint token_idx = cm_global_id(2);

    // token_idx -> subsequence_idx
    if (token_idx >= subsequence_begins[batch_size_in_sequences]) return;
    uint subsequence_idx = 0;
    for (uint i = 0; i < batch_size_in_sequences; i++) {
        if (token_idx >= subsequence_begins[i] && token_idx < subsequence_begins[i + 1]) {
            subsequence_idx = i;
            break;
        }
    }
    // printf("wg:%d.%d, token_idx: %d, subsequence_idx: %d\n", wg_id, wg_local_id, token_idx, subsequence_idx);

    const uint subsequence_begin_idx = subsequence_begins[subsequence_idx];

    const uint past_len = past_lens[subsequence_idx];

    const uint current_block_idx = (past_len + token_idx - subsequence_begin_idx) / PAGED_ATTENTION_BLOCK_SIZE;
    const uint token_start_pos = (past_len + token_idx - subsequence_begin_idx) % PAGED_ATTENTION_BLOCK_SIZE;

    const uint block_offset = block_indices_begins[subsequence_idx] + current_block_idx;

    {
        uint block_k_base_offset = (block_indices[block_offset] * KV_HEADS_NUM + head_idx) * ADJUSTED_K_HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE;
        uint key_out_offset = block_k_base_offset + token_start_pos * K_HEAD_SIZE;
        uint key_in_offset = token_idx * key_pitch + head_idx * K_HEAD_SIZE;

        vector<half, K_HEAD_SIZE> key_data;
        key_data.format<int>() = cm_ptr_load<int, K_HEAD_SIZE / 2>((int*)key, key_in_offset * (int)sizeof(half));
        cm_ptr_store<int, K_HEAD_SIZE / 2>((int*)key_cache, key_out_offset * (int)sizeof(half), key_data.format<int>());
    }
    {
        uint block_v_base_offset = (block_indices[block_offset] * KV_HEADS_NUM + head_idx) * ADJUSTED_V_HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE;
        uint value_out_offset = block_v_base_offset + token_start_pos * V_HEAD_SIZE;
        uint value_in_offset = token_idx * value_pitch + head_idx * V_HEAD_SIZE;

        vector<half, V_HEAD_SIZE> value_data;
        value_data.format<int>() = cm_ptr_load<int, V_HEAD_SIZE / 2>((int*)value, value_in_offset * (int)sizeof(half));
        cm_ptr_store<int, V_HEAD_SIZE / 2>((int*)value_cache, value_out_offset * (int)sizeof(half), value_data.format<int>());
    }
}