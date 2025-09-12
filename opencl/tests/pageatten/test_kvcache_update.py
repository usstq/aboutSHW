import os
import time
import math

from numpy import diff
import torch
import functools

from clops import cl
from clops import compare
from clops.utils import Colors

kv_cache_compression_enabled = 1

class pa_kvcache_update_cm:
    def __init__(self, num_kv_heads, k_head_size, v_head_size, block_size):
        self.num_kv_heads = num_kv_heads
        self.k_head_size = k_head_size
        self.v_head_size = v_head_size
        self.block_size = block_size
        self.wg_size = 16

        src = r'''#include "pa_kv_cache_update_ref.hpp"'''
        cwd = os.path.dirname(os.path.realpath(__file__))
        print(f"compiling {cwd} {num_kv_heads=} {k_head_size=} {v_head_size=} ...")

        if kv_cache_compression_enabled:
            adjusted_k_head_size = k_head_size + 4
            adjusted_v_head_size = v_head_size + 4
        else:
            adjusted_k_head_size = k_head_size
            adjusted_v_head_size = v_head_size

        jit_option = '-abortonspill -noschedule '
        self.kernels = cl.kernels(src,
                      (f' -cmc -Qxcm_jit_option="{jit_option}" -Qxcm_register_file_size=256 -mCM_printregusage -I{cwd}'
                      f" -DKV_HEADS_NUM={num_kv_heads}"
                      f" -DK_HEAD_SIZE={k_head_size}"
                      f" -DV_HEAD_SIZE={v_head_size}"
                      f" -DADJUSTED_K_HEAD_SIZE={adjusted_k_head_size}"
                      f" -DADJUSTED_V_HEAD_SIZE={adjusted_v_head_size}"
                      f" -DPAGED_ATTENTION_BLOCK_SIZE={self.block_size}"
                      f" -DWG_SIZE={self.wg_size}"
                      f" -DKV_CACHE_COMPRESSION_PER_TOKEN={kv_cache_compression_enabled}"
                      f" -mdump_asm -g2")
                    )

    def __call__(self, key:torch.Tensor,
                 value:torch.Tensor,
                key_cache:torch.Tensor,
                value_cache:torch.Tensor,
                past_lens:list,
                subsequence_begins:list,
                block_indices:list,
                block_indices_begins:list,
                n_repeats = 1):
        batch_size_in_tokens, _ = key.shape
        batch_size_in_sequences = len(past_lens)
        key_pitch = key.stride()[0]
        val_pitch = value.stride()[0]
        # print(f"============ {key.shape=} {key.is_contiguous()=}")
        # print(f"============ {value.shape=} {value.is_contiguous()=}")
        # print(f"============ {batch_size_in_tokens=} {batch_size_in_sequences=}")
        # print(f"============ {key.stride()=} {value.stride()=}")

        t_key = cl.tensor(key.to(torch.float16).detach().numpy())
        t_value = cl.tensor(value.to(torch.float16).detach().numpy())
        
        if kv_cache_compression_enabled:
            kv_cache_type = torch.uint8
        else:
            kv_cache_type = torch.float16
        t_key_cache = cl.tensor(key_cache.to(kv_cache_type).detach().numpy())
        t_value_cache = cl.tensor(value_cache.to(kv_cache_type).detach().numpy())

        t_block_indices=cl.tensor(torch.tensor(block_indices).to(torch.int32).detach().numpy())
        t_past_lens=cl.tensor(torch.tensor(past_lens).to(torch.int32).detach().numpy())
        t_block_indices_begins=cl.tensor(torch.tensor(block_indices_begins).to(torch.int32).detach().numpy())
        t_subsequence_begins=cl.tensor(torch.tensor(subsequence_begins).to(torch.int32).detach().numpy())

        wg_count = (batch_size_in_tokens + self.wg_size - 1) // self.wg_size
        GWS = [1, self.num_kv_heads, int(wg_count * self.wg_size)]
        LWS = [1, 1, self.wg_size]

        for i in range(0, n_repeats):
            print(f'{Colors.GREEN}calling pa_kv_cache_update {GWS=} {LWS=} {key_pitch=} {val_pitch=} {batch_size_in_sequences=} at {i}/{n_repeats} times {Colors.END}')
            self.kernels.enqueue("pa_kv_cache_update", GWS, LWS, t_key, t_value,
                            t_past_lens, t_block_indices, t_block_indices_begins, t_subsequence_begins, 
                            t_key_cache, t_value_cache,
                            key_pitch, val_pitch, batch_size_in_sequences)
            ns = cl.finish()
            for i, time_opt in enumerate(ns):
                print(f'(pa_kv_cache_update)TPUT_{i}:[{key.numel()=}]+[{value.numel()=}] {time_opt*1e-3:,.0f} us')
                if kv_cache_compression_enabled:
                    total_bytes = batch_size_in_tokens * self.num_kv_heads * (3 * self.k_head_size + 3 * self.v_head_size + 8)
                else:
                    total_bytes = batch_size_in_tokens * self.num_kv_heads * (4 * self.k_head_size + 4 * self.v_head_size)
                tput = total_bytes / time_opt
                print(f'(pa_kv_cache_update)TPUT_{i}:[{total_bytes*1e-6:,} MB] {tput/1e3:,.2f} GB/s')

        return t_key_cache.numpy(), t_value_cache.numpy()
                    
    @staticmethod
    @functools.cache
    def create_instance(num_kv_heads, k_head_size, v_head_size, block_size):
        return pa_kvcache_update_cm(num_kv_heads, k_head_size, v_head_size, block_size)

def test_pa_kv_cache_update(num_tokens:list, past_lens:list, num_kv_heads=1, k_head_size=64, v_head_size=64, block_size=16, check_perf=False):   
    batch_size_in_sequences = len(num_tokens)
    assert(batch_size_in_sequences == len(past_lens))

    # prepare page attention inputs
    key_data = []
    value_data = []
    subsequence_begins = []
    block_indices_begins = []    
    subsequence_begins.append(0)
    block_indices_begins.append(0)
    for i in range(batch_size_in_sequences):
        subsequence_length = num_tokens[i] + past_lens[i]

        k = torch.rand(subsequence_length, num_kv_heads*k_head_size).to(dtype=torch.float16)
        v = torch.rand(subsequence_length, num_kv_heads*v_head_size).to(dtype=torch.float16)
        key_data.append(k)
        value_data.append(v)
        
        subsequence_start_pos = subsequence_begins[i]
        subsequence_end_pos = subsequence_start_pos + num_tokens[i]
        subsequence_begins.append(subsequence_end_pos)

        required_blocks = (subsequence_length + block_size - 1) // block_size

        block_indices_start_pos = block_indices_begins[i];
        block_indices_end_pos = block_indices_start_pos + required_blocks;
        block_indices_begins.append(block_indices_end_pos);

    # simulate random block allocation
    num_blocks = block_indices_begins[-1]
    block_indices = torch.arange(num_blocks)
    perm_idx = torch.randperm(block_indices.shape[0])
    inv_per_idx = torch.argsort(perm_idx)
    block_indices = block_indices[inv_per_idx]
    # print(f'{Colors.BLUE} ============ {subsequence_begins=} {Colors.END}')
    # print(f'{Colors.BLUE} ============ {block_indices_begins=} {Colors.END}')
    # print(f'{Colors.BLUE} ============ {block_indices=} {Colors.END}')

    # generate key / value inputs
    def get_kv_input(num_kv_heads, head_size, input_data):
        batch_size_in_tokens = subsequence_begins[-1]
        mem = torch.zeros(batch_size_in_tokens, num_kv_heads*head_size).to(torch.float16)
        for i in range(batch_size_in_sequences):
            mem[subsequence_begins[i] : subsequence_begins[i+1], :] = input_data[i][past_lens[i]:, :]
        return mem
    key = get_kv_input(num_kv_heads, k_head_size, key_data)
    value = get_kv_input(num_kv_heads, v_head_size, value_data)
    # print(f'{Colors.BLUE} ============ {key.shape=} {key.is_contiguous()=}" {Colors.END}')
    # print(f'{Colors.BLUE} ============ {value.shape=} {value.is_contiguous()=}" {Colors.END}')
    # print(f'{Colors.BLUE} {key_data=} {Colors.END}')
    # print(f'{Colors.BLUE} {key=} {Colors.END}')
    # print(f'{Colors.BLUE} {value_data=} {Colors.END}')
    # print(f'{Colors.BLUE} {value=} {Colors.END}')

    def round_to_even(tensor):
        rounded = torch.floor(tensor + 0.5)
        adjustment = (rounded % 2 != 0) & (torch.abs(tensor - rounded) == 0.5000)
        adjustment = adjustment | (rounded > 255)  # also handle overflow
        return rounded - adjustment.to(rounded.dtype)

    def quant_per_token(kv):
        blk_num, kv_heads, blksz, *_ = kv.shape
        kv_max = kv.amax(dim=-1, keepdim = True).to(dtype=torch.float)
        kv_min = kv.amin(dim=-1, keepdim = True).to(dtype=torch.float)
        qrange = (kv_max - kv_min).to(dtype=torch.float)

        U8_MAX = torch.tensor(255.0, dtype=torch.float)
        U8_MIN = torch.tensor(0.0, dtype=torch.float)
        U8_RANGE = (U8_MAX - U8_MIN).to(dtype=torch.float)
        kv_scale = ((U8_RANGE)/qrange).to(dtype=torch.float)
        kv_scale_div = (1.0/kv_scale).to(dtype=torch.float)
        kv_zp = ((0.0-kv_min)*kv_scale+U8_MIN).to(dtype=torch.float)

        # kv_u8 = torch.round((kv*kv_scale+kv_zp)).to(dtype=torch.uint8).reshape(blk_num,kv_heads,-1)
        kv_u8 = round_to_even((kv*kv_scale+kv_zp)).to(dtype=torch.uint8).reshape(blk_num,kv_heads,-1)

        # torch.set_printoptions(precision=6)
        # print("kv_fp16:\n", kv.reshape(blk_num, kv_heads, blksz, k_head_size))
        # print("kv_max:\n", kv_max.reshape(blk_num, kv_heads, -1))
        # print("kv_min:\n", kv_min.reshape(blk_num, kv_heads, -1))
        # print("kv_scale:\n", kv_scale.reshape(blk_num, kv_heads, -1))
        # print("kv_scale_div:\n", kv_scale_div.reshape(blk_num, kv_heads, -1))
        # print("kv_zp:\n", kv_zp.reshape(blk_num, kv_heads, -1))
        # print("kv_quant_fp16:\n", (kv*kv_scale+kv_zp).reshape(blk_num, kv_heads, blksz, k_head_size))

        # print("quant_scale =", (1.0/kv_scale).reshape(blk_num,kv_heads,-1))
        # print("quant_zp    =", kv_zp.reshape(blk_num,kv_heads,-1))

        dq_scale = kv_scale_div.to(dtype=torch.half).view(dtype=torch.uint8).reshape(blk_num,kv_heads,-1)
        # kv_scale = ((U8_RANGE)/qrange).to(dtype=torch.half)
        # dq_scale = (kv_scale).view(dtype=torch.uint8).reshape(blk_num,kv_heads,-1)
        kv_zp = kv_zp.to(dtype=torch.half).view(dtype=torch.uint8).reshape(blk_num,kv_heads,-1)
        if blksz < block_size:
            kv_pad = torch.zeros(blk_num, kv_heads, (block_size - blksz)*k_head_size).to(dtype=torch.uint8)
            kv_u8 = torch.cat((kv_u8, kv_pad), dim=-1)
            scale_zp_pad = torch.zeros(blk_num, kv_heads, (block_size - blksz)*2).to(dtype=torch.uint8)
            dq_scale = torch.cat((dq_scale, scale_zp_pad), dim=-1)
            kv_zp = torch.cat((kv_zp, scale_zp_pad), dim=-1)

        # print("dq_scale: ", dq_scale)
        # print("kz_zp: ", kv_zp)
        return torch.concat((kv_u8, dq_scale, kv_zp), dim=-1)

    # generate key_cache / value_cache
    # input_data list of torch.Tensor with shape [subsequence_length, num_kv_heads, kv_head_size] for each sequence
    def get_kv_cache(num_blocks, block_size, num_kv_heads, head_size, input_data, skip_input = True):
        cache_data = torch.zeros(num_blocks, block_size, num_kv_heads*head_size).to(torch.float16)
        for i in range(batch_size_in_sequences):
            process_len = past_lens[i] if skip_input else past_lens[i] + num_tokens[i]
            if process_len > 0:
                blocks_num = (process_len + block_size - 1) // block_size
                for block_idx in range(blocks_num):
                    last_token_idx = process_len % block_size if block_idx == blocks_num -1 else block_size
                    if last_token_idx == 0: last_token_idx = block_size
                    # print(f'{Colors.RED} {block_idx=} {blocks_num=} {process_len=} {last_token_idx=} {Colors.END}')
                    for token_idx in range(last_token_idx):
                        input_token_offset = block_idx * block_size + token_idx
                        block_pos = block_indices[block_indices_begins[i] + block_idx]
                        cache_data[block_pos, token_idx, :] = input_data[i][input_token_offset, :]
        return cache_data.reshape(num_blocks, block_size, num_kv_heads, head_size).transpose(1, 2).contiguous()
    
    def get_kv_cache_u8(num_blocks, block_size, num_kv_heads, head_size, input_data, skip_input = True):
        cache_data = torch.zeros(num_blocks, num_kv_heads, block_size * (head_size + 4)).to(torch.uint8)
        for i in range(batch_size_in_sequences):
            process_len = past_lens[i] if skip_input else past_lens[i] + num_tokens[i]
            if process_len > 0:
                blocks_num = (process_len + block_size - 1) // block_size
                for block_idx in range(blocks_num):
                    last_token_idx = process_len % block_size if block_idx == blocks_num -1 else block_size
                    if last_token_idx == 0: last_token_idx = block_size
                    # print(f'{Colors.RED} {block_idx=} {blocks_num=} {process_len=} {last_token_idx=} {Colors.END}')
                    for h in range(num_kv_heads):
                        # input_data[seq_num][token_num, head_size * kv_head_num]
                        token_start_idx = block_idx * block_size
                        token_end_idx = token_start_idx + last_token_idx
                        input_block_per_head = input_data[i][token_start_idx:token_end_idx, h*head_size:(h+1)*head_size].reshape(1, 1, -1, head_size)
                        input_block_per_head_q = quant_per_token(input_block_per_head).reshape(-1)

                        # print()
                        # print(f'head_idx = {h} token_start_idx = {token_start_idx} token_end_idx = {token_end_idx} last_token_idx = {last_token_idx}')
                        # print('input_block_per_head.shape = {input_block_per_head.shape}')
                        # print('input_block_per_head = ',input_block_per_head)
                        # print('input_block_per_head_q.shape = ',input_block_per_head_q.reshape(1,1,-1,head_size).shape)
                        # print('input_block_per_head_q = ',input_block_per_head_q.reshape(1,1,-1,head_size))

                        block_pos = block_indices[block_indices_begins[i] + block_idx]
                        cache_data[block_pos, h, :] = input_block_per_head_q
        # if skip_input == False:
        #     print("cache_data =", cache_data)
        return cache_data
    
    
    def print_kv_cache_u8(kv_cache_u8, blk_size, kv_head_size, name="key_cache"):
        blk_num, kv_heads, kv_block_bytes_per_head = kv_cache_u8.shape
        print("name =", name, ",blk_num =", blk_num, ",kv_heads =", kv_heads, ",blk_size =", blk_size, ",kv_head_size =", kv_head_size)
        for b in range(blk_num):
            for h in range(kv_heads):
                print(f'blk={b} head={h}')
                block_head_data = kv_cache_u8[b,h,:blk_size * kv_head_size].reshape(blk_size, kv_head_size)
                block_head_scale = kv_cache_u8[b,h,blk_size * kv_head_size : blk_size * kv_head_size + blk_size * 2].reshape(blk_size, 2).view(dtype=torch.float16)
                block_head_zp = kv_cache_u8[b,h,blk_size * kv_head_size + blk_size * 2 : ].reshape(blk_size, 2).view(dtype=torch.float16)
                print('data: shape = ', block_head_data.shape, "\n", block_head_data)
                print('scale: shape = ', block_head_scale.shape, '\n', block_head_scale.reshape(1,blk_size))
                print('zp: shape = ', block_head_zp.shape, '\n', block_head_zp.reshape(1,blk_size))

                # block_head_data_f16 = block_head_data.to(dtype=torch.float16)
                # for i in range(blk_size):
                #     block_head_data_f16[i,:] = (block_head_data_f16[i,:] - block_head_zp[i,0]) * block_head_scale[i,0]
                # print('dequant_data_f16: shape = ', block_head_data_f16.shape, '\n', block_head_data_f16)

    if kv_cache_compression_enabled:
        key_cache = get_kv_cache_u8(num_blocks, block_size, num_kv_heads, k_head_size, key_data)
        value_cache = get_kv_cache_u8(num_blocks, block_size, num_kv_heads, v_head_size, value_data)

        # generate reference key/value cache
        key_cache_ref = get_kv_cache_u8(num_blocks, block_size, num_kv_heads, k_head_size, key_data, False)
        value_cache_ref = get_kv_cache_u8(num_blocks, block_size, num_kv_heads, v_head_size, value_data, False)

        # print("key_f16 = ", key_data)
        # print_kv_cache_u8(key_cache_ref, block_size, k_head_size, "key_cache_ref")
        # print_kv_cache_u8(value_cache_ref, block_size, v_head_size, "value_cache_ref")
    else:
        key_cache = get_kv_cache(num_blocks, block_size, num_kv_heads, k_head_size, key_data)
        value_cache = get_kv_cache(num_blocks, block_size, num_kv_heads, v_head_size, value_data)
        # print(f'{Colors.BLUE} ============ {key_cache.shape=} {key_cache.is_contiguous()=} {Colors.END}')
        # print(f'{Colors.BLUE} ============ {value_cache.shape=} {value_cache.is_contiguous()=} {Colors.END}')
        # print(f'{Colors.BLUE} {key_cache=} {Colors.END}')
        # print(f'{Colors.BLUE} {value_cache=} {Colors.END}')

        # generate reference key/value cache
        key_cache_ref = get_kv_cache(num_blocks, block_size, num_kv_heads, k_head_size, key_data, False)
        value_cache_ref = get_kv_cache(num_blocks, block_size, num_kv_heads, v_head_size, value_data, False)
        # print(f'{Colors.BLUE} ============ {key_cache_ref.shape=} {key_cache_ref.is_contiguous()=} {Colors.END}')
        # print(f'{Colors.BLUE} ============ {value_cache_ref.shape=} {value_cache_ref.is_contiguous()=} {Colors.END}')
        # print(f'{Colors.BLUE} {key_cache_ref=} {Colors.END}')
        # print(f'{Colors.BLUE} {value_cache_ref=} {Colors.END}')
    
    # opt
    pa_cm = pa_kvcache_update_cm.create_instance(num_kv_heads, k_head_size, v_head_size, block_size)
    n_repeats = 20 if check_perf else 1
    out_key_cache, out_value_cache = pa_cm(key, value, key_cache, value_cache, past_lens, subsequence_begins, block_indices, block_indices_begins, n_repeats)

    # if kv_cache_compression_enabled:
    #     out_key_cache = torch.tensor(out_key_cache)
    #     out_value_cache = torch.tensor(out_value_cache)
    #     print(f'{Colors.BLUE} ============ {out_key_cache.shape=} {out_key_cache.dtype} {Colors.END}')
    #     print("out_key_cache: value = ", out_key_cache[:,:,:block_size * k_head_size].reshape(num_blocks, num_kv_heads, block_size, k_head_size))
    #     print("out_key_cache: scale = ", out_key_cache[:,:,block_size * k_head_size : block_size * k_head_size + block_size * 2].reshape(num_blocks, num_kv_heads, 2 * block_size).view(dtype=torch.float16))
    #     print("out_key_cache: zp = ", out_key_cache[:,:,block_size * k_head_size + block_size * 2 : block_size * k_head_size + block_size * 4].reshape(num_blocks, num_kv_heads, 2 * block_size).view(dtype=torch.float16))
    #     print("out_value_cache = ", out_value_cache[:,:,:block_size * v_head_size].reshape(num_blocks, num_kv_heads, block_size, v_head_size))
    #     out_key_cache = out_key_cache.detach().numpy()
    #     out_value_cache = out_value_cache.detach().numpy()

    # print(f"key_cache_ref = \n{key_cache_ref.reshape(num_blocks, num_kv_heads, block_size, k_head_size + 4)}")
    # print(f"out_key_cache = \n{out_key_cache.reshape(num_blocks, num_kv_heads, block_size, k_head_size + 4)}")

    if kv_cache_compression_enabled:
        out_key_cache=torch.tensor(out_key_cache).to(dtype=torch.uint8)
        out_value_cache=torch.tensor(out_value_cache).to(dtype=torch.uint8)
        compare(key_cache_ref[:,:,:block_size * k_head_size].to(dtype=torch.int).detach().numpy(), out_key_cache[:,:,:block_size * k_head_size].to(dtype=torch.int).detach().numpy(),1)
        compare(key_cache_ref[:,:,block_size * k_head_size :].view(dtype=torch.half).detach().numpy(), out_key_cache[:,:,block_size * k_head_size : ].view(dtype=torch.half).detach().numpy(),1e-3)

        compare(value_cache_ref[:,:,:block_size * k_head_size].to(dtype=torch.int).detach().numpy(), out_value_cache[:,:,:block_size * k_head_size].to(dtype=torch.int).detach().numpy(),1)
        compare(value_cache_ref[:,:,block_size * k_head_size :].view(dtype=torch.half).detach().numpy(), out_value_cache[:,:,block_size * k_head_size : ].view(dtype=torch.half).detach().numpy(),1e-3)
    else:
        compare(key_cache_ref.detach().numpy(), out_key_cache)
        compare(value_cache_ref.detach().numpy(), out_value_cache)
    print(f'{Colors.GREEN}kv_cache_update passed{Colors.END}')

if __name__ == "__main__":
    torch.manual_seed(3)
    torch.set_printoptions(linewidth=1024)
    
    cl.profiling(True)
    
    # test_pa_kv_cache_update([1024, 16, 17], [16, 0, 1])
    test_pa_kv_cache_update([32*1024], [0], num_kv_heads=8, k_head_size=128, v_head_size=128, block_size=256, check_perf=True)
    test_pa_kv_cache_update([64*1024], [0], num_kv_heads=8, k_head_size=128, v_head_size=128, block_size=256, check_perf=True)
    test_pa_kv_cache_update([128*1024], [0], num_kv_heads=8, k_head_size=128, v_head_size=128, block_size=256, check_perf=True)
    test_pa_kv_cache_update([32*1024], [4*1024], num_kv_heads=8, k_head_size=128, v_head_size=128, block_size=256, check_perf=True)
    test_pa_kv_cache_update([128*1024], [1*1024], num_kv_heads=8, k_head_size=128, v_head_size=128, block_size=256, check_perf=True)
    
    test_pa_kv_cache_update([1024], [0], num_kv_heads=8, k_head_size=128, v_head_size=128, block_size=256, check_perf=False)
    # test_pa_kv_cache_update([1], [0], num_kv_heads=8, k_head_size=128, v_head_size=128, block_size=256, check_perf=False)
    # test_pa_kv_cache_update([1024], [0], num_kv_heads=2, k_head_size=16, v_head_size=16, block_size=32, check_perf=False)
    # test_pa_kv_cache_update([129], [0], num_kv_heads=2, k_head_size=64, v_head_size=64, block_size=16, check_perf=True)