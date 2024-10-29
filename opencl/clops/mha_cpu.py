from . import cl
import numpy as np
from .utils import *

# reference torch-cpu implementation for debugging purpose
class MHA_cpu:
    def __init__(self, head_cnt_q, head_cnt_k, head_size, max_kv_len, use_ref=False, kv_block=256):
        '''
            kv-cache is allocated internally here
        '''
        self.head_cnt_q = head_cnt_q  # query heads
        self.head_cnt_k = head_cnt_k
        self.head_cnt_qkv = head_cnt_q + 2 * head_cnt_k
        self.S = head_size
        self.max_kv_len = max_kv_len
        self.head_group_size = head_cnt_q // head_cnt_k
        self.head_size_scaler = 1.0 / (head_size ** 0.5)
        self.use_ref = use_ref
        self.kv_block = kv_block
        self.sub_group_size = 16

    # reference torch-cpu implementation for debugging purpose
    def repeat_kv(self, hidden_states, n_rep: int):
        """
        This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
        num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
        """
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    def __call__(self, qkv, attention_mask):
        import torch, math
        qkv = to_torch(qkv)
        attention_mask = to_torch(attention_mask)
        if not hasattr(self, "kv_cache"):
            B, L1, S = qkv.shape
            self.kv_cache = torch.zeros(2, B, self.head_cnt_k, self.max_kv_len, self.S, dtype=torch.float16)

        hidden_size = self.head_cnt_q * self.S
        num_heads = self.head_cnt_q
        num_key_value_heads = self.head_cnt_k
        head_dim = self.S
        # https://github.com/huggingface/transformers/blob/cc3e4781854a52cf090ffde28d884a527dab6708/src/transformers/models/llama/modeling_llama.py#L331
        # query_states : B, L, (H*S)+(H*S)+(H*S)
        bsz, q_len, _ = qkv.size()

        q_size = self.head_cnt_q * self.S
        kv_size = self.head_cnt_k * self.S
        query_states = qkv[:,:,:q_size].view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
        key_states = qkv[:,:,q_size:q_size+kv_size].view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
        value_states = qkv[:,:,q_size+kv_size:].view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)

        # q/k/v states : [batch, nHead, q_len, head_dim]
        kv_seq_len = attention_mask.shape[-1]
        kv_seq_len0 = kv_seq_len - q_len

        # kv_cache [2 * n_layers, batch, n_head, max_kv_len, head_size]
        self.kv_cache[0, :, :, kv_seq_len0:kv_seq_len, :] = key_states
        self.kv_cache[1, :, :, kv_seq_len0:kv_seq_len, :] = value_states

        # us beam_idx to gather(reorder kv cache), skipped in greedy case
        key_states = self.kv_cache[0, :, :, :kv_seq_len, :]
        value_states = self.kv_cache[1, :, :, :kv_seq_len, :]

        num_key_value_groups = num_heads // num_key_value_heads

        key_states = self.repeat_kv(key_states, num_key_value_groups)
        value_states = self.repeat_kv(value_states, num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)

        # [B, H, q_len, kv_len] [B, kv_len]
        attn_weights = attn_weights + attention_mask[:, None, None, :]

        # apply causal mask, only the last q-token can access all kv-cache
        # [batch, num_heads, q_len ,kv_len] 
        for k in range(q_len-1):
            attn_weights[:, :, k, (k - q_len + 1):] = torch.finfo(torch.float32).min

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, hidden_size)
        return to_cl(attn_output)

