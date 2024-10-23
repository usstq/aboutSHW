import numpy as np
import sys, os
import argparse
import time

import math
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from functools import partial
from contextlib import nullcontext

import cl_ops

def rms_norm(input, weight, epsilon, name_suffix=''):
    input_dtype = input.dtype
    input = input.to(torch.float32)
    variance = input.pow(2).mean(-1, keepdim=True)
    input = input * torch.rsqrt(variance + epsilon)
    return weight * input.to(input_dtype)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class Layer:
    pass

class LlamaModel:
    def __init__(self, hf_model_id, rope_base = 10000):
        hf_model = AutoModelForCausalLM.from_pretrained(hf_model_id, trust_remote_code=True).to('cpu').eval()

        print(hf_model.config)
        #assert(hf_model.config.num_key_value_heads == hf_model.config.num_attention_heads)
        assert(hf_model.config.hidden_act in ['silu'])
        assert(hf_model.config.rope_scaling is None)
        
        self.hf_config = hf_model.config
        self.embed_tokens = partial(F.embedding, weight=hf_model.model.embed_tokens.weight)
        self.norm = partial(rms_norm, weight=hf_model.model.norm.weight, epsilon = hf_model.config.rms_norm_eps)
        # self.lm_head = partial(F.linear, weight=hf_model.lm_head.weight, bias=hf_model.lm_head.bias)
        self.lm_head = cl_ops.Linear(weight=hf_model.lm_head.weight, bias=hf_model.lm_head.bias)
        self.layers = []
        for l in hf_model.model.layers:
            d = Layer()
            d.input_layernorm = partial(rms_norm, weight=l.input_layernorm.weight, epsilon = hf_model.config.rms_norm_eps)
            # combine qkv : 
            qkv_weight = torch.cat([l.self_attn.q_proj.weight, l.self_attn.k_proj.weight, l.self_attn.v_proj.weight], dim=0)
            assert(l.self_attn.q_proj.bias == None)
            assert(l.self_attn.k_proj.bias == None)
            assert(l.self_attn.v_proj.bias == None)
            d.qkv_proj = cl_ops.Linear(weight=qkv_weight, bias=None)

            d.o_proj = cl_ops.Linear(weight=l.self_attn.o_proj.weight, bias=l.self_attn.o_proj.bias)
            d.post_attention_layernorm = partial(rms_norm, weight=l.post_attention_layernorm.weight, epsilon = hf_model.config.rms_norm_eps)
            d.gate_proj = cl_ops.Linear(weight=l.mlp.gate_proj.weight, bias=l.mlp.gate_proj.bias)
            d.up_proj = cl_ops.Linear(weight=l.mlp.up_proj.weight, bias=l.mlp.up_proj.bias)
            d.down_proj = cl_ops.Linear(weight=l.mlp.down_proj.weight, bias=l.mlp.down_proj.bias)
            d.id = len(self.layers)
            self.layers.append(d)
        self.rotary_dim = int(self.hf_config.hidden_size // self.hf_config.num_attention_heads)
        self.head_size = hf_model.config.hidden_size // hf_model.config.num_attention_heads
        self.inv_freq = 1.0 / (rope_base ** (torch.arange(0, self.rotary_dim, 2).float().to("cpu") / self.rotary_dim))
        self.rope = cl_ops.ROPE(self.inv_freq, self.rotary_dim,
                                hf_model.config.num_attention_heads,
                                hf_model.config.num_key_value_heads,
                                self.head_size)

    def mha(self, layer, qkv, kv_cache, attention_mask):
        layer_idx = layer.id
        rotary_dim = self.rotary_dim
        hidden_size = self.hf_config.hidden_size
        num_heads = self.hf_config.num_attention_heads
        num_key_value_heads = self.hf_config.num_key_value_heads
        
        head_dim = hidden_size//num_heads
        # https://github.com/huggingface/transformers/blob/cc3e4781854a52cf090ffde28d884a527dab6708/src/transformers/models/llama/modeling_llama.py#L331
        # query_states : B, L, (H*S)+(H*S)+(H*S)
        bsz, q_len, _ = qkv.size()

        q_size = self.head_size * num_heads
        kv_size = self.head_size * num_key_value_heads
        # derive total kv length from attn (has limitation)
        # apply_rotary_pos_emb to key_states/value_states
        kv_seq_len = attention_mask.shape[-1]
        kv_seq_len0 = kv_seq_len - q_len

        qkv = self.rope(qkv, kv_seq_len0)

        query_states = qkv[:,:,:q_size].view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
        key_states = qkv[:,:,q_size:q_size+kv_size].view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
        value_states = qkv[:,:,q_size+kv_size:].view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)

        # q/k/v states : [batch, nHead, q_len, head_dim]
        def rope_embedd(x):
            half_rotary_dim = rotary_dim//2
            for k in range(q_len):
                position_idx = kv_seq_len0 + k
                for i0 in range(half_rotary_dim):
                    i1 = i0 + half_rotary_dim
                    xita = (self.inv_freq[i0] * position_idx)
                    vcos = math.cos(xita)
                    vsin = math.sin(xita)
                    y0 = vcos * x[:, :, k, i0] - vsin * x[:, :, k, i1]
                    y1 = vsin * x[:, :, k, i0] + vcos * x[:, :, k, i1]
                    x[:, :, k, i0] = y0
                    x[:, :, k, i1] = y1

        #rope_embedd(query_states)
        #rope_embedd(key_states)

        # kv_cache [2 * n_layers, batch, n_head, max_kv_len, head_size]
        kv_cache[2*layer_idx + 0, :, :, kv_seq_len0:kv_seq_len, :] = key_states
        kv_cache[2*layer_idx + 1, :, :, kv_seq_len0:kv_seq_len, :] = value_states

        # us beam_idx to gather(reorder kv cache), skipped in greedy case
        key_states = kv_cache[2*layer_idx + 0, :, :, :kv_seq_len, :]
        value_states = kv_cache[2*layer_idx + 1, :, :, :kv_seq_len, :]

        num_key_value_groups = num_heads // num_key_value_heads

        key_states = repeat_kv(key_states, num_key_value_groups)
        value_states = repeat_kv(value_states, num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)

        attn_weights = attn_weights + attention_mask

        # apply causal mask, only the last q-token can access all kv-cache
        # [batch, num_heads, q_len ,kv_len] 
        for k in range(q_len-1):
            attn_weights[:, :, k, (k - q_len + 1):] = torch.finfo(torch.float32).min

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, hidden_size)
        return attn_output

    def forward_layer(self, layer, hidden_states, attn_mask, kv_cache):
        # layerNorm operation
        input_layernorm = layer.input_layernorm(hidden_states)

        qkv = layer.qkv_proj(input_layernorm)

        attn_output = self.mha(layer, qkv, kv_cache, attn_mask)

        attn_output = layer.o_proj(attn_output)

        attn_output = hidden_states + attn_output

        post_attention_layernorm = layer.post_attention_layernorm(attn_output)

        def mlp(states):
            gate_proj = layer.gate_proj(states)
            up_proj = layer.up_proj(states)
            mul = F.silu(gate_proj) * up_proj
            down_proj = layer.down_proj(mul)
            return down_proj

        mlp_output = mlp(post_attention_layernorm)
        # residual connection.
        output = attn_output + mlp_output
        return output

    def forward(self, input_ids, attn_mask, kv_cache):
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = self.forward_layer(layer, hidden_states, attn_mask, kv_cache)

        final_layernorm = self.norm(hidden_states)
        logits = self.lm_head(final_layernorm)
        return logits

def simple_pipeline(hf_model_path, prompt0):
    global inv_freq
    print(f"load Tokenizer from {hf_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.padding_side = "left"             # pad to left

    streamer = TextStreamer(tokenizer)


    print(f"load config/weight from HF model {hf_model_path} ...")
    model = LlamaModel(hf_model_path)

    batch_size = 1
    max_kv_len = 2048

    kv_cache = torch.zeros(model.hf_config.num_hidden_layers * 2, batch_size, model.hf_config.num_key_value_heads, max_kv_len, model.head_size, dtype=torch.float32)
    position_id = 0
    attn_mask = torch.zeros(batch_size, 0, dtype=torch.float32)

    new_tokens = 0
    do_trace = prompt0 is not None
    if do_trace:
        from viztracer import VizTracer

    with torch.no_grad(), VizTracer(output_file="optional.json") if do_trace else nullcontext() as tracer:
        while True:
            # attn_mask : [batch, query_len+past_len]
            print("\033[0;32m")
            if prompt0:
                prompt = prompt0
            else:
                try:
                    prompt = input(">")
                except EOFError:
                    break
            inputs = tokenizer(f"<|user|>{prompt}</s><|assistant|>", return_tensors="pt", padding=True, return_token_type_ids=False)
            #inputs = tokenizer(f"[INST] {prompt} [/INST]", return_tensors="pt", padding=True, return_token_type_ids=False)
            input_ids = inputs["input_ids"]
            # zero means valid, np.finfo(np.float32).min means invalid(padding-part)
            _amask = (1 - inputs["attention_mask"]) * torch.finfo(torch.float32).min
            attn_mask = torch.cat([attn_mask, _amask], dim=-1)

            t0 = time.time()
            # logits    : [batch, q_len, vocab_size]
            logits = model.forward(input_ids, attn_mask, kv_cache)

            # only the last token
            next_token_logits = logits[:, -1, :]
            next_tokens = torch.argmax(next_token_logits, dim=-1)

            print("\033[0;33m")
            t1 = time.time()
            while tokenizer.eos_token_id not in next_tokens:
                next_tokens = next_tokens.reshape(batch_size, 1)
                input_ids = next_tokens

                streamer.put(next_tokens)

                attn_mask = torch.cat([attn_mask, torch.zeros(batch_size, 1, dtype=torch.float32)], dim=-1)

                logits = model.forward(input_ids, attn_mask, kv_cache)
                next_tokens = torch.argmax(logits, dim=-1)
                new_tokens += 1
                if new_tokens > 8:
                    break
            t2 = time.time()
            streamer.put(next_tokens)
            streamer.end()
            if prompt0:
                print(f"\033[00m  {(t1-t0):.3f} s + [{(t2-t1)/new_tokens*1e3:.3f} ms/tok x {new_tokens} tokens]")
                break
            print("\033[00m")

# HF_TOKEN=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 python3 ./test_llama.py -p "What's Oxygen?"

if __name__ == "__main__":
    parser = argparse.ArgumentParser('')
    # meta-llama/Llama-2-7b-chat-hf
    # TinyLlama/TinyLlama-1.1B-Chat-v1.0
    # /mnt/llm_irs/models_original/TinyLlama/TinyLlama-1.1B-Chat-v1.0
    # /mnt/llm_irs/models_original/llama-2-7b-chat/pytorch
    parser.add_argument('-p', "--prompt0", type=str, default=None)
    parser.add_argument('--hf_model_path', type=str, nargs='?', default='/mnt/llm_irs/models_original/TinyLlama/TinyLlama-1.1B-Chat-v1.0')
    args = parser.parse_args()
    simple_pipeline(args.hf_model_path, args.prompt0)