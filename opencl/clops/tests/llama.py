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

import clops

# We design empty Layer on purpose, this make the network hierarchy flatten and easier/more clear to see & optimize
# although it's an object, but we prefer the whole network topology & logic to be in a more functional style
class Layer:
    pass

class LlamaModel:
    def __init__(self, hf_model_id, max_kv_len, quant_type, rope_base = 10000):
        hf_model = AutoModelForCausalLM.from_pretrained(hf_model_id, trust_remote_code=True).to('cpu').eval()

        print(hf_model.config)
        #assert(hf_model.config.num_key_value_heads == hf_model.config.num_attention_heads)
        assert(hf_model.config.hidden_act in ['silu'])
        assert(hf_model.config.rope_scaling is None)

        if quant_type == "f16":
            Linear = clops.Linear
        elif quant_type == "w4a":
            Linear = clops.Linear_woq_I4  # INT4 ASYN
        else:
            assert(f"quant_type {quant_type} is unexpected!")

        self.hf_config = hf_model.config
        self.embed_tokens = partial(F.embedding, weight=hf_model.model.embed_tokens.weight)
        self.norm = clops.RMSNorm(weight=hf_model.model.norm.weight, epsilon = hf_model.config.rms_norm_eps)
        # self.lm_head = partial(F.linear, weight=hf_model.lm_head.weight, bias=hf_model.lm_head.bias)
        self.lm_head = Linear(weight=hf_model.lm_head.weight, bias=hf_model.lm_head.bias)

        self.rotary_dim = int(self.hf_config.hidden_size // self.hf_config.num_attention_heads)
        self.head_size = hf_model.config.hidden_size // hf_model.config.num_attention_heads
        self.inv_freq = 1.0 / (rope_base ** (torch.arange(0, self.rotary_dim, 2).float().to("cpu") / self.rotary_dim))
        self.rope = clops.ROPE(self.inv_freq, self.rotary_dim,
                                hf_model.config.num_attention_heads,
                                hf_model.config.num_key_value_heads,
                                self.head_size)
        self.layers = []
        for l in hf_model.model.layers:
            d = Layer()
            d.input_layernorm = clops.RMSNorm(weight=l.input_layernorm.weight, epsilon = hf_model.config.rms_norm_eps)
            # combine qkv : 
            qkv_weight = torch.cat([l.self_attn.q_proj.weight, l.self_attn.k_proj.weight, l.self_attn.v_proj.weight], dim=0)
            assert(l.self_attn.q_proj.bias == None)
            assert(l.self_attn.k_proj.bias == None)
            assert(l.self_attn.v_proj.bias == None)
            d.qkv_proj = Linear(weight=qkv_weight, bias=None)

            d.o_proj = Linear(weight=l.self_attn.o_proj.weight, bias=l.self_attn.o_proj.bias)
            d.post_attention_layernorm = clops.RMSNorm(weight=l.post_attention_layernorm.weight, epsilon = hf_model.config.rms_norm_eps)
            d.gate_proj = Linear(weight=l.mlp.gate_proj.weight, bias=l.mlp.gate_proj.bias)
            d.up_proj = Linear(weight=l.mlp.up_proj.weight, bias=l.mlp.up_proj.bias)
            d.down_proj = Linear(weight=l.mlp.down_proj.weight, bias=l.mlp.down_proj.bias)
            d.id = len(self.layers)
            d.mha = clops.MHA(self.hf_config.num_attention_heads,
                              self.hf_config.num_key_value_heads,
                              self.head_size,
                              max_kv_len)

            self.layers.append(d)

    def forward_layer(self, layer, hidden_states, attn_mask):
        # layerNorm operation
        input_layernorm = layer.input_layernorm(hidden_states)

        qkv = layer.qkv_proj(input_layernorm)

        query_seq_len = qkv.shape[1]
        kv_seq_len = attn_mask.shape[-1]
        qkv = self.rope(qkv, kv_seq_len - query_seq_len)

        attn_output = layer.mha(qkv, attn_mask)
        attn_output = layer.o_proj(attn_output)

        attn_output += hidden_states

        post_attention_layernorm = layer.post_attention_layernorm(attn_output)

        def mlp(states):
            gate_proj = layer.gate_proj(states)
            up_proj = layer.up_proj(states)
            gate_proj.iSilu()
            gate_proj *= up_proj
            down_proj = layer.down_proj(gate_proj)
            return down_proj

        mlp_output = mlp(post_attention_layernorm)
        # residual connection.
        attn_output += mlp_output
        return attn_output

    def forward(self, input_ids, attn_mask):
        # embedding is done on CPU so far (to save GPU memory since embedding is memory-bounded)
        inputs_embeds = self.embed_tokens(input_ids)
        # the rest is done on GPU
        hidden_states = clops.to_cl(inputs_embeds.half())
        for layer in self.layers:
            hidden_states = self.forward_layer(layer, hidden_states, attn_mask)

        final_layernorm = self.norm(hidden_states)
        logits = self.lm_head(final_layernorm)
        return logits.torch().float()

def simple_pipeline(hf_model_path, prompt0, do_trace, max_new_tokens, max_kv_len, quant_type):
    global inv_freq
    print(f"load Tokenizer from {hf_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.padding_side = "left"             # pad to left

    streamer = TextStreamer(tokenizer)

    batch_size = 1
    position_id = 0
    attn_mask = torch.zeros(batch_size, 0, dtype=torch.float32)

    print(f"load config/weight from HF model {hf_model_path} ...")
    model = LlamaModel(hf_model_path, max_kv_len, quant_type)

    new_tokens = 0
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
            #inputs = tokenizer(f"Hi", return_tensors="pt", padding=True, return_token_type_ids=False)
            #inputs = tokenizer(f"[INST] {prompt} [/INST]", return_tensors="pt", padding=True, return_token_type_ids=False)
            input_ids = inputs["input_ids"]
            # zero means valid, np.finfo(np.float32).min means invalid(padding-part)
            _amask = (1 - inputs["attention_mask"]) * torch.finfo(torch.float32).min
            attn_mask = torch.cat([attn_mask, _amask], dim=-1)

            t0 = time.time()
            # logits    : [batch, q_len, vocab_size]
            logits = model.forward(input_ids, attn_mask)

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

                logits = model.forward(input_ids, attn_mask)
                next_tokens = torch.argmax(logits, dim=-1)
                new_tokens += 1
                if new_tokens > max_new_tokens:
                    break
            t2 = time.time()
            streamer.put(next_tokens)
            streamer.end()
            if prompt0:
                print(f"\033[00m  {(t1-t0):.3f} s + [{(t2-t1)/new_tokens*1e3:.3f} ms/tok x {new_tokens} tokens]")
                break
            print("\033[00m")

# HF_TOKEN=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 python3 ./test_llama.py -p "What's Oxygen?"
# cliloader -d -cdt --dump-dir ./dump/ python3 ./test_llama.py -p "What's Oxygen?"

if __name__ == "__main__":
    parser = argparse.ArgumentParser('')
    # meta-llama/Llama-2-7b-chat-hf
    # TinyLlama/TinyLlama-1.1B-Chat-v1.0
    # /mnt/llm_irs/models_original/TinyLlama/TinyLlama-1.1B-Chat-v1.0
    # /mnt/llm_irs/models_original/llama-2-7b-chat/pytorch
    parser.add_argument('-p', "--prompt0", type=str, default=None)
    parser.add_argument('-t', "--trace", action="store_true")
    parser.add_argument('-n', "--max_new_tokens", type=int, default=8)
    parser.add_argument('-c', "--max_kv_len", type=int, default=256)
    parser.add_argument('-q', "--quant_type", type=str, default="w4a", choices=['f16', 'w4a'])

    # /mnt/llm_irs/models_original/llama-2-7b-chat/pytorch/
    parser.add_argument('-hf', '--hf_model_path', type=str, nargs='?', default='/mnt/llm_irs/models_original/TinyLlama/TinyLlama-1.1B-Chat-v1.0')
    args = parser.parse_args()
    simple_pipeline(args.hf_model_path, args.prompt0, args.trace, args.max_new_tokens, args.max_kv_len, args.quant_type)