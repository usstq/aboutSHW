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
from tqdm import tqdm
import pickle

# We design empty Layer on purpose, this make the network hierarchy flatten and easier/more clear to see & optimize
# although it's an object, but we prefer the whole network topology & logic to be in a more functional style
class Layer:
    pass

class LlamaLikeModel:
    def __init__(self) -> None:
        super(LlamaLikeModel, self).__init__()    

    def load_from_hf(self, hf_model_id, max_kv_len, quant_type, rope_base = 1000000):
        print(f"loading {hf_model_id}...")
        hf_model = AutoModelForCausalLM.from_pretrained(hf_model_id, trust_remote_code=True).to('cpu', dtype=torch.float16).eval()
        print(hf_model.config)
        
        is_qwen2 = "Qwen2ForCausalLM" in hf_model.config.architectures
        is_llama = "LlamaForCausalLM" in hf_model.config.architectures
        
        if is_qwen2:
            assert(hf_model.config.hidden_act in ['silu'])
            #assert(hf_model.config.rope_scaling is None)
            assert(hf_model.config.use_sliding_window == False)
            assert(hf_model.config.rope_theta == 1000000.0)
            rope_base = hf_model.config.rope_theta
        elif is_llama:
            assert(hf_model.config.hidden_act in ['silu'])
            assert(hf_model.config.rope_scaling is None)
            rope_base = 10000
        else:
            raise Exception(f"Unknown model architectures {hf_model.config.architectures}")

        Linear = getattr(clops, f"Linear_{quant_type}")
        MHA = clops.MHA
        #MHA = clops.MHA_cpu
        ROPE = clops.ROPE
        print(f"converting & loading model into GPGPU : {hf_model_id} ...")
        self.hf_config = hf_model.config
        # self.embed_tokens = partial(F.embedding, weight=hf_model.model.embed_tokens.weight)
        self.embed_tokens = clops.Embedding(weight=hf_model.model.embed_tokens.weight)
        self.norm = clops.RMSNorm(weight=hf_model.model.norm.weight, epsilon = hf_model.config.rms_norm_eps)
        # self.lm_head = partial(F.linear, weight=hf_model.lm_head.weight, bias=hf_model.lm_head.bias)
        self.lm_head = Linear(weight=hf_model.lm_head.weight, bias=hf_model.lm_head.bias)

        self.rotary_dim = int(self.hf_config.hidden_size // self.hf_config.num_attention_heads)
        self.head_size = hf_model.config.hidden_size // hf_model.config.num_attention_heads
        self.inv_freq = 1.0 / (rope_base ** (torch.arange(0, self.rotary_dim, 2).float().to("cpu") / self.rotary_dim))
        self.rope = ROPE(self.inv_freq, self.rotary_dim,
                         hf_model.config.num_attention_heads,
                         hf_model.config.num_key_value_heads,
                         self.head_size)
        self.layers = []
        for l in tqdm(hf_model.model.layers):
            d = Layer()
            d.id = len(self.layers)

            d.input_layernorm = clops.RMSNorm(weight=l.input_layernorm.weight, epsilon = hf_model.config.rms_norm_eps)
            # combine qkv : 
            qkv_weight = torch.cat([l.self_attn.q_proj.weight, l.self_attn.k_proj.weight, l.self_attn.v_proj.weight], dim=0)
            
            if l.self_attn.q_proj.bias is None:
                assert(l.self_attn.k_proj.bias is None)
                assert(l.self_attn.v_proj.bias is None)
                qkv_bias = None
            else:
                qkv_bias = torch.cat([l.self_attn.q_proj.bias, l.self_attn.k_proj.bias, l.self_attn.v_proj.bias], dim=0)
            d.qkv_proj = Linear(weight=qkv_weight, bias=qkv_bias)

            d.o_proj = Linear(weight=l.self_attn.o_proj.weight, bias=l.self_attn.o_proj.bias)
            d.post_attention_layernorm = clops.RMSNorm(weight=l.post_attention_layernorm.weight, epsilon = hf_model.config.rms_norm_eps)
            
            if Linear == clops.Linear_f16xmx or Linear == clops.Linear_w4x:
                assert l.mlp.gate_proj.bias is None
                assert l.mlp.up_proj.bias is None
                d.gate_up_proj = Linear(weight=l.mlp.gate_proj.weight, bias=None,
                                        weight_up = l.mlp.up_proj.weight)
            else:
                d.gate_up_proj = None
                dq_per_token = True
                d.gate_proj = Linear(weight=l.mlp.gate_proj.weight, bias=l.mlp.gate_proj.bias, dq_per_token=dq_per_token)
                d.up_proj = Linear(weight=l.mlp.up_proj.weight, bias=l.mlp.up_proj.bias, dq_per_token=dq_per_token)
            d.down_proj = Linear(weight=l.mlp.down_proj.weight, bias=l.mlp.down_proj.bias)
            d.is_last = False
            d.mha = MHA(self.hf_config.num_attention_heads,
                              self.hf_config.num_key_value_heads,
                              self.head_size,
                              max_kv_len)

            self.layers.append(d)
        self.layers[-1].is_last = True
        del hf_model
        import gc
        gc.collect()

    def forward_layer(self, layer, hidden_states, attn_mask):
        # layerNorm operation
        input_layernorm = layer.input_layernorm(hidden_states)
        qkv = layer.qkv_proj(input_layernorm)

        query_seq_len = qkv.shape[1]
        kv_seq_len = attn_mask.shape[-1]
        qkv = self.rope(qkv, kv_seq_len - query_seq_len)

        attn_output = layer.mha(qkv, attn_mask)

        # slice the last token for first-token [batch, query_seq_len, hidden_states]
        if layer.is_last and query_seq_len > 1:
            attn_output = attn_output.slice(1, query_seq_len-1, query_seq_len)
            hidden_states = hidden_states.slice(1, query_seq_len-1, query_seq_len)

        attn_output = layer.o_proj(attn_output)

        attn_output += hidden_states
        post_attention_layernorm = layer.post_attention_layernorm(attn_output)

        def mlp(states):
            if layer.gate_up_proj:
                gate_proj = layer.gate_up_proj(states)
            else:
                gate_proj = layer.gate_proj(states)
                up_proj = layer.up_proj(states)
                gate_proj.iSilu()
                gate_proj *= up_proj
            down_proj = layer.down_proj(gate_proj)
            return down_proj

        mlp_output = mlp(post_attention_layernorm)
        attn_output += mlp_output
        return attn_output

    def forward(self, input_ids, attn_mask):
        # embedding is done on CPU so far (to save GPU memory since embedding is memory-bounded)
        cl_input_ids = clops.to_cl(input_ids.int())
        hidden_states = self.embed_tokens(cl_input_ids)
        for layer in self.layers:
            hidden_states = self.forward_layer(layer, hidden_states, attn_mask)

        final_layernorm = self.norm(hidden_states)
        logits = self.lm_head(final_layernorm)
        return logits.torch().float()

    def load(self, path) -> None:
        print(f"loading model from {path}...")
        with open(path, 'rb') as f:
            tmp_dict = pickle.load(f)
        self.__dict__.update(tmp_dict)

    def save(self, path):
        print(f"saving model to {path}...")
        with open(path, 'wb') as f:
            pickle.dump(self.__dict__, f)

def simple_pipeline(hf_model_path,
                    prompt0,
                    do_trace,
                    max_new_tokens,
                    max_kv_len,
                    quant_type,
                    repeat,
                    save_path,
                    load_path):
    print(f"load Tokenizer from {hf_model_path}...")
    if load_path is None:
        tokenizer = AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(load_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.padding_side = "left"             # pad to left

    streamer = TextStreamer(tokenizer)

    batch_size = len(prompt0)

    print(prompt0)
    # overwrite max_kv_len according to [prompt length + max_new_tokens]
    if prompt0:
        inputs = tokenizer(prompt0, return_tensors="pt", padding=True, return_token_type_ids=False)
        input_ids = inputs["input_ids"]
        max_kv_len = (input_ids.shape[-1] + max_new_tokens + 63 )//64 * 64

    model = LlamaLikeModel()
    if load_path is None:
        model.load_from_hf(hf_model_path, max_kv_len, quant_type)
    else:
        model.load(os.path.join(load_path, "clops.pickle"))
    print(f" batch_size, max_kv_len = {batch_size}, {max_kv_len} ")

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        model.save(os.path.join(save_path, "clops.pickle"))
        tokenizer.save_pretrained(save_path, trust_remote_code=True)
        print(f"model {hf_model_path} saved to {save_path}.")
        return

    if do_trace:
        from viztracer import VizTracer

    with torch.no_grad(), VizTracer(output_file="optional.json") if do_trace else nullcontext() as tracer:
        for r in range(repeat):
            new_tokens = 0
            gen_tokens = torch.zeros(batch_size, 0, dtype=torch.int64)
            attn_mask = torch.zeros(batch_size, 0, dtype=torch.float32)
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
                #inputs = tokenizer(f"<|user|>{prompt}</s><|assistant|>", return_tensors="pt", padding=True, return_token_type_ids=False)
                #inputs = tokenizer(f"Hi", return_tensors="pt", padding=True, return_token_type_ids=False)
                
                #inputs = tokenizer("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n1+1=?<|im_end|>\n<|im_start|>assistant",return_tensors="pt")
                inputs = tokenizer(f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant", return_tensors="pt", padding=True, return_token_type_ids=False)
                #inputs = tokenizer(f'''<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat's Oxygen?<|im_end|>\n<|im_start|>assistant''',return_tensors="pt", padding=True, return_token_type_ids=False)
                # inputs = tokenizer(f"[INST] {prompt} [/INST]", return_tensors="pt", padding=True, return_token_type_ids=False)
                #inputs = tokenizer(prompt, return_tensors="pt", padding=True, return_token_type_ids=False)

                input_ids = inputs["input_ids"]
                _amask = (1 - inputs["attention_mask"]) * torch.finfo(torch.float32).min
                attn_mask = torch.cat([attn_mask, _amask], dim=-1) # [batch, query_len+past_len] 0:valid, -max: invalid

                prompt_tok_length = input_ids.shape[-1]
                t0 = time.time()
                logits = model.forward(input_ids, attn_mask) # [batch, q_len, vocab_size]
                next_token_logits = logits[:, -1, :]         # only the last token
                next_tokens = torch.argmax(next_token_logits, dim=-1)
                print("\033[0;33m")
                t1 = time.time()
                while tokenizer.eos_token_id not in next_tokens:
                    next_tokens = next_tokens.reshape(batch_size, 1)
                    input_ids = next_tokens
                    if batch_size == 1:
                        streamer.put(next_tokens[0,:])
                    else:
                        gen_tokens = torch.cat([gen_tokens, next_tokens], dim=1)

                    new_tokens += 1
                    if new_tokens >= max_new_tokens:
                        break

                    attn_mask = torch.cat([attn_mask, torch.zeros(batch_size, 1, dtype=torch.float32)], dim=-1)
                    logits = model.forward(input_ids, attn_mask)
                    next_tokens = torch.argmax(logits, dim=-1)
                t2 = time.time()
                if batch_size == 1:
                    streamer.put(next_tokens[0,:])
                streamer.end()
                if prompt0:
                    if (gen_tokens.numel() > 0):
                        print("\033[00m")
                        for i in range(batch_size):
                            out = tokenizer.decode(gen_tokens[i,:], skip_special_tokens=True)
                            print(f"[{i}] \033[0;32m{prompt0[i]}\033[0;33m{out}\033[00m")
                    print(f"\033[00m {batch_size} x ({prompt_tok_length} + {new_tokens}) tokens,  [{t1-t0:.3f} sec + {t2-t1:.3f} sec]     [{batch_size*prompt_tok_length/(t1-t0):.3f} tok/sec + {(t2-t1)/(batch_size*new_tokens)*1e3:.3f} ms/tok]")
                    break
                print("\033[00m")

if __name__ == "__main__":
    parser = argparse.ArgumentParser('')
    parser.add_argument('-p', "--prompt0", type=str, default="What's Oxygen?")
    parser.add_argument('-x', "--prompt_tokens", type=str, default=None)
    parser.add_argument('-t', "--trace", action="store_true")
    parser.add_argument('-n', "--max_new_tokens", type=int, default=32)
    parser.add_argument('-c', "--max_kv_len", type=int, default=256)
    parser.add_argument('-r', "--repeat", type=int, default=1)
    parser.add_argument('-q', "--quant_type", type=str, default="onednn", choices=['f16', 'f16b1', 'w4a', 'w4a_cpu', 'f16xmx', 'w4x', 'onednn'])

    parser.add_argument('-hf', '--hf_model_path', type=str, nargs='?', default='/mnt/llm_irs/models_original/Qwen2-0.5B-Instruct/')
    parser.add_argument('--save', type=str, nargs='?', default=None)
    parser.add_argument('--load', type=str, nargs='?', default=None)

    args = parser.parse_args()

    if args.prompt_tokens:
        shapes = [int(s) for s in args.prompt_tokens.split("x")]
        if len(shapes) == 2: # batch x prompt_len
            args.prompt0 = ["Hi"*shapes[1]] * shapes[0]
        elif len(shapes) == 1: # only batch is provided, just batch the prompt0
            args.prompt0 = [args.prompt0] * shapes[0]
    else:
        args.prompt0 = [args.prompt0]
    simple_pipeline(args.hf_model_path, args.prompt0, args.trace, args.max_new_tokens, args.max_kv_len, args.quant_type,
                    repeat = args.repeat,
                    save_path = args.save,
                    load_path = args.load)