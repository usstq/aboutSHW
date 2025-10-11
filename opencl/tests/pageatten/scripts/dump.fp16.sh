#!/bin/bash

current_path=$(pwd)
echo "Current path is: $current_path"

export OV_GPU_DUMP_MEMORY_POOL=1
export OV_GPU_DUMP_ITERATIONS="0 1"

export OV_VERBOSE=4

cd $current_path
mkdir -p fp16.kvcache/32k/thresh0.1 && cd fp16.kvcache/32k/thresh0.1
model_dir=/home/ceciliapeng/llm_irs/WW35_llm-optimum_2025.3.0-19807-RC2/qwen3-8b/pytorch/ov/OV_FP16-4BIT_DEFAULT
prompts_dir=/home/ceciliapeng/frameworks.ai.openvino.llm.prompts/xattn-32K.jsonl
OV_GPU_XATTN_THRESH=0.1   python /home/ceciliapeng/openvino.genai/tools/llm_bench/benchmark.py -m $model_dir --disable_prompt_permutation -d GPU -pf $prompts_dir --load_config '{"KV_CACHE_PRECISION": "f16"}' --cb_config "{\"enable_prefix_caching\":false, \"max_num_batched_tokens\": 4096}" -n 1 --infer_count 1 > LOG 2>&1

cd $current_path
mkdir -p fp16.kvcache/32k/thresh0.9 && cd fp16.kvcache/32k/thresh0.9
model_dir=/home/ceciliapeng/llm_irs/WW35_llm-optimum_2025.3.0-19807-RC2/qwen3-8b/pytorch/ov/OV_FP16-4BIT_DEFAULT
prompts_dir=/home/ceciliapeng/frameworks.ai.openvino.llm.prompts/xattn-32K.jsonl
OV_GPU_XATTN_THRESH=0.9   python /home/ceciliapeng/openvino.genai/tools/llm_bench/benchmark.py -m $model_dir --disable_prompt_permutation -d GPU -pf $prompts_dir --load_config '{"KV_CACHE_PRECISION": "f16"}' --cb_config "{\"enable_prefix_caching\":false, \"max_num_batched_tokens\": 4096}" -n 1 --infer_count 1 > LOG 2>&1

cd $current_path
mkdir -p fp16.kvcache/64k/thresh0.1 && cd fp16.kvcache/64k/thresh0.1
model_dir=/home/ceciliapeng/llm_irs/WW35_llm-optimum_2025.3.0-19807-RC2/qwen3-8b/pytorch/ov/OV_FP16-4BIT_DEFAULT
prompts_dir=/home/ceciliapeng/frameworks.ai.openvino.llm.prompts/xattn-64K.jsonl
OV_GPU_XATTN_THRESH=0.1   python /home/ceciliapeng/openvino.genai/tools/llm_bench/benchmark.py -m $model_dir --disable_prompt_permutation -d GPU -pf $prompts_dir --load_config '{"KV_CACHE_PRECISION": "f16"}' --cb_config "{\"enable_prefix_caching\":false, \"max_num_batched_tokens\": 4096}" -n 1 --infer_count 1 > LOG 2>&1

cd $current_path
mkdir -p fp16.kvcache/64k/thresh0.9 && cd fp16.kvcache/64k/thresh0.9
model_dir=/home/ceciliapeng/llm_irs/WW35_llm-optimum_2025.3.0-19807-RC2/qwen3-8b/pytorch/ov/OV_FP16-4BIT_DEFAULT
prompts_dir=/home/ceciliapeng/frameworks.ai.openvino.llm.prompts/xattn-64K.jsonl
OV_GPU_XATTN_THRESH=0.9   python /home/ceciliapeng/openvino.genai/tools/llm_bench/benchmark.py -m $model_dir --disable_prompt_permutation -d GPU -pf $prompts_dir --load_config '{"KV_CACHE_PRECISION": "f16"}' --cb_config "{\"enable_prefix_caching\":false, \"max_num_batched_tokens\": 4096}" -n 1 --infer_count 1 > LOG 2>&1