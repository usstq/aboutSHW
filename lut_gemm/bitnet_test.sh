#!/bin/bash

# LINUX_PERF=dump numactl -C0,2,4,6 python ~/tingqian/BitNet/utils/e2e_benchmark.py -m ~/tingqian/BitNet/models/bitnet_b1_58-3B/ggml-model-tl2.gguf -n 32 -p 1 -t 4
LINUX_PERF=dump numactl -C0 python ~/tingqian/BitNet/utils/e2e_benchmark.py -m ~/tingqian/BitNet/models/bitnet_b1_58-3B/ggml-model-tl2.gguf -n 32 -p 1 -t 1
