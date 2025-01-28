# LUT based gemm

Excellent examples:

 - https://github.com/microsoft/T-MAC
 - https://github.com/microsoft/BitNet


```bash
$ git clone --recursive https://github.com/microsoft/BitNet.git
$ pip install -r requirements.txt
$ export HF_ENDPOINT=https://hf-mirror.com
$ huggingface-cli download --resume-download 1bitLLM/bitnet_b1_58-3B --local-dir models/bitnet_b1_58-3B
$ python setup_env.py --hf-repo  1bitLLM/bitnet_b1_58-3B --quant-type tl2 -p


# on i9-14900K with 8 P-Cores: 0,2,4,6,8,10,12,14
#  memory bandwidth is 88 GB/s
$ numactl -C0,2,4,8 python ./utils/e2e_benchmark.py -m=models/bitnet_b1_58-3B/ggml-model-tl2.gguf -n 32 -p 32 -t 4
| model                          |       size |     params | backend    | threads | n_batch |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ------: | ------------: | -------------------: |
[LINUX_PERF:362]  LINUX_PERF is unset, example: LINUX_PERF=dump:switch-cpu:L2_MISS=0x10d1
| bitnet 3B TL2                  | 846.01 MiB |     3.32 B | CPU        |       4 |       1 |          pp32 |         57.14 ± 3.67 |
| bitnet 3B TL2                  | 846.01 MiB |     3.32 B | CPU        |       4 |       1 |          tg32 |         60.11 ± 0.01 |

$ numactl -C0,2,4,6,8,10,12,14 python ./utils/e2e_benchmark.py -m=models/bitnet_b1_58-3B/ggml-model-tl2.gguf -n 32 -p 32 -t 8
| model                          |       size |     params | backend    | threads | n_batch |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ------: | ------------: | -------------------: |
[LINUX_PERF:362]  LINUX_PERF is unset, example: LINUX_PERF=dump:switch-cpu:L2_MISS=0x10d1
| bitnet 3B TL2                  | 846.01 MiB |     3.32 B | CPU        |       8 |       1 |          pp32 |         63.88 ± 0.30 |
| bitnet 3B TL2                  | 846.01 MiB |     3.32 B | CPU        |       8 |       1 |          tg32 |         74.74 ± 0.20 |
```



