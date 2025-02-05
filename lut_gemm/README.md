# LUT based gemm

Excellent examples:
 - https://arxiv.org/abs/2206.09557 LUT-GEMM: Quantized Matrix Multiplication based on LUTs ...
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

## T-MAC Paper & Code

```python
######### compute #########
# https://github.com/microsoft/T-MAC/blob/8f29b961b83c5e335d1199153f59ce20032650d6/python/t_mac/ops/qgemm.py#L183
    CBits = te.compute(
        (N, M),
        lambda n, m: te.sum(
            _scale_first(m, n, k, LUT[n, k, _get_Abits(m, k)]),
            axis=k,
        ),
        name="CBits",
    )

    C = te.compute(
        (N, M // self.bits),
        lambda n, m: _scale_final(m, n,
            sum([
                CBits[
                    n,
                    te.indexdiv(m, self.simd_n_out) * self.simd_n_out * self.bits
                        + te.indexmod(m, self.simd_n_out)
                        + b * self.simd_n_out
                ].astype("float32") * alphas[b]
                for b in range(self.bits)
            ]),
        ).astype(self.out_dtype),
        name="C",
    )
######### schedule #########
    LUT = tensors[1]
    K = int(LUT.shape[1] * self.g)
    C = tensors[-1]
    sch: te.Schedule = te.create_schedule(C.op)

    n, m = sch[C].op.axis

    CC = sch.cache_write(C, "global")
    no, mo, ni, mi = sch[C].tile(n, m, self.bn, self.bm // self.bits)
    sch[CC].compute_at(sch[C], mo)

    CBits = CC.op.input_tensors[0]
    sch[CBits].compute_at(sch[C], mo)

    nC, mC = sch[CBits].op.axis
    (kC,) = sch[CBits].op.reduce_axis
    koC, kiC = sch[CBits].split(kC, factor=self.kfactor)
    sch[CBits].reorder(koC, nC, kiC, mC)
```


```bash
numactl -C0,2,4,6 pytest -rP -k mm_i2s
HW_CPU_CYCLES(1278302) HW_INSTRUCTIONS(2211083)  [mm_i2s] 0.225 ms  5.7 GHz  CPI:312.1  596.7 GFLOPS  74.6 GB/s
```


## `Conclusion` VNNI based vs LUT based
gemm problem is gradually become compute-bounded when M > 1, and if the implementation is VNNI based, the best compute-bounded kernel would be decompress(pre-unpack) weight into VNNI format and kernel can read VNNI weight's w/o decompress it at runtime.

But LUT based kernel using `vpshufb`, each instruction provides `(M*K*N)*2=(1*4*32)*2=256` OPS, which is `4x` of OPS VNNI could provide (`1*4*8*2=64`).
so LUT methods can be used on compute-bounded case too.

## `Pain point` - SIMD intrinsics are hard to use

unlike plain C language which supports versatile behaviour programing using very small set of syntax, SIMD instructions are instead very complex and hard to understood or remember, especially the data-movement related instructions, we can categorize them into few types:
 - shuffle within a single source register : `vpshufb`/`vpshufd`/`vpshuflw`/`vshufps`/`vpermps`/`vpermd`/
 - pick out from 2 source registers according to a pre-defined pattern : `vpunpcklbw`/`vpunpckhbw`
 - pick out from 2 source registers according to cutom pattern : `vperm2i128`/`vperm2f128`/``

the pain point is, there is no all-purpose instruction which does everything, each instruction has its own limitations (which is often strange & unnatural) so you have to choose them carefully and combine them according to real cases, this is very hard, since programmer needs to memorize all instructions and do smart choices to combine them.

## `Coding practice` - invoking CMake at runtime

Builing system CMake conditionally build only when source is changed, invoking it at runtime saves extra step of compiling C++ part and smoothes
the python integration during development.

```python
# make csrc & import lib
subprocess.check_output(["cmake", "-B", "build", "-DCMAKE_BUILD_TYPE=Debug"], shell=False)
subprocess.check_output(["cmake", "--build", "build", "--config", "Debug"], shell=False)
from build import lut_gemm
```

## `Coding practice` - using pytest

Good thing about pytest, invoking `pytest` or `pytest -rA` will automatically find test cases (just any normal python function prefixed with `test_`) and execute them, w/o the need to organize test functions. the test can simply use `assert` to report failure. the test can also including pybind11 based C++ part.

## `Coding practice` - develop optimized kernel from low to high

`low` means the inner-most CPU depedent micro kernel, which is the first one to be implemented and tested, based on which we can estimate performance early and built upper-level kernels based on that.

The inner-most kernel is responsible for choosing the instructions & register blocking, cache related blocking strategy would be considered later only after this kernel is implemented.

