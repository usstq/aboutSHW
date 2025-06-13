# clops

`clops` is a simple wrapper of OpenCL/SYCL which allows major works of enabling NN workloads done on python side (instead of C++).

some highlights in the design:
 - `cl.tensor` : a pure GPU device memory object
 - `buffer_pool` : all `cl.tensor`s are allocated/returned from/to this pool, this pool allows reuse of OpenCL buffer object across NN layers.
 - on purpose flattened NN-class & object hierarchy, just like functional-programming
 - shottened the distance between network-description & implementation optimization

```bash


# install cm-compiler
sudo apt-get install -y gpg-agent wget
wget -qO - https://repositories.intel.com/graphics/intel-graphics.key |  sudo apt-key add -
sudo apt-add-repository 'deb [arch=amd64] https://repositories.intel.com/graphics/ubuntu focal main'
sudo apt update
sudo apt install intel-igc-cm

# need intel compiler to work
# win32
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
# linux
source /opt/intel/oneapi/setvars.sh 

# install clops
pip install -e .

# on Windows, download & extract OpenCL SDK from https://github.com/KhronosGroup/OpenCL-SDK/releases
# then tell cmake where to find_package(OpenCL) by export(or set) following env
export OpenCL_ROOT="C:\Users\openvino-adlh\Downloads\OpenCL-SDK-v2024.10.24-Win-x64"

# run unit test (cmake will be automatically called to reflect the change in csrc)
python -m clops.linear_f16xmx
python -m clops.linear_w4x
...

# [optional]
# in case using C-for-metal: build & install cm-compiler  (libclangFEWrapper.so)
# https://github.com/intel/cm-compiler/blob/cmc_monorepo_110/clang/Readme.md
cd $ROOT
git clone https://github.com/intel/cm-compiler.git -b cmc_monorepo_110 llvm-project
git clone https://github.com/intel/vc-intrinsics.git llvm-project/llvm/projects/vc-intrinsics
git clone https://github.com/KhronosGroup/SPIRV-LLVM-Translator.git -b llvm_release_110 llvm-project/llvm/projects/SPIRV-LLVM-Translator
mkdir build && cd build
cmake -DLLVM_ENABLE_Z3_SOLVER=OFF -DCLANG_ANALYZER_ENABLE_Z3_SOLVER=OFF -DCMAKE_INSTALL_PREFIX=../install -DLLVM_ENABLE_PROJECTS="clang" -DLLVM_TARGETS_TO_BUILD="" ../llvm-project/llvm
make install -j8

# CM_FE_DIR must contains libclangFEWrapper.so
$ CM_FE_DIR=/path/to/install/lib/ python ./tests/test_cm.py


# run whole llama like model with w4x quantization type
python llama.py -hf ../../models/Llama-2-7b-chat-hf/ -q w4x -p "[INST] What's Oxygen? [/INST]"
python llama.py -hf ../../models/Llama-2-7b-chat-hf/ -q w4x -x 16x1024 -n 0 -r 2
python llama.py -hf ../../models/Llama-2-7b-chat-hf/ -q w4x -n128 --save clops-llama2-7b-model
python llama.py -hf ../../models/Llama-2-7b-chat-hf/ -q w4x -n128 --load clops-llama2-7b-model

# profiling with opencl-intercept-layer: build from source
$ git clone https://github.com/intel/opencl-intercept-layer
$ mkdir build && cd build
$ cmake ..
$ cmake --build . --config RelWithDebInfo --target install

# profiling with opencl-intercept-layer: profiling
$ cliloader -dv -cdt  --dump-dir ./dump/ python3 -m clops.tests.llama -p "What's Oxygen?" -n 32

# https://github.com/intel/pti-gpu/tree/master/tools/unitrace
$ pti-gpu/tools/unitrace/build/unitrace --output-dir-path trace -d -h --opencl --chrome-call-logging  --chrome-kernel-logging --chrome-device-logging   python -m clops.tests.llama -p "What's Oxygen"

# build C++ examples
$ g++ -O2 ./max_gflops.cpp -lOpenCL
$ ./a.out
# disassemble using ocloc tool from intel-opencl-icd
$ ocloc disasm -file .build/bin_0

```


### How OpenCL ICD(Installable Client Drivers) loader works?
- user app links to ICD loader `libOpenCL.so` ([github repo](`https://github.com/KhronosGroup/OpenCL-ICD-Loader`)) to use OpenCL API, we can verify that by `readelf -Ws /lib/x86_64-linux-gnu/libOpenCL.so.1`

- ICD (provided by GPGPU vendor) must provide `clGetExtensionFunctionAddress`, ICD loader will use this function to get function pointer to `clIcdGetPlatformIDsKHR`
- `clIcdGetPlatformIDsKHR` returns the platform object (the root of OpenCL object hierarchy)
   [src](https://github.com/KhronosGroup/OpenCL-ICD-Loader/blob/804b6f040503c47148bee535230070da6b857ae4/loader/icd.c#L108).
- all OpenCL objects created from root object (Device/Context/Program/Queue/...) has been associated with a dispatch table pointer (like `vtable` in C++) as the first field, so following OpenCL API calls happens on them can be correctly dispatched ([source code](https://github.com/OCL-dev/ocl-icd/blob/fdde6677b21329432db8b481e2637cd10f7d3cb2/ocl_icd_loader.c#L633))

# Intel(R) Graphics Compute Runtime for oneAPI Level Zero and OpenCL(TM) Driver

 - Ubuntu package of Intel GPU ICD : `intel-opencl-icd` ([github repo](https://github.com/intel/compute-runtime)).

```bash
$ dpkg -L intel-opencl-icd
/.
/etc/OpenCL/vendors/intel.icd   # one-line: /usr/lib/x86_64-linux-gnu/intel-opencl/libigdrcl.so
/usr/bin/ocloc                  # ocloc: important binary tools for OpenCL program binaries
/usr/include/ocloc_api.h        # all functions of ocloc is in libocloc.so, this header provides C API
/usr/lib/x86_64-linux-gnu/intel-opencl/libigdrcl.so # the actual OpenCL implementation
/usr/lib/x86_64-linux-gnu/libocloc.so

# ocloc can be used to disassemble the binaries dumped by clGetProgramInfo(..., CL_PROGRAM_BINARIES, ...)
ocloc disasm -file .build/bin_0
```


### ARC A770 

https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2024-1/xe-arch.html
https://chipsandcheese.com/p/microbenchmarking-intels-arc-a770


|  component name  |      count         | functions  |
|-----------------:|:-------------------|:-----------|
| Xe-HPG render slice   | 8             | `4x Xe-cores` + `graphics pipeline` |
| XC(Xe-core)           | 8x4=32        | `16x (XVE + XMX)` + `Load/Store_Unit` + `L1$/SLM` @`2.1GHz~2.4GHz` |
| XVE (EU)              | 8x4x16=512    | SIMD-8 ALU `8-FP32-MAD per-cycle` |
| Threads               | 8x4x16x8=4096 | `4096*2*2.1G ~ 17.2 TFLOPS` |
| XMX                   | 8x4x16=512    | `64-FP16-MAD per-cycle`/`128-INT8-MAD per-cycle`<br> totally `137.6/275.2 TFLOPS @2.1GHz` |


### MTL and SDPA

https://chipsandcheese.com/p/intels-ambitious-meteor-lake-igpu?utm_source=publication-search

|  component name  |      count         | functions  |
|-----------------:|:-------------------|:-----------|
| Xe-LP render slice    | 2             | `4x Xe-cores` + `graphics pipeline` |
| XC(Xe-core)           | 2x4=8         | `16x (XVE)` + `Load/Store_Unit` + `L1$/SLM` @`0.8GHz~2.3GHz` |
| XVE (EU)              | 2x4x16=128    | SIMD-8 ALU `8-FP32-MAD per-cycle` |
| Threads               | 2x4x16x8=1024 | `1024*2*2.3G ~ 4.7 TFLOPS` |


Say 8K input length for QWen2 sdpa first token, the measured perf so far -
1. Gemm gflops / layer : 2x gemms of [1, 28, 8553, 128]  : 28\*8553\*128\*2\*8553/1024/1024/1024*2 = 976.7 Gflops / layer  => 27.328 Tflops / infer
2. Softmax gflops / layer : 28\*8553\*8553\*2/1024/1024/1024 => 3.8 Gflops / layer => 106.8 Gflops / infer
3. Ideal time of softmax : 3.8/80G  = 47 ms / layer => 1.3 sec / infer
3. Ideal time of 2 Gemms : 976.7/4.7T = 207ms / layer => 5.8 sec / infer


### Develop & Debug & Profile kernels inside torch framework

Torch is a good framework to test & optimize kernels because it's easy to add new kernel (or even backend) and integrate with existing models,
there are few references:

 - https://pytorch.org/tutorials/advanced/privateuseone.html
 - https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html#
 - https://github.com/artyom-beilis/pytorch_dlprim
 - https://github.com/inducer/pyopencl

To our needs, we just want to test & optimize a sub-graph inside a real-model, so the best & simplest choice is just wrap our kernel into a
torch extension and integrate it into a real-model to do testing. `pyopencl` is also a choice but it adds another layer of complexity and
requires additional learning since original OpenCL is C-API.

The [unitrace](https://github.com/intel/pti-gpu/tree/master/tools/unitrace) could be used to profile the performance. The following command line would generate a `python.pid.json` in folder `trace` and it could be viewed by chrome tracing tool:
```bash
unitrace --output-dir-path trace -d --opencl --chrome-kernel-logging python -m clops.tests.llama -p "What's Oxygen"`
```

### References

Hardware:
 - Intel-Graphics ISA https://www.intel.com/content/dam/develop/external/us/en/documents/micro2015-isa-igc-tutorial.pdf
 - Intel-Graphics-Compiler: https://github.com/intel/intel-graphics-compiler
 - Intel-Graphics-Compiler Virtual-ISA: https://github.com/intel/intel-graphics-compiler/tree/master/documentation/visa/instructions
 - ARC770 spec: https://www.techpowerup.com/gpu-specs/arc-a770.c3914
 - Detailed ARC GPU doc: https://www.x.org/docs/intel/ACM/intel-gfx-prm-osrc-acm-vol09-renderengine.pdf

Software:
 - custom torch types & ops : https://pytorch.org/tutorials/advanced/privateuseone.html
 - custom torch types & ops : https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html#
 - off-tree torch OpenCL backend: https://github.com/artyom-beilis/pytorch_dlprim
 - OpenCL python wrapper: https://github.com/inducer/pyopencl


 - https://github.com/rsnemmen/OpenCL-examples/blob/master/Hello_World/hello.c
 - https://github.com/yohanesgultom/parallel-programming-assignment/blob/master/PR2/opencl/device_query.c
 - https://github.com/KhronosGroup/OpenCL-Guide/blob/main/chapters/opencl_programming_model.md

 - https://www.iwocl.org/wp-content/uploads/iwocl2017-ben-ashbaugh-subgroups.pdf
 - https://www.codeproject.com/Articles/994769/SGEMM-for-Intel-Processor-Graphics


 - OpenCL.lib:  C:\Program Files (x86)\Intel\oneAPI\compiler\2023.2.1\windows\lib
 - cl.h :       C:\Program Files (x86)\Intel\oneAPI\compiler\2023.2.1\windows\include\sycl\CL
 - opencl.hpp : https://github.com/KhronosGroup/OpenCL-CLHPP/blob/main/include/CL/opencl.hpp

 - ocloc tools: C:\Program Files (x86)\Intel\oneAPI\compiler\2023.2.1\windows\lib\ocloc\gen9-11
 - clintercept-3.0.3-win64: https://github.com/intel/opencl-intercept-layer


# Occupancy
https://oneapi-src.github.io/oneAPI-samples/Tools/GPU-Occupancy-Calculator/index.html
