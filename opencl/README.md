# Build

```bash
$ g++ -O2 ./max_gflops.cpp -lOpenCL
$ ./a.out
# disassemble using ocloc tool from intel-opencl-icd
$ ocloc disasm -file .build/bin_0
```

# How OpenCL ICD(Installable Client Drivers) loader works?
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


# ARC A770 

https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2024-1/xe-arch.html
https://chipsandcheese.com/p/microbenchmarking-intels-arc-a770


|  component name  |      count         | functions  |
|-----------------:|:-------------------|:-----------|
| Xe-HPG render slice   | 8             | `4x Xe-cores` + `graphics pipeline` |
| XC(Xe-core)           | 8x4=32        | `16x (XVE + XMX)` + `Load/Store_Unit` + `L1$/SLM` @`2.1GHz~2.4GHz` |
| XVE (EU)              | 8x4x16=512    | SIMD-8 ALU `8-FP32-MAD per-cycle` |
| Threads               | 8x4x16x8=4096 | `4096*2*2.1G ~ 17.2 TFLOPS` |
| XMX                   | 8x4x16=512    | `64-FP16-MAD per-cycle`/`128-INT8-MAD per-cycle`<br> totally `137.6/275.2 TFLOPS @2.1GHz` |


# Develop & Debug kernels inside torch framework
Torch is a good framework to test & optimize kernels because it's easy to add new kernel (or even backend) and integrate with existing models,
there are few references:

 - https://pytorch.org/tutorials/advanced/privateuseone.html
 - https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html#
 - https://github.com/artyom-beilis/pytorch_dlprim
 - https://github.com/inducer/pyopencl

To our needs, we just want to test & optimize a sub-graph inside a real-model, so the best & simplest choice is just wrap our kernel into a
torch extension and integrate it into a real-model to do testing. `pyopencl` is also a choice but it adds another layer of complexity and
requires additional learning since original OpenCL is C-API.

# References

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

# [test_llama.py](./test_llama.py)

```bash
pip install pybind11
pip install -e .

# one-shot greedy search
$ python3 ./test_llama.py -p "What's Oxygen?"

# profiling with opencl-intercept-layer
$ git clone https://github.com/intel/opencl-intercept-layer
$ mkdir build && cd build
$ cmake ..
$ cmake --build . --config RelWithDebInfo --target install

$ cliloader -d -cdt  --dump-dir ./dump/ python3 ./test_llama.py -p "What's Oxygen?"
```
