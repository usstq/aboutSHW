# clops

OpenCL is perfect to be embedded into python since it's compiled on-the-fly, just like Python. `clops` is a simple wrapper of OpenCL which allows major works of enabling NN workloads done on python side (instead of C++).

some highlights in the design:
 - `cl.tensor` : a pure GPU device memory object
 - `cl_buffer_pool` : all `cl.tensor`s are allocated/returned from/to this pool, this pool allows reuse of OpenCL buffer object across NN layers.
 - on purpose flattened NN-class & object hierarchy, just like functional-programming
 - shottened the distance between network-description & implementation optimization

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

# cpp_kernels

JIT is better than AOT for kernel optimization, if we just focus on optimizing computational kernels, JIT is better than AOT:
 - it can generate codes based on some parameters only available at run-time, like machine-type, kernel hyper-parameter, topology variants, ...
 - it can be done pretty fast if only applied to kernel instead of applying the complicated compilation & linking process over entire project. (it ofcause still take some time, comparing to direct calling AOT kernels, but it's acceptable considering only used kernels will be compiled).
 - it makes the optimization process much faster to iterate & open-source, since even in released version, user can modify the optimized kernel easily and enjoy the performance immediately, a very pythonic way of developing & deploying.

Assembly based JIT (https://github.com/herumi/xbyak) are not prefered because the learning efforts, but intrinsic or some higher-abstraction based on intrinsics are much better choice. this brings the idea of implementing a C based JIT embedded in python.

Some similar but far more pythonic implementations are :
 - A Just-In-Time Compiler for Numerical Functions in Python :  https://github.com/numba/numba
 - An imperative, parallel programming language for high-performance numerical computation: https://github.com/taichi-dev/taichi
 - A Python-like programming language which enables researchers with no CUDA experience to write highly efficient GPU code: https://openai.com/index/triton/

#### Call ABI

Support runtime calling C function (dlsym() in shared lib) from script language is not an easy task:
 - https://eli.thegreenplace.net/2011/09/06/stack-frame-layout-on-x86-64
 - https://stackoverflow.com/questions/37502841/calling-printf-in-extended-inline-asm/37503773#37503773
 - https://stackoverflow.com/questions/50205294/portable-way-to-push-arguments-onto-the-call-stack-c/79130546#79130546


#### References
- a basic x86-64 JIT compiler from scratch in stock Python: https://github.com/cslarsen/minijit
- A CLang&LLVM-JIT based program who can compile & run C-source code at run-time: https://github.com/Kray-G/clang-jit
- EasyJIT: Just-In-Time code generation for C++ codes: https://github.com/jmmartinez/easy-just-in-time
- CLangJIT: Enhancing C++ with Just-in-Time Compilation: https://arxiv.org/pdf/1904.08555
- https://github.com/hfinkel/llvm-project-cxxjit

