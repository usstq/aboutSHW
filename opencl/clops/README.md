# clops

OpenCL is perfect to be embedded into python since it's compiled on-the-fly, just like Python. `clops` is a simple wrapper of OpenCL which allows major works of enabling NN workloads done on python side (instead of C++).

some highlights in the design:
 - `cl.tensor` : a pure GPU device memory object
 - `cl_buffer_pool` : all `cl.tensor`s are allocated/returned from/to this pool, this pool allows reuse of OpenCL buffer object across NN layers.
 - on purpose flattened NN-class & object hierarchy, just like functional-programming
 - shottened the distance between network-description & implementation optimization

```bash
# install (from parent folder)
pip install pybind11
pip install -e .

# test llama inference
python3 -m clops.tests.llama -p "What's Oxygen?" -n 32

# profiling with opencl-intercept-layer: build from source
$ git clone https://github.com/intel/opencl-intercept-layer
$ mkdir build && cd build
$ cmake ..
$ cmake --build . --config RelWithDebInfo --target install

# profiling with opencl-intercept-layer: profiling
$ cliloader -dv -cdt  --dump-dir ./dump/ python3 -m clops.tests.llama -p "What's Oxygen?" -n 32

# https://github.com/intel/pti-gpu/tree/master/tools/unitrace
$ pti-gpu/tools/unitrace/build/unitrace --output-dir-path trace -d -h --opencl --chrome-call-logging  --chrome-kernel-logging --chrome-device-logging   python -m clops.tests.llama -p "What's Oxygen"
 
```

## References

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