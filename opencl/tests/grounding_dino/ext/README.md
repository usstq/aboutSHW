# How to use
### init openvino
```
source /mnt/luocheng/env/bin/activate
source /mnt/luocheng/openvino_gpu/build/install/setupvars.sh
```

### set env
# CM
```
# cm plugin
export CM_FE_DIR=/home/openvino-ci-74/tingqian/
# load cm/cl code from this dir instead of embedding string
export MYCL_DIR=/mnt/luocheng/openvino_gpu/src/plugins/intel_gpu/src/graph/impls/ocl_v2/
```

### compile extension
mkdir build && cd build
cmake .. && make

### make test unit
refer to `onnx_swin.py/torch_swin.py`

### gpu kernel
ov branch: https://github.com/luo-cheng2021/openvino/tree/luocheng/dino_gpu
refer to `src/plugins/intel_gpu/src/graph/impls/ocl_v2/stub_swattn.cpp/cm`
