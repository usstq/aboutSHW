# How to explore and do the optimization
### init openvino
```
# compile openvino
# create python env & compile openvino
source /mnt/luocheng/openvino_gpu/build/install/setupvars.sh
```

### set env
```
# cm plugin
export CM_FE_DIR=/mnt/luocheng/
# [optional, only for debug] load cm/cl code from this dir instead of embedding string
export MYCL_DIR=/mnt/luocheng/openvino_gpu/src/plugins/intel_gpu/src/graph/impls/ocl_v2/
```

### compile op extension
```
mkdir build && cd build
cmake .. && make
```

### modify pytorch model and extract pattern then make test unit
refer to `onnx_swin*.py/torch_swin.py`

### write gpu kernel
ov branch: https://github.com/luo-cheng2021/openvino/tree/luocheng/dino_gpu
refer to `src/plugins/intel_gpu/src/graph/impls/ocl_v2/stub_swattn.cpp/cm`

# A complete example: how to run grouding_dino
### init openvino
```
# create python env & compile openvino
source /mnt/luocheng/env/bin/activate
source /mnt/luocheng/openvino_gpu/build/install/setupvars.sh
```

### set env
```
# cm plugin(may be copied from 10.67.109.3)
export CM_FE_DIR=/mnt/luocheng/
```

### compile op extension
```
mkdir build && cd build
cmake .. && make
```

### export&run grouding_dino model
```
git clone https://github.com/luo-cheng2021/mmdetection -b luocheng/grouding_dino
# config grouding_dino
cd mmdetection/z_test
# export optimized onnx model
./export.sh
./test.sh
```
*Note: the following hard code should be changed in `mmdection` repo(models may be copied from 10.67.109.3):*
```
1, bert model path in `configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365.py`
2, bert model path in `z_test/export_groundingdino_openvino.py`
3, ov op extension path in `z_test/ov.py`
4, cm compiler path in `z_test/test.sh`
5, picture, model config and pytorch model path in `z_test/export.sh`
```
