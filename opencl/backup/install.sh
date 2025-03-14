#!/bin/bash
source /opt/intel/oneapi/setvars.sh
rm ./build/ -rf
rm ./clops/cl*.so
pip install -e .
