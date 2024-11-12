#!/bin/bash
source /opt/intel/oneapi/setvars.sh
CC=icpx CXX=icpx CFLAGS="-Wl,-rpath,/opt/intel/oneapi/compiler/latest/lib" pip install -e .
