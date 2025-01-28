cmake_minimum_required(VERSION 3.0)

project(lut_gemm)

add_subdirectory(../thirdparty/pybind11 pybind11)
pybind11_add_module(lut_gemm csrc/main.cpp)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_include_directories(lut_gemm PRIVATE ../include)

target_compile_definitions(lut_gemm
                           PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})

