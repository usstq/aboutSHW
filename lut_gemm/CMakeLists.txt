cmake_minimum_required(VERSION 3.0)

project(lut_gemm)

add_subdirectory(../thirdparty/pybind11 pybind11)
pybind11_add_module(lut_gemm csrc/main.cpp csrc/i2s.cpp csrc/perf.cpp csrc/tl1.cpp)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_include_directories(lut_gemm PRIVATE ../include)

target_compile_definitions(lut_gemm
                           PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})

