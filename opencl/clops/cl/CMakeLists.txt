cmake_minimum_required(VERSION 3.0)

set(CMAKE_C_COMPILER "icx")
set(CMAKE_CXX_COMPILER "icpx")
project(csrc)

find_package(pybind11 REQUIRED)
find_package(IntelSYCL REQUIRED)

FILE(GLOB CPP_SRCS ./*.cpp)

pybind11_add_module(csrc ${CPP_SRCS})

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=core-avx2")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-rpath=/opt/intel/oneapi/compiler/latest/lib")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -lOpenCL")

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.

target_include_directories(csrc PRIVATE ../../include)
# explictly include sycl headers, make intelligence work
target_include_directories(csrc PRIVATE ${SYCL_INCLUDE_DIR})
target_include_directories(csrc PRIVATE sycl)
target_include_directories(csrc PRIVATE sycl/xetla)

set_target_properties(csrc
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/../"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

target_compile_definitions(csrc
                           PRIVATE PACKAGE_NAME=csrc)
