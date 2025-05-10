cmake_minimum_required(VERSION 3.23)

project(csrc)

find_package(pybind11 REQUIRED)

find_package(IntelSYCL)
if(IntelSYCL_FOUND)
    if (WIN32)
        set(CMAKE_C_COMPILER "icx")
        set(CMAKE_CXX_COMPILER "icx")
    else()
        set(CMAKE_C_COMPILER "icx")
        set(CMAKE_CXX_COMPILER "icpx") 
    endif()
endif()

FILE(GLOB CPP_SRCS ./*.cpp)

pybind11_add_module(csrc ${CPP_SRCS})
target_compile_features(csrc PRIVATE cxx_std_17)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=core-avx2")
#set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-rpath=/opt/intel/oneapi/compiler/latest/lib")

############# onednn ov-fork ##################
macro(ov_add_compiler_flags)
    foreach(flag ${ARGN})
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${flag}")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${flag}")
    endforeach()
endmacro()

    set(THREADING TBB)
    set(CMAKE_COMPILE_WARNING_AS_ERROR OFF)
    set(CMAKE_DISABLE_FIND_PACKAGE_PythonInterp ON)

    set(DNNL_ENABLE_JIT_PROFILING ${BUILD_SHARED_LIBS} CACHE BOOL "" FORCE)
    if(BUILD_SHARED_LIBS AND ENABLE_PROFILING_ITT)
        set(DNNL_ENABLE_ITT_TASKS ON CACHE BOOL "" FORCE)
    else()
        set(DNNL_ENABLE_ITT_TASKS OFF CACHE BOOL "" FORCE)
    endif()
    
    set(DNNL_ENABLE_CONCURRENT_EXEC ON CACHE BOOL "" FORCE)
    set(DNNL_ENABLE_PRIMITIVE_CACHE ON CACHE BOOL "" FORCE) # Enable primitive cache for global sharing
    set(DNNL_ENABLE_MAX_CPU_ISA ON CACHE BOOL "" FORCE)
    set(DNNL_LIBRARY_TYPE "STATIC" CACHE STRING "" FORCE)
    set(DNNL_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(DNNL_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(DNNL_CPU_RUNTIME "${THREADING}" CACHE STRING "" FORCE)
    set(DNNL_GPU_RUNTIME "NONE" CACHE STRING "" FORCE)
    set(DNNL_BLAS_VENDOR "NONE" CACHE STRING "" FORCE)
    set(ONEDNN_ENABLE_GEMM_KERNELS_ISA "SSE41" CACHE STRING "" FORCE)
    # plugin does not use onednn graph
    set(ONEDNN_BUILD_GRAPH OFF CACHE BOOL "" FORCE)
    # select needed primitives
    set(DNNL_ENABLE_PRIMITIVE "CONVOLUTION;DECONVOLUTION;CONCAT;LRN;INNER_PRODUCT;MATMUL;POOLING;REDUCTION;REORDER;RNN;SOFTMAX" CACHE STRING "" FORCE)
    set(DNNL_ENABLE_WORKLOAD "INFERENCE" CACHE STRING "" FORCE)
    set(DNNL_LIBRARY_NAME "openvino_onednn_cpu" CACHE STRING "" FORCE)

    # allow oneDNN verbose 
    set(DNNL_VERBOSE ON CACHE STRING "" FORCE)
    set(DNNL_TARGET_ARCH "X64" CACHE STRING "" FORCE)

    find_package(TBB
        2017.0
        CONFIG
        PATHS "/home/openvino-ci-87/tingqian/openvino/temp/tbb/lib/cmake/TBB"
        NO_DEFAULT_PATH
        NO_CMAKE_FIND_ROOT_PATH)

    set(TBB_cmake_included ON)
    set(DNNL_CPU_THREADING_RUNTIME "${THREADING}")
    function(find_package_tbb)
        # dummy
    endfunction()
    link_libraries(TBB::tbb)

    ov_add_compiler_flags(-Wno-error)
    ov_add_compiler_flags(-Wno-undef)
    ov_add_compiler_flags(-Wno-missing-declarations)
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 11 AND CMAKE_COMPILER_IS_GNUCXX)
        ov_add_compiler_flags(-Wno-array-bounds)
        ov_add_compiler_flags(-Wno-stringop-overflow)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-overloaded-virtual")
        if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 12)
            ov_add_compiler_flags(-Wno-restrict)
        endif()
    endif()
    if(OV_COMPILER_IS_INTEL_LLVM)
        ov_add_compiler_flags(-Wno-deprecated-this-capture)
        ov_add_compiler_flags(-Wno-deprecated-enum-enum-conversion)
    endif()

    add_subdirectory(oneDNN EXCLUDE_FROM_ALL)

    target_link_libraries(csrc PRIVATE dnnl)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.

target_include_directories(csrc PRIVATE ../../include)
if(IntelSYCL_FOUND)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -lOpenCL")

    # explictly include sycl headers, make intelligence work
    target_include_directories(csrc PRIVATE ${SYCL_INCLUDE_DIR})
    target_include_directories(csrc PRIVATE sycl)
    target_include_directories(csrc PRIVATE sycl/xetla)
    target_link_libraries(csrc PRIVATE OpenCL IntelSYCL::SYCL_CXX)
else()
    target_link_libraries(csrc PRIVATE OpenCL)
endif()

target_compile_definitions(csrc PRIVATE CL_TARGET_OPENCL_VERSION=300)

# target_link_directories(csrc PRIVATE ${SYCL_LIBRARY_DIR})

set_target_properties(csrc
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/../"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

target_compile_definitions(csrc
                           PRIVATE PACKAGE_NAME=csrc)

# try to install oneapi dependencies: not working
if(DNNL_FOUND)
install(TARGETS csrc LIBRARY DESTINATION lib)
install(IMPORTED_RUNTIME_ARTIFACTS DNNL::dnnl DESTINATION lib)
endif()