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

############# onednn ##################
if(DEFINED ENV{OV_BUILD_PATH})
    message("++++++++++ $ENV{OV_BUILD_PATH} ++++++++++++++++ ")
    add_library(onednn_gpu STATIC IMPORTED) # or STATIC instead of SHARED
    set_target_properties(onednn_gpu PROPERTIES
        IMPORTED_LOCATION "$ENV{OV_BUILD_PATH}/src/plugins/intel_gpu/thirdparty/onednn_gpu_install/lib/libopenvino_onednn_gpu.a"
        INTERFACE_INCLUDE_DIRECTORIES "$ENV{OV_BUILD_PATH}/src/plugins/intel_gpu/thirdparty/onednn_gpu_install/include/"
    )
    target_link_libraries(csrc PRIVATE onednn_gpu)
    target_compile_definitions(csrc PRIVATE WITH_ONEDNN=1)
else()
    message(WARNING "onednn from OneAPI release may cause segmentfault due to ext_oneapi_get_state, using OV GPU plugin builds w/o DNNL_WITH_SYCL please (OV_DNNL_PATH)")
    find_package(DNNL)
    if(DNNL_FOUND) 
        target_link_libraries(csrc PRIVATE DNNL::dnnl)
        target_compile_definitions(csrc PRIVATE WITH_ONEDNN=1)
    else()
        message(WARNING "onednn from OneAPI release not found, will skip onednn")
    endif()
endif()

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