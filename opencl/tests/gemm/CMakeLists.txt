cmake_minimum_required(VERSION 3.10)

project(gemm)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
if(MSVC)
  add_compile_options(/std:c++17)
endif()

if(NOT DEFINED ENV{CSDK_IGC})
  message(FATAL_ERROR "CSDK_IGC environment variable is not set - did you run 'setenv.bat' ?")
endif()

# set paths
set(CSDK_IGC $ENV{CSDK_IGC})
set(CMEMU_PATH ${CSDK_IGC}/cmemu)

# locate all relevant packages
foreach(program CMC)
  string(TOLOWER ${program} binary)
  find_program(${program} ${binary} REQUIRED)
  if(NOT ${program})
    message(FATAL_ERROR "Unable to locate ${binary} executable - did you run 'setenv.bat' ?")
  else()
    message(INFO " using ${binary} from ${${program}}")
  endif()
endforeach(program)
find_library(LIB_CM_64 NAMES cm libcm PATHS ${CMEMU_PATH}/lib/x64)
find_library(LIB_SHIM_64 NAMES shim PATHS ${CMEMU_PATH}/lib/x64)
find_library(LIB_SHIM_L0_64 NAMES shim_l0 PATHS ${CMEMU_PATH}/lib/x64)
find_library(LIB_CMEMU_64 NAMES igfxcmrt_emu igfxcmrt64_emu PATHS ${CMEMU_PATH}/lib/x64)
find_library(LIB_OPENCL NAMES Intel_OpenCL_ICD64 PATHS ${CSDK_IGC}/runtime/opencl/lib)
find_library(LIB_LEVEL0 NAMES ze_loader PATHS ${CSDK_IGC}/runtime/level_zero/lib)

# our sources
set(KERNEL ${CMAKE_SOURCE_DIR}/kernel.cpp)
set(HOST_OCL ${CMAKE_SOURCE_DIR}/host.cpp)
set(HOST_L0  ${CMAKE_SOURCE_DIR}/host_l0.cpp)

# os-specific
if (CMAKE_HOST_SYSTEM_NAME MATCHES Windows)
  set(dll ${CMAKE_SHARED_LIBRARY_SUFFIX})
  set(HW_PLATFORM tgllp)
else()
  set(dll)
  set(HW_PLATFORM skl)
endif()
string(TOUPPER ${HW_PLATFORM} HW_PLATFORM_UPPER)

set(INSTALL_DIR ${CMAKE_BINARY_DIR}/bin)


#######
# SHIM / OCL
add_library(kernel.shim MODULE ${KERNEL})
# 1276--> ARL
# 1290--> BMG 1295-->LNL
set_target_properties(kernel.shim PROPERTIES COMPILE_FLAGS "-DCM_GENX=1295 -DCMRT_EMU -DSHIM")
target_include_directories(kernel.shim PUBLIC ${CMEMU_PATH}/include/libcm ${CMEMU_PATH}/include/shim)
target_link_libraries(kernel.shim ${LIB_CM_64})
install(TARGETS kernel.shim DESTINATION ${INSTALL_DIR})

add_executable(vector.shim ${HOST_OCL})
set_target_properties(vector.shim PROPERTIES COMPILE_FLAGS -DKERNEL=\\\"kernel.shim${dll}\\\")
target_include_directories(vector.shim PUBLIC ${CMEMU_PATH}/include/libcm ${CSDK_IGC}/runtime/opencl/include)
target_link_libraries(vector.shim ${LIB_SHIM_64} ${LIB_CMEMU_64} ${LIB_CM_64})
add_dependencies(vector.shim kernel.shim)
install(TARGETS vector.shim DESTINATION ${INSTALL_DIR})

# all targets to build
add_custom_target(build ALL)
add_dependencies(build vector.shim)
