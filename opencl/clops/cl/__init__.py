import os
import subprocess
import pybind11
if os.name == 'nt':
    '''
    according to https://learn.microsoft.com/en-us/windows/win32/dlls/dynamic-link-library-search-order
    since python is packaged apps, PATH is not searched when loading DLL (the pybind11 part).
    we have to explicitly add following path for using SYCL/DPC++
    '''
    for path in ["C:\\Program Files (x86)\\Intel\\oneAPI\\compiler\\latest\\windows\\redist\\intel64_win\\compiler",
                 "C:\\Program Files (x86)\\Intel\\oneAPI\\compiler\\latest\\windows\\bin",
                 "C:\\Program Files (x86)\\Intel\\oneAPI\\compiler\\latest\\bin"]:
        if os.path.exists(path):
            os.add_dll_directory(path)

cwd = os.path.dirname(os.path.realpath(__file__))
build_path=os.path.join(cwd, "build")
dir_path = cwd
btype = "RelWithDebInfo"
pybind11_dir = pybind11.get_cmake_dir()
#btype = "Debug"
subprocess.check_output(["cmake", "-B", build_path , "-S", dir_path, f"-DCMAKE_BUILD_TYPE={btype}", "-Wno-dev", f"-DCMAKE_PREFIX_PATH={pybind11_dir}", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"], shell=False)
subprocess.check_output(["cmake", "--build", build_path, "--config", btype], shell=False)

from .csrc import *