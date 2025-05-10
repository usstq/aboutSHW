import os
import subprocess
import pybind11
if os.name == 'nt':
    '''
    according to https://learn.microsoft.com/en-us/windows/win32/dlls/dynamic-link-library-search-order
    since python is packaged apps, PATH is not searched when loading DLL (the pybind11 part).
    we have to explicitly add following path for using SYCL/DPC++
    '''
    for path in [#"C:\\Program Files (x86)\\Intel\\oneAPI\\compiler\\latest\\windows\\redist\\intel64_win\\compiler",
                 #"C:\\Program Files (x86)\\Intel\\oneAPI\\compiler\\latest\\windows\\bin",
                 r"C:\\Program Files (x86)\\Intel\\oneAPI\\compiler\\latest\\bin",
                 r"C:\Program Files (x86)\Intel\oneAPI\dnnl\latest\bin",
                 r"C:\Program Files (x86)\Common Files\intel\Shared Libraries\bin"
                 #'C:\\Program Files (x86)\\Intel\\oneAPI\\tbb\\latest\\bin'
                ]:
        if os.path.exists(path):
            os.add_dll_directory(path)

cwd = os.path.dirname(os.path.realpath(__file__))
build_path=os.path.join(cwd, "build")
dir_path = cwd

# where can cmake find packages
cmake_search_dir=":".join([pybind11.get_cmake_dir()])

# on windows, custom compiler requires Ninja instead of VC++
generator="-GNinja" if os.name == 'nt' else ""

#btype = "RelWithDebInfo"
btype = "Debug"

cmake_need_config = not os.path.isfile(os.path.join(build_path, "CMakeCache.txt")) or int(os.environ.get("DO_CMAKE", "0"))

if cmake_need_config:
    subprocess.run(["cmake", "-B", build_path ,
                    "-S", dir_path,
                    f"-DCMAKE_BUILD_TYPE={btype}",
                    "-Wno-dev",
                    f"-DCMAKE_PREFIX_PATH={cmake_search_dir}",
                    "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
                    generator], shell=False, check=True)

subprocess.run(["cmake", "--build", build_path, "--config", btype, "-j32"], shell=False, check=True)

from .csrc import *

