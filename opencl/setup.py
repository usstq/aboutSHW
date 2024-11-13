import setuptools
from pybind11.setup_helpers import Pybind11Extension, build_ext

__version__ = "0.1"

import os
include_dirs = ["./clops/csrc/"]
library_dirs = []
if os.name == 'nt':
    # new version
    include_dirs.append(r'C:/Program Files (x86)/Intel/oneAPI/compiler/latest/include/sycl/')
    library_dirs.append(r'C:/Program Files (x86)/Intel/oneAPI/compiler/latest/lib')

    # old version
    include_dirs.append(r'C:/Program Files (x86)/Intel/oneAPI/compiler/latest/windows/include\sycl/')
    library_dirs.append(r'C:/Program Files (x86)/Intel/oneAPI/compiler/latest/windows/lib/')
    library_dirs.append(r'C:/Program Files (x86)/Intel/oneAPI/compiler/latest/windows/compiler/lib/intel64')

ext_modules = [
    Pybind11Extension("clops.cl",
        ["./clops/csrc/cl.cpp", "./clops/csrc/ops.cpp"],
        define_macros = [('VERSION_INFO', __version__)],
        include_dirs = include_dirs,
        library_dirs = library_dirs,
        extra_compile_args=['-fopenmp', '-fPIC', "-fsycl"],
        extra_link_args=["-fopenmp", "-fsycl"],
        libraries=["OpenCL"]
        ),
]

from distutils.spawn import spawn, find_executable

class oneapi_build_ext(build_ext):
    def build_extensions(self):
        # https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/get-started-guide/2024-0/get-started-on-windows.html
        if os.name != 'nt':
            compiler = "icpx"
            self.compiler.set_executable('compiler_so', [compiler])
            self.compiler.set_executable("compiler_cxx", [compiler])
            self.compiler.set_executable('linker_so', [compiler, "-shared"])
        self.compiler.spawn = self.spawn
        build_ext.build_extensions(self)

    def spawn(self, cmd, search_path=1, verbose=0, dry_run=0):
        if 1:
            print("======= spawn ===========")
            print(cmd)
            print(search_path)
            print(verbose)
            print(dry_run)
            print("==================")
        
        # hack on windows
        if os.name == 'nt' and 'cl.exe' in cmd[0]: cmd[0] = "icx-cl"
        spawn(cmd, search_path, verbose, dry_run)
        
        return

setuptools.setup(
    name='clops',
    version="0.1",
    packages=["clops"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": oneapi_build_ext},
    #setup_requires=["pybind11"]
    install_requires=["pybind11"]
)
