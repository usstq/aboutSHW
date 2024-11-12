import setuptools
from pybind11.setup_helpers import Pybind11Extension, build_ext

# execute following command to build w/o install
#    python setup.py build

__version__ = "0.1"

import os
include_dirs = []
library_dirs = []
if os.name == 'nt':
    include_dirs.append(r'C:/Program Files (x86)/Intel/oneAPI/compiler/latest/include\sycl/')
    library_dirs.append(r'C:/Program Files (x86)/Intel/oneAPI/compiler/latest/lib/')

ext_modules = [
    Pybind11Extension("clops.cl",
        ["./clops/csrc/cl.cpp", "./clops/csrc/ops.cpp"],
        define_macros = [('VERSION_INFO', __version__)],
        include_dirs = include_dirs,
        library_dirs = library_dirs,
        extra_compile_args=['-fopenmp', '-fsycl'],
        extra_link_args=["-fopenmp", '-fsycl'],
        libraries=["OpenCL"]
        ),
]

setuptools.setup(
    name='clops',
    version="0.1",
    packages=["clops"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    #setup_requires=["pybind11"]
    install_requires=["pybind11"]
)
