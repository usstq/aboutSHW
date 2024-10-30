import setuptools
from pybind11.setup_helpers import Pybind11Extension, build_ext

# execute following command to build w/o install
#    python setup.py build

__version__ = "0.1"

import os
include_dirs = []
library_dirs = []
if os.name == 'nt':
    include_dirs.append('C:\\Program Files (x86)\\Intel\\oneAPI\\compiler\\2023.2.1\\windows\\include\\sycl\\')
    library_dirs.append('C:\\Program Files (x86)\\Intel\\oneAPI\\compiler\\2023.2.1\\windows\\lib\\')

ext_modules = [
    Pybind11Extension("clops.cl",
        ["./clops/cl.cpp"],
        define_macros = [('VERSION_INFO', __version__)],
        include_dirs = include_dirs,
        library_dirs = library_dirs,
        extra_compile_args=['-fopenmp'],
        extra_link_args=["-fopenmp"],
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
