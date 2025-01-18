import setuptools
from pybind11.setup_helpers import Pybind11Extension, build_ext

__version__ = "0.1"

'''
    Pybind11Extension("pycpp.gemm",
        ["./pycpp/csrc/gemm.cpp"],
        define_macros = [('VERSION_INFO', __version__)],
        include_dirs = ["./"],
        library_dirs = ["./"],
        extra_compile_args = ['-fopenmp','-march=core-avx2', '-fPIC', '-std=c++11', '-O2', '-g'],
        extra_link_args = ['-fopenmp'],
        ),


'''
ext_modules = [
    Pybind11Extension("pycpp.perf",
        ["./pycpp/csrc/perf.cpp"],
        define_macros = [('VERSION_INFO', __version__)],
        include_dirs = ["./"],
        library_dirs = ["./"],
        extra_compile_args = ['-fopenmp', '-fPIC', '-std=c++11'],
        extra_link_args = ['-fopenmp'],
        ),
]

setuptools.setup(
    name='pycpp',
    version="0.1",
    packages=["pycpp"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    #setup_requires=["pybind11"]
    install_requires=["pybind11"]
)
