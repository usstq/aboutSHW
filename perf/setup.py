from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='torchPerfProfiler',
      ext_modules=[
        cpp_extension.CppExtension(
                    'torchPerfProfiler', ['torch_ext.cpp'],
                    extra_compile_args=[ '-fopenmp',
                                        '-mno-avx256-split-unaligned-load',
                                        '-mno-avx256-split-unaligned-store',
                                        '-march=native',
                                        #'-DOV_CPU_WITH_PROFILER'
                                        #'-g'
                                        ],
                    extra_link_args=['-lgomp'])
      ],
      include_dirs=['../include'],
      cmdclass={'build_ext': cpp_extension.BuildExtension}
      )