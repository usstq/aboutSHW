import os
if os.name == 'nt':
    '''
    according to https://learn.microsoft.com/en-us/windows/win32/dlls/dynamic-link-library-search-order
    since python is packaged apps, PATH is not searched when loading DLL (the pybind11 part).
    we have to explicitly add following path for using SYCL/DPC++
    '''
    os.add_dll_directory("C:\\Program Files (x86)\\Intel\\oneAPI\\compiler\\latest\\windows\\redist\\intel64_win\\compiler") # old versions
    os.add_dll_directory("C:\\Program Files (x86)\\Intel\\oneAPI\\compiler\\latest\\windows\\bin") # old versions
    os.add_dll_directory("C:\\Program Files (x86)\\Intel\\oneAPI\\compiler\\latest\\bin") # 

from .ops import *
from .mha import MHA
from .mha_cpu import MHA_cpu
from .linear_w4a import Linear_w4a
from .linear_f16 import Linear_f16
from .linear_f16b1 import Linear_f16b1
from .linear_f16xmx import Linear_f16xmx
from .linear_w4x import Linear_w4x
from .rms_norm import RMSNorm
from .rope import ROPE