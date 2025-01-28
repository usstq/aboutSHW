
from .cfunc import clib
import sys
nargs = len(sys.argv)

if nargs < 3:
    raise Exception(f"at least provide cpp header file & function name")

header_fpath = sys.argv[1]
func_name = sys.argv[2]

extra_flags = ""

# show predefined macro
# echo | gcc -dM -E -march=native -

@clib(f"-std=c++11 -march=native -g -O2 {extra_flags}")
def mylib():
    return r'''
#include "{}"
'''.format(header_fpath)

func = getattr(mylib, func_name)

func(*sys.argv[3:])
