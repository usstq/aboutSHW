import subprocess
import tempfile
import os, sys
import ctypes
import numpy as np

class CFunc:
    def __init__(self, ctype_callable, name):
        self.func = ctype_callable
        self.name = name
        # with this: return 64-bit pointer is possible
        self.func.restype = ctypes.c_longlong

    def __call__(self, *args):
        # translate args into ctypes
        # test shows that use 64bit c_longlong even when actual arg type is 32bit int is OK on x86-64
        cargs = []
        for a in args:
            if isinstance(a, int):
                # Note: isinstance(a, int) is True when a is bool type, 
                cargs.append(ctypes.c_longlong(a))
            elif isinstance(a, float):
                cargs.append(ctypes.c_float(a))
            elif isinstance(a, str):
                cargs.append(ctypes.c_char_p(a.encode('utf-8')))
            elif isinstance(a, np.ndarray):
                cargs.append(a.ctypes.data_as(ctypes.c_void_p))
            elif a is None:
                cargs.append(ctypes.c_void_p())
            elif isinstance(a, ctypes.c_void_p):
                cargs.append(a)
            else:
                raise Exception(f"Unspported type '{type(a)}' to C function")
        return self.func(*cargs)

class CLib:
    def __init__(self, src, options, lineno_base, co_filename, disasm):
        so_path = os.path.join('./clib.so')
        args = f"gcc -fopenmp -shared -o {so_path} -Wall -fpic -x c++ - -lstdc++ {options}"
        print(args)
        # insert empty lines into source code so source line number can match
        src = "\n"*lineno_base + src
        cc = subprocess.Popen(args.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        cc.stdin.write(src)
        cc.stdin.close()

        # Read the output of the grep process
        output = cc.stderr.read()
        cc.stderr.close()

        cc.wait()

        if cc.returncode != 0:
            # lineno_base
            for s in output.splitlines():
                if s.startswith("<stdin>:"):
                    parts = s.split(":")
                    parts[0] = co_filename
                    #if (parts[1].isnumeric()):
                    #    parts[1] = str(int(parts[1]) + lineno_base)
                    s = ":".join(parts)
                print(f"\033[31m{s}\033[0m", file=sys.stderr)

            raise Exception(f"CLib compilation failed with command line:\n{args}")

        # print(f"{so_path} genearted.")

        self.dll = ctypes.cdll.LoadLibrary(so_path)
        
        if disasm:
            subprocess.run(f"objdump -d {so_path} -M intel".split())

        # self.dll.test(ctypes.pointer(ctypes.c_float(5)), ctypes.c_int(1))

    def __getattr__(self, name):
        return CFunc(getattr(self.dll, name), name)

import inspect

def clib(options="", disasm=None):
    def _clib(f):
        frame = inspect.currentframe().f_back
        src_lines, line_no = inspect.getsourcelines(f)
        src = f()
        return CLib(src, options, line_no + 1, co_filename=frame.f_code.co_filename, disasm=disasm)
    return _clib

if __name__ == "__main__":
    @clib("")
    def mylib():
        return '''
        #include "common.hpp"
        extern "C" void test(float *a, int count) {
            for(int i = 0; i < count; i++)
                std::cout << ">>" << i << " :" << a[i] << std::endl;
        }
        '''

    a = np.random.randint(-8, 8, [8]).astype(np.float32)

    print(a)

    mylib.test(a, 4)