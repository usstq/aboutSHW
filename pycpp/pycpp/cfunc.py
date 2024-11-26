import subprocess
import tempfile
import os, sys
import ctypes
import numpy as np

class CFunc:
    def __init__(self, ctype_callable, name):
        self.func = ctype_callable
        self.name = name

    def __call__(self, *args):
        # translate args into ctypes
        cargs = []
        for a in args:
            if isinstance(a, int):
                cargs.append(ctypes.c_int(a))
            elif isinstance(a, float):
                cargs.append(ctypes.c_float(a))
            elif isinstance(a, str):
                cargs.append(ctypes.c_char_p(a.encode('utf-8')))
            elif isinstance(a, np.ndarray):
                cargs.append(a.ctypes.data_as(ctypes.c_void_p))
            else:
                raise Exception(f"Unspported type '{type(a)}' to C function")
        self.func(*cargs)

class CLib:
    def __init__(self, src, options, lineno_base, co_filename, disasm):
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_name = next(tempfile._get_candidate_names())
            so_path = os.path.join(tmp_dir,temp_name+'.so')
            args = f"gcc -fopenmp -shared -o {so_path} {options} -Wall -fpic -x c++ - -lstdc++"

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
                        if (parts[1].isnumeric()):
                            parts[1] = str(int(parts[1]) + lineno_base)
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