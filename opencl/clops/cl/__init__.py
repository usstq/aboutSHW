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
                 #"C:\\Program Files (x86)\\Intel\\oneAPI\\compiler\\latest\\bin",
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

btype = "RelWithDebInfo"
#btype = "Debug"

cmake_need_config = not os.path.isfile(os.path.join(build_path, "CMakeCache.txt")) or int(os.environ.get("DO_CMAKE", "0"))

if cmake_need_config:
    subprocess.run(["cmake", "-B", build_path ,
                    "-S", dir_path,
                    f"-DCMAKE_BUILD_TYPE={btype}",
                    "-Wno-dev",
                    f"-DCMAKE_PREFIX_PATH={cmake_search_dir}",
                    "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
                    generator], shell=False, check=True)

subprocess.run(["cmake", "--build", build_path, "--config", btype], shell=False, check=True)

from .csrc import *

'''
decorator for CM code:
    options="-cmc -mdump_asm -g2"

'''
def source(options=""):
    import inspect
    def _cl_kernel(f):
        frame = inspect.currentframe().f_back
        src_lines, line_no = inspect.getsourcelines(f)
        src = "\n"*(line_no + 1) + f()
        return kernels(src, options)
    return _cl_kernel


import json
class ChromeTraceDumpper:
    def __init__(self, filename):
        self.filename = filename

    def __enter__(self):
        self.f = open(self.filename, 'w')
        self.f.write('''{"schemaVersion" : 1, "traceEvents" : [\n''')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.f.write(json.dumps({
            "name" : "Profiler End",
            "ph" : "i",
            "s" : "g",
            "pid" : "Traces",
            "tid" : "Trace OV Profiler",
            "ts" : 0
        }))
        self.f.write("\n]}")
        self.f.close()
        print(f"[ChromeTraceDumpper]: {self.filename} is dumpped!")

    def phb(self, name, cat, _id, pid, tid, begin_us, end_us, args = None):
        # name + cat + id is the unique key to match phb with phe
        ph_begin = {
            "ph" : "b",
            "cat" : cat,
            "name" : name,
            "id" : _id,
            "pid" : pid,
            "tid" : tid,
            "ts" : begin_us,
        }
        if args:
            ph_begin["args"] = args
        ph_end = {
            "ph" : "e",
            "cat" : cat,
            "name" : name,
            "id" : _id,
            "pid" : pid,
            "tid" : tid,
            "ts" : end_us
        }
        self.f.write(json.dumps(ph_begin))
        self.f.write(",\n")
        self.f.write(json.dumps(ph_end))
        self.f.write(",\n")
    def phX(self, name, cat, pid, tid, begin_us, end_us, args = None):
        phX = {
            "ph" : "X",
            "name" : name,
            "cat" : cat,
            "pid" : pid,
            "tid" : tid,
            "ts" : begin_us,
            "dur" : end_us - begin_us
        }
        if args:
            phX["args"] = args
        self.f.write(json.dumps(phX))
        self.f.write(",\n")
