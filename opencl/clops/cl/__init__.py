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
                 r"C:\\Program Files (x86)\\Intel\\oneAPI\\compiler\\latest\\bin",
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
generator="" # when there is no Ninja install

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

subprocess.run(["cmake", "--build", build_path, "--config", btype, "-j8"], shell=False, check=True)
subprocess.run(["cmake", "--install", build_path, "--config", btype, "--prefix", dir_path], shell=False, check=True)

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

# ================================================================
# Boilerplate
import numpy as np
np.random.seed(0)
np.set_printoptions(linewidth=1024)

# ================================================================
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

class SGTracer:
    code = r'''
        ulong __attribute__((overloadable)) intel_get_cycle_counter( void );
        uint __attribute__((overloadable)) intel_get_active_channel_mask( void);
        uint __attribute__((overloadable)) intel_get_hw_thread_id( void );
        uint __attribute__((overloadable)) intel_get_slice_id( void );
        uint __attribute__((overloadable)) intel_get_dual_subslice_id( void );
        uint __attribute__((overloadable)) intel_get_subslice_id( void );
        uint __attribute__((overloadable)) intel_get_eu_id( void );
        uint __attribute__((overloadable)) intel_get_eu_thread_id( void );
        void __attribute__((overloadable)) intel_eu_thread_pause( uint value );

        inline void SGTracer_begin(__global ulong** psg_info) {
            if (*psg_info && get_sub_group_local_id() == 0) {
                uint wg_id = get_group_id(0) + (get_group_id(1) + get_group_id(2)*get_num_groups(1)) * get_num_groups(0);
                uint sg_id = wg_id * get_num_sub_groups() + get_sub_group_id();
                *psg_info += sg_id * 3;

                uint hw_tid = intel_get_hw_thread_id();
                uint slice_id = intel_get_slice_id();
                uint sub_slice_id = intel_get_subslice_id();
                uint eu_id = intel_get_eu_id();
                uint eu_tid = intel_get_eu_thread_id();
                ulong geuid = (slice_id & 0xF);               // Render-Slice
                geuid = (geuid << 4) | (sub_slice_id & 0xF); // xe-core
                geuid = (geuid << 4) | (eu_id & 0xF);
                geuid = (geuid << 4) | (eu_tid & 0xF);
                geuid = (geuid << 32) | (wg_id);

                (*psg_info)[0] = geuid;
                (*psg_info)[1] = intel_get_cycle_counter();
            }
        }
        inline void SGTracer_end(__global ulong** psg_info) {
            if (*psg_info && get_sub_group_local_id() == 0) {
                (*psg_info)[2] = intel_get_cycle_counter();
            }
        }
    '''

    @classmethod
    def dump(cls, sg_info, json_file_name = "ocl.json"):
        with ChromeTraceDumpper(json_file_name) as ctd:
            #ctd.phb("name","cat", 1, "100", "1", 0, 1000, {"EV": 1.78, "xxx":"hwllo"})
            #ctd.phb("name","cat", 2, "100", "1", 100, 1200)
            #ctd.phX("name","catXXX", "100", "1", 100, 1200)
            ts_base = sg_info[:,1].min()
            sg_info[:,1:] -= ts_base

            for sg_id in range(sg_info.shape[0]):
                loc = int(sg_info[sg_id,0])
                wg_id = loc & 0xFFFFFFFF; loc = loc >> 32
                eu_tid = loc & 0xF; loc = loc >> 4
                eu_id  = loc & 0xF; loc = loc >> 4
                sub_slice_id  = loc & 0xF; loc = loc >> 4
                slice_id  = loc & 0xF; loc = loc >> 4

                cycle_start = sg_info[sg_id,1]
                cycle_end = sg_info[sg_id,2]
                if 0:
                    ctd.phb(name = f"{wg_id}",
                            cat = f"{wg_id}",
                            pid = f"SubSlice:{slice_id}.{sub_slice_id}",
                            tid = f"EU-thread:{eu_id}.{eu_tid}",
                            begin_us = float(cycle_start)/1e3,
                            end_us = float(cycle_end)/1e3,
                            args = {
                                "cycle_start":int(cycle_start),
                                "cycle_end":int(cycle_end),
                                "loc" : f"{slice_id}.{sub_slice_id}.{eu_id}.{eu_tid}",
                                "work-group":wg_id,
                                "sub-group" : sg_id
                            })
                else:
                    ctd.phX(name = f"{wg_id}",
                            cat = f"{wg_id}",
                            pid = f"SubSlice:{slice_id}.{sub_slice_id}.{eu_id}",
                            tid = f"EU-thread:{eu_tid}",
                            begin_us = float(cycle_start)/1e3,
                            end_us = float(cycle_end)/1e3,
                            args = {
                                "cycle_start":int(cycle_start),
                                "cycle_end":int(cycle_end),
                                "loc" : f"{slice_id}.{sub_slice_id}.{eu_id}.{eu_tid}",
                                "work-group":wg_id,
                                "sub-group" : sg_id
                            })


class CMTracer:
    code = r'''
        CM_INLINE uint64_t _get_clock() {
            auto clk = cm_clock();
            return ((uint64_t)clk[1]) << 32 | clk[0];
        }
    
        CM_INLINE void CMTracer_begin(__global uint64_t** psg_info) {
            if (*psg_info && cm_linear_local_id() == 0) {
                *psg_info += cm_linear_group_id() * 3;

                uint gid0 = cm_group_id(0);
                uint gid1 = cm_group_id(1);
                uint gid2 = cm_group_id(2);
                uint group_id = cm_linear_group_id();
                uint64_t geuid = (gid0 & 0xFFFF);
                geuid = (geuid << 16) | (gid1 & 0xFFFF);
                geuid = (geuid << 16) | (gid2 & 0xFFFF);
                geuid = (geuid << 16) | (group_id & 0xFFFF);
                (*psg_info)[0] = geuid;
                (*psg_info)[1] = _get_clock();
            }
        }
        CM_INLINE void CMTracer_end(__global uint64_t** psg_info) {
            if (*psg_info && cm_linear_local_id() == 0) {
                (*psg_info)[2] = _get_clock();
            }
        }
    '''

    @classmethod
    def dump(cls, sg_info, gpu_freq = None, json_file_name = "ocl.json"):

        cycle2us = None
        if gpu_freq:
            cycle2us = 1e6/gpu_freq
            
        def cycle_cvt(cycle):
            if cycle2us: return cycle * cycle2us
            return cycle

        with ChromeTraceDumpper(json_file_name) as ctd:
            #ctd.phb("name","cat", 1, "100", "1", 0, 1000, {"EV": 1.78, "xxx":"hwllo"})
            #ctd.phb("name","cat", 2, "100", "1", 100, 1200)
            #ctd.phX("name","catXXX", "100", "1", 100, 1200)
            ts_base = sg_info[:,1].min()
            sg_info[:,1:] -= ts_base

            for sg_id in range(sg_info.shape[0]):
                loc = int(sg_info[sg_id,0])
                group_id = loc & 0xFFFF; loc = loc >> 16
                gid2 = loc & 0xFFFF; loc = loc >> 16
                gid1 = loc & 0xFFFF; loc = loc >> 16
                gid0 = loc & 0xFFFF; loc = loc >> 16

                cycle_start = cycle_cvt(sg_info[sg_id,1])
                cycle_end = cycle_cvt(sg_info[sg_id,2])
                if 0:
                    ctd.phb(name = f"{wg_id}",
                            cat = f"{wg_id}",
                            pid = f"SubSlice:{slice_id}.{sub_slice_id}",
                            tid = f"EU-thread:{eu_id}.{eu_tid}",
                            begin_us = float(cycle_start)/1e3,
                            end_us = float(cycle_end)/1e3,
                            args = {
                                "cycle_start":int(cycle_start),
                                "cycle_end":int(cycle_end),
                                "loc" : f"{slice_id}.{sub_slice_id}.{eu_id}.{eu_tid}",
                                "work-group":wg_id,
                                "sub-group" : sg_id
                            })
                else:
                    ctd.phX(name = f"{group_id}",
                            cat = f"{group_id}",
                            pid = f"SubSlice:{gid0}.{gid1}.{gid2}",
                            tid = f"EU-thread:{gid2}",
                            begin_us = float(cycle_start)/1e3,
                            end_us = float(cycle_end)/1e3,
                            args = {
                                "cycle_start":int(cycle_start),
                                "cycle_end":int(cycle_end),
                            })
