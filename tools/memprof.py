import os, sys
import psutil
import inspect
import time

class Colors:
    """ ANSI color codes """
    BLACK = "\033[0;30m"
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    BROWN = "\033[0;33m"
    BLUE = "\033[0;34m"
    PURPLE = "\033[0;35m"
    CYAN = "\033[0;36m"
    LIGHT_GRAY = "\033[0;37m"
    DARK_GRAY = "\033[1;30m"
    LIGHT_RED = "\033[1;31m"
    LIGHT_GREEN = "\033[1;32m"
    YELLOW = "\033[1;33m"
    LIGHT_BLUE = "\033[1;34m"
    LIGHT_PURPLE = "\033[1;35m"
    LIGHT_CYAN = "\033[1;36m"
    LIGHT_WHITE = "\033[1;37m"
    BOLD = "\033[1m"
    FAINT = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    BLINK = "\033[5m"
    NEGATIVE = "\033[7m"
    CROSSED = "\033[9m"
    END = "\033[0m"


class memlog:
    def __init__(self):
        self.TRACE_INTO_FILES = {}
        self.pid = os.getpid()
        self._TWO_20 = float(2 ** 20)
        self.process = psutil.Process(self.pid)
        self.include_children = True
        self.last_rss, self.last_vms, self.last_shr = self.get_memory_MB()
        print(f"{Colors.LIGHT_RED} ************* memlog created! ************* {Colors.END}")
        self.last_line_no = 0
        self.last_file_name = None

    def get_memory_MB(self):
        minfo = self.process.memory_info()
        vms, rss, shr = minfo.vms, minfo.rss, minfo.shared
        if self.include_children:
            for child in self.process.children(recursive=True):
                minfo = child.memory_info()
                vms += minfo.vms
                rss += minfo.rss
                shr += minfo.shared
        return vms/self._TWO_20, rss/self._TWO_20, shr/self._TWO_20

    def __call__(self):
        vms, rss, shr = self.get_memory_MB()
        caller = inspect.getframeinfo(inspect.stack()[1][0])
        delta_rss = rss - self.last_rss
        delta_vms = vms - self.last_vms
        delta_shr = shr - self.last_shr
        self.last_rss = rss
        self.last_vms = vms
        self.last_shr = shr
        print(f"{Colors.LIGHT_CYAN} vms:rss:shr  {vGB(vms)}, {vGB(rss)}, {vGB(shr)} GB (delta={dGB(delta_vms)},{dGB(delta_rss)},{dGB(delta_shr)} GB) {Colors.END} {caller.filename}:{caller.lineno}")

    def trace_lines(self, frame, event, arg):
        if event != 'line': return
        co = frame.f_code
        func_name = co.co_name
        line_no = frame.f_lineno
        filename = co.co_filename
        # print(f"trace_lines: before executing {filename}:{line_no} ...")
        vms, rss, shr = self.get_memory_MB()
        cur_time = time.time()

        def vGB(v):
            if v < 0.1:
                return "      "
            if v > 1024:
                return f"{v/1024:6.3f}G"
            return f"{v:5.1f}M"

        for prefix_path in self.TRACE_INTO_FILES:
            if filename.startswith(prefix_path):
                thr_MB = self.TRACE_INTO_FILES[prefix_path]["thr_MB"]

        def dGB(v):
            if v > thr_MB:
                return f"{Colors.LIGHT_RED}{vGB(v)}{Colors.LIGHT_CYAN}"
            return vGB(v)

        if self.last_file_name:
            fname1 = self.last_file_name
            fname2 = filename
            if fname1 == fname2:
                if line_no == self.last_line_no + 1:
                    line_info = f"{fname1}:{self.last_line_no}"
                else:
                    line_info = f"{fname1}:{self.last_line_no}~{line_no}"
            else:
                line_info = f"{fname1}:{self.last_line_no}~{fname2}:{line_no}"

            delta_rss = rss - self.last_rss
            delta_vms = vms - self.last_vms
            delta_shr = shr - self.last_shr
            delta_time = cur_time - self.last_time
            if abs(delta_rss) > 0.1 or abs(delta_vms) > 0.1 or abs(delta_shr) > 0.1:
                print(f"{Colors.LIGHT_CYAN} VMS:{dGB(delta_vms)}/{vGB(vms)} RSS:{dGB(delta_rss)}/{vGB(rss)} SHR:{dGB(delta_shr)}/{vGB(shr)}  +{delta_time:.3f} sec  {Colors.END} {line_info}")

        self.last_line_no = line_no
        self.last_file_name = filename
        self.last_rss = rss
        self.last_vms = vms
        self.last_shr = shr
        self.last_time = cur_time


    def trace_calls(self, frame, event, arg):
        co = frame.f_code
        func_name = co.co_name
        line_no = frame.f_lineno
        filename = co.co_filename
        #print(f' Calls {func_name} @ {filename}:{line_no}')
        for prefix_path in self.TRACE_INTO_FILES:
            if filename.startswith(prefix_path):
                # Trace into this function
                return self.trace_lines
        return

    def register(self, prefix_path = None, thr_MB=1):
        caller_frameinfo = inspect.stack()[1]
        if prefix_path is None:
            prefix_path = caller_frameinfo.filename
        #print(caller_frameinfo.frame)
        self.TRACE_INTO_FILES[prefix_path] = {"thr_MB":thr_MB}
        caller_frameinfo.frame.f_trace = self.trace_calls
        sys.settrace(self.trace_calls)
        print(f"register {prefix_path}:{caller_frameinfo.lineno} thr_MB={thr_MB}")


memlogger = memlog()



if __name__ == "__main__":
    import numpy as np
    memlogger.register(2)

    def my_func():
        a = np.zeros([1024, 1024], dtype=np.int8)
        b = np.zeros([1024*10, 1024], dtype=np.int8)
        a = a + 1
        del b
        return a

    my_func()
    my_func()
    