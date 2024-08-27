## [linux_perf.hpp](./linux_perf.hpp)

pure single header C++11 linux perf event API helper.

example environment var `LINUX_PERF=dump:switch-cpu:cpus=56,57:XSNP_FWD=0x04d2:XSNP_MISS=0x01d2`:
  - `dump`: dump json file to be viewed in `chrome://tracing/`
  - `switch-cpu`: capture thread switching event
  - `cpus=56,57` : only capture event on cpu 56 & 57
  - `XSNP_FWD=0x04d2` : also capture event counter of custom PMU [XSNP_FWD](https://perfmon-events.intel.com/index.html?pltfrm=spxeon.html&evnt=MEM_LOAD_L3_HIT_RETIRED.XSNP_FWD)
  - `XSNP_MISS=0x01d2` : also capture event counter of custom PMU [XSNP_MISS](https://perfmon-events.intel.com/index.html?pltfrm=spxeon.html&evnt=MEM_LOAD_L3_HIT_RETIRED.XSNP_MISS)
