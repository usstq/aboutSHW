# Multi-threading

Multi-threading library like OpenMP & TBB have their own overhead, which becomes an issue when kernel is light-weighted (parallel latency in dozens of micro-seconds).

the measurement of pure overhead of 100 parallel region run sequentially:

| num worker threads/cores| OpenMP(gopm) | OpenMP(iomp) | SyncThreads1 | SyncThreads2| ThreadPool |
|------------------------ | ------------ | ------------ | ------------ | ----------- | ---------- |
|             1           |     18.888   |    10.376    |     0.495    |     0.377   |            |
|             2           |     138.863  |    59.496    |     25.678   |     39.139  |            |
|             4           |     189.006  |    81.877    |     44.504   |     69.203  |            |
|             8           |     277.589  |    133.292   |     62.867   |     92.592  |            |
|            16           |     571.836  |    152.812   |     86.619   |     73.324  |            |
|            32           |     910.589  |    217.453   |     193.204  |     134.547 |            |
|            56           |     896.588  |    223.615   |     200.768  |     165.032 | 160.531    |


> Note: replacing gomp with iomp is simply done at runtime with `LD_PRELOAD=/opt/intel/oneapi/compiler/2023.1.0/linux/compiler/lib/intel64_lin/libiomp5.so` and requires no compile-time changes. (`g++ -O2 -fopenmp ./test-threading.cpp`)

we can see IOMP has much better performance or much lower overhead comparing to GOMP (per-region overhead 9us vs 2.2us), but lock-free SyncThreads can run even faster (per-region overhead down to 2us~1.6us).


# MESI-based lock-free atomic Load & Store

as pointed out at [here](https://stackoverflow.com/questions/38447226/atomicity-of-loads-and-stores-on-x86), `RMW`(read-modify-write) operations are multi-ops & non-atomic in nature (even though single instruction encoding exist:like `XADD`/`XCHG`/...). they require special LOCK# signal to be atomic, but cache-aligned load & store is atomic w/o the help of LOCK# signal, which is super-light-weighted, on x86 they are used as the [implementation of C++11's atomic load & store ](https://stackoverflow.com/questions/38447226/atomicity-of-loads-and-stores-on-x86), so we rely on atomic load & store to implement high-performance multi-thread synchronization (`SyncThreads1`/`SyncThreads2`/`ThreadPool`).



