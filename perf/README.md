
custom profiling based on Linux's [perf API](https://www.man7.org/linux/man-pages/man2/perf_event_open.2.html) is useful.


in mmap + ring-buffer mode, kernel will update mmap data time to time, and the update may happens
anywhere between the do-while code block, to avoid reading inconsistent structure from user-space,
[Seqlock](https://en.wikipedia.org/wiki/Seqlock) was used.

below code example didn't check if seq is odd (which means writer has being to write before the do-while loop) which seems to be a bug.
```c
/*
cap_user_rdpmc (since Linux 3.12)
        If the hardware supports user-space read of performance
        counters without syscall (this is the "rdpmc" instruction
        on x86), then the following code can be used to do a read:
*/
            u32 seq, time_mult, time_shift, idx, width;
            u64 count, enabled, running;
            u64 cyc, time_offset;

            do {
                seq = pc->lock;
                barrier();
                enabled = pc->time_enabled;
                running = pc->time_running;

                if (pc->cap_usr_time && enabled != running) {
                    cyc = rdtsc();
                    time_offset = pc->time_offset;
                    time_mult   = pc->time_mult;
                    time_shift  = pc->time_shift;
                }

                idx = pc->index;
                count = pc->offset;

                if (pc->cap_usr_rdpmc && idx) {
                    width = pc->pmc_width;
                    count += rdpmc(idx - 1);
                }

                barrier();
            } while (pc->lock != seq);
```

## Usage


install torch extension by `pip install -e .`.


please check [test.cpp](./test.cpp) and [test-torch.py](./test-torch.py) out 