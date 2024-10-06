#define CL_HPP_ENABLE_EXCEPTIONS

#include "common.hpp"

// https://github.com/intel/pti-gpu

int main(void) {
    // https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroups.html
    //
    //
    select_default_platform({"cl_intel_subgroups","cl_intel_required_subgroup_size"});

    // flushing denormals to zero on CPU side
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

    auto M = getenv("M", 1);
    auto N = getenv("N", 1);
    auto K = getenv("K", 4096000);
    auto LM = getenv("LM", 32);
    auto LN = getenv("LN", 32);

    tensorND<uint64_t> ids({M, N, 16}, -1);
    tensorND<int> info({16}, -1);

    // https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_C.html#work-item-functions
    // 
    CLkernels kernel(R"CLC(
        ulong __attribute__((overloadable)) intel_get_cycle_counter( void );
        uint __attribute__((overloadable)) intel_get_slice_id( void );
        uint __attribute__((overloadable)) intel_get_subslice_id( void );
        uint __attribute__((overloadable)) intel_get_dual_subslice_id( void );
        uint __attribute__((overloadable)) intel_get_eu_id( void );

        __attribute__((intel_reqd_sub_group_size(8)))
        __kernel void fill_id(
                __global ulong * ids,
                __global int * info,
                int K) {

            // (get_global_id(1) - get_global_offset(1)) * get_global_size(0) + (get_global_id(0) - get_global_offset(0))
            int i = get_global_linear_id();

            ids[i*16 + 0] = get_group_id(0);
            ids[i*16 + 1] = get_group_id(1);
            ids[i*16 + 2] = get_local_id(0);
            ids[i*16 + 3] = get_local_id(1);
            ids[i*16 + 4] = get_sub_group_id();
            ids[i*16 + 5] = get_sub_group_local_id();
            ids[i*16 + 6] = intel_get_slice_id();
            ids[i*16 + 7] = intel_get_subslice_id();
            ids[i*16 + 8] = intel_get_eu_id();

            ids[i*16 + 9] = intel_get_cycle_counter();
                float c = 0.0f;
                float a = get_local_id(0);
                float b = get_local_id(1);
                for(int k = 0; k < K; k++) {
                    c = fma(a, b, c);
                }
            ids[i*16 + 10] = intel_get_cycle_counter() - ids[i*16 + 9];

            if (c == 0) info[0] = 1;

            if (i == 0) {
                info[0] = get_global_size(0);
                info[1] = get_global_size(1);
                info[2] = get_local_size(0);
                info[3] = get_local_size(1);
                info[4] = get_num_groups(0);
                info[5] = get_num_groups(1);
                info[6] = get_sub_group_size();
                info[7] = get_max_sub_group_size();
                info[8] = get_num_sub_groups();
                info[8] = sizeof(ulong);
            }
        }
            )CLC");

    kernel.show_info("fill_id", {1024,1024}, 3);

    int precision = 2, width = 3;

    cl::EnqueueArgs enqArgs({N, M},{LN, LM});

    kernel.call("fill_id", enqArgs, ids.to_gpu(), info.to_gpu(), N);
    ids.to_cpu();
    info.to_cpu();
    std::cout << "info = " << info.repr(precision, width) << "\n";
    //std::cout << "ids = " << ids.repr(precision, width) << "\n";
    std::cout << "===========================" << std::endl;

    ChromeTraceDumpper dumpper("ocl.json");

    struct EUWork {
        uint64_t group_id0;
        uint64_t group_id1;
        uint64_t local_id0;
        uint64_t local_id1;
        uint64_t sub_group_id;
        uint64_t sub_group_local_id;
        uint64_t slice_id;
        uint64_t sub_slice_id;
        uint64_t eu_id;
        uint64_t cycle_start;
        uint64_t cycle_dur;
        int thr_cnt = 0;
        void dump(ChromeTraceDumpper & dumpper) {
            if (thr_cnt <= 0) return;
            std::stringstream ss;
            std::stringstream ss_cat;
            std::stringstream ss_pid;
            std::stringstream ss_tid;
            ss << "kernel(" << group_id0 << "," << group_id1 << ").";
            ss_pid << "slice.subslice:" << slice_id << "." << sub_slice_id;
            //ss_tid << "(" << local_id0 << "+" << thr_cnt << "," << local_id1 << ") EU" << eu_id;
            //ss_tid << "(" << group_id0 << "," << group_id1 << ")." << sub_group_id;
            ss_tid << "(" << group_id0 << "," << group_id1 << ")." << sub_group_id << " EU_" << eu_id ;
            //ss << "(" << local_id0 << "+" << thr_cnt << "," << local_id1 << ") sub-group:" << sub_group_id << "." << sub_group_local_id;
            dumpper.phX(ss.str(), ss_cat.str(), ss_pid.str(), ss_tid.str(), cycle_start, cycle_dur);
        }
    } euwork;

    euwork.group_id0 = std::numeric_limits<uint64_t>::max();

    uint64_t min_cycle_start = std::numeric_limits<uint64_t>::max();
    for(int m = 0; m < M; m++) {
        for(int n = 0; n < N; n++) {
            auto* pw = reinterpret_cast<EUWork*>(ids.ptr(m, n));
            min_cycle_start = std::min(min_cycle_start, pw->cycle_start);
        }
    }
    for(int m = 0; m < M; m++) {
        for(int n = 0; n < N; n++) {
            auto* pw = reinterpret_cast<EUWork*>(ids.ptr(m, n));
            pw->cycle_start -= min_cycle_start;
        }
    }

    for(int m = 0; m < M; m++) {
        for(int n = 0; n < N; n++) {
            auto* pdata = ids.ptr(m, n);
            auto* pw = reinterpret_cast<EUWork*>(pdata);

            if (euwork.group_id0 == pw->group_id0 &&
                euwork.group_id1 == pw->group_id1 &&
                euwork.sub_group_id == pw->sub_group_id) {
                ASSERT(euwork.slice_id == pw->slice_id);
                ASSERT(euwork.sub_slice_id == pw->sub_slice_id);
                ASSERT(euwork.eu_id == pw->eu_id);
                ASSERT(euwork.cycle_start == pw->cycle_start);
                ASSERT(euwork.cycle_dur == pw->cycle_dur);
                euwork.thr_cnt ++;
            } else {
                euwork.dump(dumpper);
                euwork = *pw;
                euwork.thr_cnt = 1;
            }
        }
    }
    euwork.dump(dumpper);

    return 0;
}