#define CL_HPP_ENABLE_EXCEPTIONS

#include "common.hpp"



int main(void) {
    // https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroups.html
    //
    //
    select_default_platform({"cl_intel_subgroups","cl_intel_required_subgroup_size"});

    // flushing denormals to zero on CPU side
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

    auto M = getenv("M", 20);
    auto N = getenv("N", 16);
    auto LM = getenv("LM", 4);
    auto LN = getenv("LN", 8);

    tensorND<int> ids({M, N, 8}, -1);
    tensorND<int> info({16}, -1);

    // https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_C.html#work-item-functions
    // 
    CLkernels kernel(R"CLC(
        __attribute__((intel_reqd_sub_group_size(8)))
        __kernel void fill_id(
                __global int * ids,
                __global int * info,
                int stride) {

            // (get_global_id(1) - get_global_offset(1)) * get_global_size(0) + (get_global_id(0) - get_global_offset(0))
            int i = get_global_linear_id();

            ids[i*8 + 0] = get_global_id(0);
            ids[i*8 + 1] = get_global_id(1);
            ids[i*8 + 2] = get_local_id(0);
            ids[i*8 + 3] = get_local_id(1);
            ids[i*8 + 4] = get_sub_group_id();
            ids[i*8 + 5] = get_sub_group_local_id();

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
    std::cout << "ids = " << ids.repr(precision, width) << "\n";

    return 0;
}