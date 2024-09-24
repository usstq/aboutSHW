#define CL_HPP_ENABLE_EXCEPTIONS

#include "common.hpp"



int main(void)
{
    select_default_platform({"cl_intel_subgroups","cl_intel_required_subgroup_size"});

    // flushing denormals to zero on CPU side
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

    CLkernels kernel(R"CLC(
        __attribute__((intel_reqd_sub_group_size(8)))
        __kernel void fill_id(__global float* out, int stride, int type) {
            int x = get_global_id(0);
            int y = get_global_id(1);
            if (type == 0) out[y*stride + x] = get_group_id(0) + get_group_id(1) * 0.01f;
            if (type == 1) out[y*stride + x] = get_local_id(0) + get_local_id(1) * 0.01f;
            if (type == 2) out[y*stride + x] = get_sub_group_id() + get_sub_group_local_id() * 0.01f;
        }

            )CLC");

    kernel.show_info("fill_id", {1024,1024}, 3);

    size_t M0 = 20;
    size_t N0 = 20;
    size_t M = 16;
    size_t N = 16;

    int precision = 2, width = 6;

    cl::EnqueueArgs enqArgs({16,16},{16,16});

    tensor2D<float> out(M0, N0);

    kernel.call("fill_id", enqArgs, out.to_gpu(), N0, 0);
    out.to_cpu();
    std::cout << "get_group_id(0,1) = " << out.repr(precision, width) << "\n";

    kernel.call("fill_id", enqArgs, out.to_gpu(), N0, 1);
    out.to_cpu();
    std::cout << "get_local_id(0,1) = " << out.repr(precision, width) << "\n";

    kernel.call("fill_id", enqArgs, out.to_gpu(), N0, 2);
    out.to_cpu();
    std::cout << "get_sub_group_id()/get_sub_group_local_id() = " << out.repr(precision, width) << "\n";
    return 0;
}