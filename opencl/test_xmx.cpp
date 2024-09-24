#define CL_HPP_ENABLE_EXCEPTIONS

#include "common.hpp"

int main(void)
{
    select_default_platform({"cl_intel_subgroups","cl_intel_subgroup_matrix_multiply_accumulate"});

    // flushing denormals to zero on CPU side
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

    CLkernels kernel(R"CLC(
        __kernel void test_ker(__global float* y, int size) {
            int i = get_global_id(0) * 4;
            y[i+0] = get_global_id(0);
            y[i+1] = get_group_id(0);
            y[i+2] = get_local_id(0);
            y[i+3] = -1;
            short a=0;
            int8 b = 1;

            y[i+1] = intel_sub_group_bf16_bf16_matrix_mad_k16(short  a, int8 b, y[i]);
        }
            )CLC");
    
    size_t N = 32;
    tensor2D<float> out(N, 4);

    kernel.call("test_ker", cl::EnqueueArgs({10,1}, {6,1}), out.to_gpu(), N);

    out.to_cpu();

    std::cout << "output = " << out.repr() << "\n";
    return 0;
}
            