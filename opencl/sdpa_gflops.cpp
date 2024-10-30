// compile command: 
// g++ -O2 ./sdpa_gflops.cpp -lOpenCL -o sdpa_gflops
// OCL_EnablePreviewFeatures=1 ./sdpa_gflops
//s
// 
#include <string>
#include <fstream>
#include <streambuf>
#include <cmath>

#define CL_HPP_ENABLE_EXCEPTIONS

#include "common.hpp"

void CL_CALLBACK NotifyFunction( const char * pErrInfo, const void * pPrivateInfo, size_t size, void * pUserData )
{
    if( pErrInfo != NULL )
    {
        std::cerr << ANSI_COLOR_ERROR << "[cl_intel_driver_diagnostics]:" << pErrInfo << ANSI_COLOR_RESET << std::endl;;
    }
};

// https://github.com/intel/pti-gpu

int main(void) {
    // https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroups.html
    //
    //
    auto selected_platform = select_default_platform({"cl_intel_subgroups","cl_intel_required_subgroup_size"});

    // https://community.intel.com/t5/OpenCL-for-CPU/private-memory-spills-and-loop-unrolling-on-HD-Graphics/td-p/1116378
    // https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_driver_diagnostics.txt
    //
    // https://community.intel.com/t5/OpenCL-for-CPU/CL-KERNEL-SPILL-MEM-SIZE-INTEL-interpretation/td-p/1120834
    cl_context_properties properties[] =
    {
        CL_CONTEXT_SHOW_DIAGNOSTICS_INTEL, (cl_context_properties)CL_CONTEXT_DIAGNOSTICS_LEVEL_GOOD_INTEL | CL_CONTEXT_DIAGNOSTICS_LEVEL_BAD_INTEL | CL_CONTEXT_DIAGNOSTICS_LEVEL_NEUTRAL_INTEL,
        CL_CONTEXT_PLATFORM, (cl_context_properties)selected_platform(),
        0
    };
    cl::Context::setDefault(cl::Context(CL_DEVICE_TYPE_GPU, &properties[0], NotifyFunction));

    cl::CommandQueue::setDefault(cl::CommandQueue(cl::QueueProperties::Profiling));

    // flushing denormals to zero on CPU side
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

    // https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_C.html#work-item-functions
    // 
    std::ifstream cl_file("cl_kernels/sdpa.cl");
    std::string cl_source((std::istreambuf_iterator<char>(cl_file)),
                     std::istreambuf_iterator<char>());
    // std::cout << cl_source << std::endl;
    CLkernels kernel(cl_source.c_str());

    const std::string kernel_name("sdpa_opt_multi_tokens_6761455398808095608_0_0__sa");

    // execution
    #define SUBGROUP_SIZE 16
    #define S 128    // head_size
    #define SG_SCALE_FACTOR 2
    #define SOURCE_SEQ_LEN_BLOCK_SIZE (S*SG_SCALE_FACTOR)
    #define TARGET_SEQ_LEN_BLOCK_SIZE 16
    const int32_t B = getenv("B", 1);
    const int32_t Hq = getenv("Hq", 28);
    const int32_t Hk = getenv("Hq", 7);
    const int32_t Lq = getenv("Lq", 8416);
    const int32_t Lk = getenv("Lk", 8416);

    assert(Hq % Hk == 0);     // implied
    assert(S % SUBGROUP_SIZE == 0);     // implied
    assert(TARGET_SEQ_LEN_BLOCK_SIZE == SUBGROUP_SIZE);   // implied

    const cl::NDRange GWS({B*Hq, int(Lq/TARGET_SEQ_LEN_BLOCK_SIZE), SG_SCALE_FACTOR*S});
    const cl::NDRange LWS({1, 1, SG_SCALE_FACTOR*S});

    auto gws = GWS.get();
    auto lws = LWS.get();
    ECOUT("GWS={", *gws, ",", *(gws+1), ",", *(gws+2), "},",
          "LWS={", *lws, ",", *(lws+1), ",", *(lws+2), "}");
    assert(SOURCE_SEQ_LEN_BLOCK_SIZE == *(lws+2));       // implied
    kernel.show_info(kernel_name.c_str(), LWS, SUBGROUP_SIZE);

    tensorND<workitem_info> winfo({static_cast<size_t>(B*Hq), static_cast<size_t>(int(Lq/TARGET_SEQ_LEN_BLOCK_SIZE)), static_cast<size_t>(SG_SCALE_FACTOR*S)}, workitem_info());
    const std::vector<int32_t> shape_info_data = {
        // input0 query
        B, Hq, 1, 1, 1, 1, Lq, S,
        // input1 key
        B, Hk, 1, 1, 1, 1, Lk, S, 0, 0,
        // input2 value
        B, Hk, 1, 1, 1, 1, Lk, S, 0, 0,
        // input3 attn_mask
        1, 1, 1, 1, 1, 1, Lq, Lk,
        // input4 scale
        // output
        B, Hq, 1, 1, 1, 1, Lq, S
    };
    tensorND<int32_t> shape_info;
    shape_info.resize(std::vector<size_t>{shape_info_data.size()}, std::vector<size_t>{1}, shape_info_data.data());

    tensorND<cl_half> query({static_cast<size_t>(B), static_cast<size_t>(Hq), static_cast<size_t>(Lq), static_cast<size_t>(S)}, 1.0f);
    tensorND<cl_half> key({static_cast<size_t>(B), static_cast<size_t>(Hq), static_cast<size_t>(Lk), static_cast<size_t>(S)}, 1.0f);
    tensorND<cl_half> value({static_cast<size_t>(B), static_cast<size_t>(Hq), static_cast<size_t>(Lk), static_cast<size_t>(S)}, 1.0f);

    tensorND<cl_half> scale({1}, 1.0f);
    tensorND<cl_half> attn_mask({static_cast<size_t>(B), 1, static_cast<size_t>(Lq), static_cast<size_t>(Lk)}, 0);

    tensorND<cl_float> exp_sums({4}, 0.0f);
    tensorND<cl_float> max_logits({4}, 0.0f);
    tensorND<cl_float> tmp_out({2}, 0.0f);

    tensorND<cl_half> output({static_cast<size_t>(B), static_cast<size_t>(Hq), static_cast<size_t>(Lq), static_cast<size_t>(S)}, 0.0f);

    cl::EnqueueArgs enqArgs(GWS, LWS);
    cl::Event evt;
    auto kernel_nargs = [&](const std::string& kernel_name) {
        auto k = kernel.kernel_map[kernel_name];
            return k.getInfo<CL_KERNEL_NUM_ARGS>();
    };
    auto nargs = kernel_nargs(kernel_name);
    if (nargs <= 10) {
        evt = kernel.call(kernel_name.c_str(), enqArgs, shape_info.to_gpu(),
                            query.to_gpu(), key.to_gpu(), value.to_gpu(),
                            attn_mask.to_gpu(), scale.to_gpu(), output.to_gpu(),
                            exp_sums.to_gpu(), max_logits.to_gpu(), tmp_out.to_gpu());
    } else {
        evt = kernel.call(kernel_name.c_str(), enqArgs, winfo.to_gpu(), shape_info.to_gpu(),
                            query.to_gpu(), key.to_gpu(), value.to_gpu(),
                            attn_mask.to_gpu(), scale.to_gpu(), output.to_gpu(),
                            exp_sums.to_gpu(), max_logits.to_gpu(), tmp_out.to_gpu());
    }
    evt.wait();


    ECOUT("=========================== end of execution ================================");
    cl_ulong last_evt_ns = 0;
    auto delta_ns = [&](cl_ulong ns) {
        auto delta = ns - last_evt_ns;
        last_evt_ns = ns;
        return Nanoseconds(delta);
    };

    ECOUT("CL_PROFILING_COMMAND_QUEUED  :  ", delta_ns(evt.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>()));
    ECOUT("CL_PROFILING_COMMAND_SUBMIT  : +", delta_ns(evt.getProfilingInfo<CL_PROFILING_COMMAND_SUBMIT>()));
    ECOUT("CL_PROFILING_COMMAND_START   : +", delta_ns(evt.getProfilingInfo<CL_PROFILING_COMMAND_START>()));
    ECOUT("CL_PROFILING_COMMAND_END     : +", delta_ns(evt.getProfilingInfo<CL_PROFILING_COMMAND_END>()));
    ECOUT("CL_PROFILING_COMMAND_COMPLETE: +", delta_ns(evt.getProfilingInfo<CL_PROFILING_COMMAND_COMPLETE>()));

    auto latency_ns = (evt.getProfilingInfo<CL_PROFILING_COMMAND_END>() - evt.getProfilingInfo<CL_PROFILING_COMMAND_START>());
    // size_t num_ops_per_workitem = 133120; //TARGET_SEQ_LEN_BLOCK_SIZE*Lk*2;
    size_t num_ops_mm1_per_workitem = std::ceil(float(Lk)/SOURCE_SEQ_LEN_BLOCK_SIZE) * (S/SUBGROUP_SIZE) * (TARGET_SEQ_LEN_BLOCK_SIZE * SUBGROUP_SIZE);
    size_t num_ops_mm2_per_workitem = std::ceil(float(Lk)/SOURCE_SEQ_LEN_BLOCK_SIZE) * std::ceil(float(SOURCE_SEQ_LEN_BLOCK_SIZE)/(SUBGROUP_SIZE*SG_SCALE_FACTOR)) * (TARGET_SEQ_LEN_BLOCK_SIZE * SUBGROUP_SIZE);
    size_t num_ops_per_workitem = num_ops_mm1_per_workitem + num_ops_mm2_per_workitem;
    ECOUT("num_ops_per_workitem = ", num_ops_per_workitem, ", latency = ", (double)latency_ns / 1000 / 1000, " ms");
    ECOUT("[A770] vtune measured benchmark sdpa_model = 498ms latency mode, 194ms throughput mode, 31.9% XVE active, 68.1% stalled.");
    ECOUT("[A770] vtune measured unittest sdpa model kernel = 77ms, 75.3% XVE active, 24.2% stalled.");
    ECOUT("[A770] vtune measured QWen2 sdpa kernel = 82ms, 76.7% XVE active, 23.2% stalled.");

    if (nargs > 10) {
        winfo.to_cpu();
        workitem_info::Dump(winfo, latency_ns, num_ops_per_workitem, SUBGROUP_SIZE);
    }

    return 0;
}