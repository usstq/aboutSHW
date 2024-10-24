#include <string>
#include <fstream>
#include <streambuf>

#define CL_HPP_ENABLE_EXCEPTIONS

#include "common.hpp"

// https://github.com/intel/pti-gpu

int main(void) {
    // https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroups.html
    //
    //
    select_default_platform({"cl_intel_subgroups","cl_intel_required_subgroup_size"});

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

    // execution
    const int32_t B = getenv("B", 1);
    const int32_t H = getenv("H", 28);
    const int32_t Lq = getenv("Lq", 8416);
    const int32_t Lk = getenv("Lk", 8416);
    const int32_t S = getenv("S", 128);
    const int32_t GQA_SIZE = getenv("GQA_SIZE", 4);
    const int32_t SG_SCALE_FACTOR = getenv("SG_SCALE_FACTOR", 2);
    const int32_t TARGET_SEQ_LEN_BLOCK_SIZE = getenv("TARGET_SEQ_LEN_BLOCK_SIZE", 16);
    #define SUBGROUP_SIZE 16

    const cl::NDRange GWS({B*H, int(Lq/TARGET_SEQ_LEN_BLOCK_SIZE), SG_SCALE_FACTOR*S});
    const cl::NDRange LWS({1, 1, SG_SCALE_FACTOR*S});

    auto gws = GWS.get();
    auto lws = LWS.get();
    ECOUT("GWS={", *gws, ",", *(gws+1), ",", *(gws+2), "},",
          "LWS={", *lws, ",", *(lws+1), ",", *(lws+2), "}");
    kernel.show_info("sdpa_opt_multi_tokens_6761455398808095608_0_0__sa", LWS, SUBGROUP_SIZE);

    tensorND<workitem_info> winfo({static_cast<size_t>(B*H), static_cast<size_t>(int(Lq/TARGET_SEQ_LEN_BLOCK_SIZE)), static_cast<size_t>(SG_SCALE_FACTOR*S)}, workitem_info());
    const std::vector<int32_t> shape_info_data = {
        // input0 query
        B, H, 1, 1, 1, 1, Lq, S,
        // input1 key
        B, int(H/GQA_SIZE), 1, 1, 1, 1, Lk, S, 0, 0,
        // input2 value
        B, int(H/GQA_SIZE), 1, 1, 1, 1, Lk, S, 0, 0,
        // input3 attn_mask
        1, 1, 1, 1, 1, 1, Lq, Lk,
        // input4 scale
        // output
        B, H, 1, 1, 1, 1, Lq, S
    };
    tensorND<int32_t> shape_info;
    shape_info.resize(std::vector<size_t>{shape_info_data.size()}, std::vector<size_t>{1}, shape_info_data.data());

    tensorND<cl_half> query({static_cast<size_t>(B), static_cast<size_t>(H), static_cast<size_t>(Lq), static_cast<size_t>(S)}, 1.0f);
    tensorND<cl_half> key({static_cast<size_t>(B), static_cast<size_t>(H), static_cast<size_t>(Lk), static_cast<size_t>(S)}, 1.0f);
    tensorND<cl_half> value({static_cast<size_t>(B), static_cast<size_t>(H), static_cast<size_t>(Lk), static_cast<size_t>(S)}, 1.0f);

    tensorND<cl_half> scale({1}, 1.0f);
    tensorND<cl_half> attn_mask({static_cast<size_t>(B), 1, static_cast<size_t>(Lq), static_cast<size_t>(Lk)}, 0);

    tensorND<cl_float> exp_sums({4}, 0.0f);
    tensorND<cl_float> max_logits({4}, 0.0f);
    tensorND<cl_float> tmp_out({2}, 0.0f);

    tensorND<cl_half> output({static_cast<size_t>(B), static_cast<size_t>(H), static_cast<size_t>(Lq), static_cast<size_t>(S)}, 0.0f);

    cl::EnqueueArgs enqArgs(GWS, LWS);
    auto evt = kernel.call("sdpa_opt_multi_tokens_6761455398808095608_0_0__sa", enqArgs, shape_info.to_gpu(),
                            query.to_gpu(), key.to_gpu(), value.to_gpu(),
                            attn_mask.to_gpu(), scale.to_gpu(), output.to_gpu(),
                            exp_sums.to_gpu(), max_logits.to_gpu(), tmp_out.to_gpu(), winfo.to_gpu());
    evt.wait();


    ECOUT("=========================== end of execution ================================");
    cl_ulong last_evt_ns = 0;
    auto delta_ns = [&](cl_ulong ns){
        auto delta = ns - last_evt_ns;
        last_evt_ns = ns;
        return Nanoseconds(delta);
    };

    winfo.to_cpu();

    ECOUT("CL_PROFILING_COMMAND_QUEUED  :  ", delta_ns(evt.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>()));
    ECOUT("CL_PROFILING_COMMAND_SUBMIT  : +", delta_ns(evt.getProfilingInfo<CL_PROFILING_COMMAND_SUBMIT>()));
    ECOUT("CL_PROFILING_COMMAND_START   : +", delta_ns(evt.getProfilingInfo<CL_PROFILING_COMMAND_START>()));
    ECOUT("CL_PROFILING_COMMAND_END     : +", delta_ns(evt.getProfilingInfo<CL_PROFILING_COMMAND_END>()));
    ECOUT("CL_PROFILING_COMMAND_COMPLETE: +", delta_ns(evt.getProfilingInfo<CL_PROFILING_COMMAND_COMPLETE>()));

    auto latency_ns = (evt.getProfilingInfo<CL_PROFILING_COMMAND_END>() - evt.getProfilingInfo<CL_PROFILING_COMMAND_START>());
    size_t num_ops_per_workitem = TARGET_SEQ_LEN_BLOCK_SIZE*Lk*2;
    ECOUT("num_ops_per_workitem = ", (double)num_ops_per_workitem/1024/1024/1024, " GFLOPS, latency = ", (double)latency_ns / 1000 / 1000, " ms");
    workitem_info::Dump(winfo, latency_ns, num_ops_per_workitem, SUBGROUP_SIZE);

    return 0;
}