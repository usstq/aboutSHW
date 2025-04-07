#include "common.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include <sycl/sycl.hpp>

#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_ocl.hpp"

using namespace dnnl;
extern sycl::queue sycl_queue;
extern std::vector<std::variant<cl_event, sycl::event>> all_events;

class onednn_context {
    dnnl::engine m_engine;
    dnnl::stream m_stream;
    onednn_context() {
        cl_context ocl_context = sycl::get_native<sycl::backend::opencl>(sycl_queue.get_context());
        cl_device_id ocl_device = sycl::get_native<sycl::backend::opencl>(sycl_queue.get_device());
        cl_command_queue cmd_queue = sycl::get_native<sycl::backend::opencl>(sycl_queue);
        m_engine = dnnl::ocl_interop::make_engine(ocl_device, ocl_context);
        m_stream = dnnl::ocl_interop::make_stream(m_engine, cmd_queue);
    }
    static onednn_context& get() {
        static onednn_context ctx;
        return ctx;
    }
public:
    static dnnl::stream& stream() {
        return get().m_stream;
    }
    static dnnl::engine& engine() {
        return get().m_engine;
    }
};

// https://uxlfoundation.github.io/oneDNN/dev_guide_matmul.html

// 
struct onednn_wmem {
    memory weight;
    memory scale;
    memory zp;
};

struct onednn_mm {
    matmul m_prim;
    memory::desc m_best_wei_md;
    memory::data_type m_w_type;
    memory::data_type m_a_type; // activation dtype
    memory::dim m_K;
    memory::dim m_N;
    memory::dim m_K_groups;
    dnnl::engine m_engine;
    dnnl::stream m_stream;


    // dtype: dtype of data
    // data:  f16/s8 [K, N]
    // scale: f32 [N, K//K_group_size]
    // zp   :  s8 [N, K//K_group_size]
    onednn_wmem get_wmem(memory::data_type dtype, tensor& data, tensor& scale, tensor& zp) {
        return to_wmem(dtype, static_cast<void*>(data), static_cast<void*>(scale), static_cast<void*>(zp));
    }

    onednn_wmem to_wmem(memory::data_type dtype, void* data, void* scale, void* zp) {
        onednn_wmem wmem;
        memory::desc raw_wei_md = memory::desc(memory::dims({m_K, m_N}), dtype, memory::format_tag::ab);

        if (raw_wei_md != m_best_wei_md) {
            wmem.weight = memory(m_best_wei_md, m_engine);
            std::cout << ">>>>>>>>>>>>>>>>>> weight layout changed : reorder is called (seems to be not working)" << std::endl;
            auto src_wei_mem = dnnl::ocl_interop::make_memory(
                                        raw_wei_md,
                                        m_engine,
                                        ocl_interop::memory_kind::usm,
                                        data);
            reorder cvt(src_wei_mem, wmem.weight);
            cvt.execute(m_stream, src_wei_mem, wmem.weight);
            m_stream.wait();
        } else {
            wmem.weight = dnnl::ocl_interop::make_memory(
                                        raw_wei_md,
                                        m_engine,
                                        ocl_interop::memory_kind::usm,
                                        data);
        }

        if (scale) {
            // https://uxlfoundation.github.io/oneDNN/page_weights_decompression_matmul_cpp.html
            // Quantization Group size for scales. Must be divisible by 32.
            auto wei_scale_md = memory::desc(memory::dims({m_K_groups, m_N}), memory::data_type::f32, memory::format_tag::ab);
            wmem.scale = dnnl::ocl_interop::make_memory(wei_scale_md, m_engine, ocl_interop::memory_kind::usm, scale);
            if (zp) {
                auto wei_zp_md = memory::desc(memory::dims({m_K_groups, m_N}), memory::data_type::s8, memory::format_tag::ab);
                wmem.zp = dnnl::ocl_interop::make_memory(wei_zp_md, m_engine, ocl_interop::memory_kind::usm, zp);
            }
        }
        return wmem;
    }

    static onednn_mm create(memory::data_type act_dtype, memory::data_type weight_dtype,
              int K,
              int N,
              int k_group_size,  // 0 means per-OC quantization
              bool with_wc_scales,
              bool with_wc_zp,
              bool with_silu) {
        return onednn_mm(act_dtype, weight_dtype, K, N, k_group_size, with_wc_scales, with_wc_zp, with_silu);
    }

    onednn_mm(memory::data_type act_dtype,
              memory::data_type weight_dtype,
              int K,
              int N,
              int k_group_size,  // 0 means per-OC quantization
              bool with_wc_scales,
              bool with_wc_zp,
              bool with_silu) {
        m_a_type = act_dtype;
        m_w_type = weight_dtype;
        m_K_groups = 0;
        if (k_group_size == 0) k_group_size = K;
        m_K_groups = K / k_group_size;
        m_K = K;
        m_N = N;
        m_engine = onednn_context::engine();
        m_stream = onednn_context::stream();

        memory::desc src_md = memory::desc(memory::dims({DNNL_RUNTIME_DIM_VAL, m_K}), m_a_type, memory::format_tag::ab);
        memory::desc dst_md = memory::desc(memory::dims({DNNL_RUNTIME_DIM_VAL, m_N}), m_a_type, memory::format_tag::ab);

        m_best_wei_md = memory::desc(memory::dims({m_K, m_N}), m_w_type, memory::format_tag::any);

        const float alpha = 1.f;
        const float beta = 0.f;
        post_ops postops;
        if (with_silu) {
            postops.append_eltwise(algorithm::eltwise_swish, alpha, beta);
        }
        primitive_attr attr;

        if (postops.len() > 0) {
            attr.set_post_ops(postops);
        }

        if (with_wc_scales) {
            // Create attributes and indicate that the alpha and zero points are
            // runtime parameters
            // Set scales with multiple scales along K and N dimensions and with groups along K.
            attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) + (1 << 1), {k_group_size, 1}, memory::data_type::f32);
            // Set fpmath mode with `apply_to_int=true` to apply fpmath mode behavior to
            // integral primitives (in this example, matmul).
            attr.set_fpmath_mode(fpmath_mode::f16, true);
            if (with_wc_zp) {
                // Set a single zero point with s8 data type.
                attr.set_zero_points(DNNL_ARG_WEIGHTS, (1 << 0) + (1 << 1), {k_group_size, 1}, memory::data_type::s8);
            }
        }

        // Create primitive descriptor.
        auto matmul_pd = matmul::primitive_desc(m_engine, src_md, m_best_wei_md, dst_md, attr);

        // Pre-packed weights stored as int8_t
        m_best_wei_md = matmul_pd.weights_desc();

        // Create the primitive.
        m_prim = matmul(matmul_pd);
    }

    void forward(const tensor& a, const onednn_wmem& wmem, tensor& c) {
        memory::dim M = a.get_shape()[0];
        memory::desc rt_src_md = memory::desc(memory::dims({M, m_K}), m_a_type, memory::format_tag::ab);
        memory::desc rt_dst_md = memory::desc(memory::dims({M, m_N}), m_a_type, memory::format_tag::ab);
        auto src_mem = dnnl::ocl_interop::make_memory(rt_src_md, m_engine, ocl_interop::memory_kind::usm, (void *)(a));
        //auto weights_mem = dnnl::ocl_interop::make_memory(m_weights_md, m_engine, ocl_interop::memory_kind::usm, (void *)(w));
        auto dst_mem = dnnl::ocl_interop::make_memory(rt_dst_md, m_engine, ocl_interop::memory_kind::usm, (void *)(c));
        //auto bias_mem = memory(bias_md, m_engine);

        std::unordered_map<int, memory> args;
        args.insert({DNNL_ARG_SRC, src_mem});
        args.insert({DNNL_ARG_WEIGHTS, wmem.weight});
        //args.insert({DNNL_ARG_BIAS, bias_mem});
        args.insert({DNNL_ARG_DST, dst_mem});

        if (wmem.scale) {
            args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, wmem.scale});
        }
        if (wmem.zp) {
            args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, wmem.zp});
        }

        m_prim.execute(m_stream, args);
    }
};


void init_ops_onednn(py::module_& m) {
    py::class_<onednn_wmem>(m, "onednn_wmem").def(py::init());

    py::enum_<memory::data_type>(m, "onednn_dtype", py::arithmetic())
        .value("s4", memory::data_type::s4)
        .value("u4", memory::data_type::u4)
        .value("s8", memory::data_type::s8)
        .value("u8", memory::data_type::u8)
        .value("f16", memory::data_type::f16)
        .value("f32", memory::data_type::f32);

    py::class_<onednn_mm>(m, "onednn_mm")
        .def(py::init<>(&onednn_mm::create))
        .def("get_wmem", &onednn_mm::get_wmem)
        .def("forward", &onednn_mm::forward);
}

