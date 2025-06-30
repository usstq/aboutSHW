#include "common.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

#ifdef WITH_ONEDNN

#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_ocl.hpp"

using namespace dnnl;
extern ocl_queue g_queue;

class onednn_context {
    dnnl::engine m_engine;
    dnnl::stream m_stream;
    onednn_context() {
        m_engine = dnnl::ocl_interop::make_engine(g_queue.device, g_queue.context);
        m_stream = dnnl::ocl_interop::make_stream(m_engine, g_queue.queue);
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

struct onednn_matmul {
    matmul m_prim;
    memory::desc m_wei_md;
    memory::desc m_wei_scale_md;
    memory::desc m_wei_zp_md;
    memory::desc m_bias_md;
    memory::desc m_src_md;
    memory::desc m_src_scale_md;
    memory::desc m_dst_md;
    memory::desc m_bin_mul_md;
    
    memory::data_type m_w_type;
    memory::data_type m_bias_type; // bias dtype
    memory::data_type m_a_type; // activation dtype
    memory::dim m_K;
    memory::dim m_N;
    memory::dim m_M;
    memory::dim m_K_groups;
    memory::dim m_act_K_groups;
    dnnl::engine m_engine;
    dnnl::stream m_stream;

    primitive_attr attr;
    post_ops postops;

    onednn_matmul(memory::data_type act_dtype,
                  memory::data_type weight_dtype,
                  memory::data_type bias_dtype,
                  int batch_size, int ic, int oc, int ic_group_size = -1) {
        m_a_type = act_dtype;
        m_w_type = weight_dtype;
        m_bias_type = bias_dtype;
        m_K_groups = 0;
        m_act_K_groups = 0;
        m_K = ic;
        m_N = oc;
        m_M = DNNL_RUNTIME_DIM_VAL;
        if (batch_size > 0) {
            // jit-gemm kernel only support static batch size
            m_M = batch_size;
        }
        m_src_md = memory::desc(memory::dims({m_M, m_K}), m_a_type, memory::format_tag::ab);
        m_dst_md = memory::desc(memory::dims({m_M, m_N}), memory::data_type::f16, memory::format_tag::ab);
        if (m_bias_type != memory::data_type::undef) {
            m_bias_md = memory::desc(memory::dims({1, m_N}), m_bias_type, memory::format_tag::ab);
        }

        if (ic_group_size >= 0) {
            w_scale(ic_group_size);
            if (m_w_type == memory::data_type::u4) {
                // u4 requires u4 zero-points(asym) (which increases dynamic-range of u4 [0, 15] to [-15, 15])
                // where s4 imply no-zero-points(sym)
                w_zp(ic_group_size);
            }
            fpmath_f16();
        }
    }

    onednn_matmul& w_scale(int k_group_size) {
        if (k_group_size <= 0) {
            m_K_groups = 1;
            // per-OC, no grouping in K dimension
            attr.set_scales(DNNL_ARG_WEIGHTS, (0 << 0) + (1 << 1), {1}, memory::data_type::f16);
        } else {
            ASSERT((k_group_size % 32) == 0);
            ASSERT((m_K % k_group_size) == 0);
            m_K_groups = m_K / k_group_size;
            attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) + (1 << 1), {k_group_size, 1}, memory::data_type::f16);
        }
        m_wei_scale_md = memory::desc(memory::dims({m_K_groups, m_N}), memory::data_type::f16, memory::format_tag::ab);
        return *this;
    }

    onednn_matmul& w_zp(int k_group_size) {
        if (k_group_size <= 0) {
            ASSERT(m_K_groups == 1);
            attr.set_zero_points(DNNL_ARG_WEIGHTS, (0 << 0) + (1 << 1), {1}, m_w_type);
        } else {
            ASSERT(m_K_groups = (m_K / k_group_size));
            attr.set_zero_points(DNNL_ARG_WEIGHTS, (1 << 0) + (1 << 1), {k_group_size, 1}, m_w_type);
        }
        m_wei_zp_md = memory::desc(memory::dims({m_K_groups, m_N}), m_w_type, memory::format_tag::ab);
        return *this;
    }
    // input activation is also quantized (per-row & k-grouped)
    onednn_matmul& a_scale(int k_group_size) {
        ASSERT((m_K % k_group_size) == 0);
        m_act_K_groups = m_K / k_group_size;
        attr.set_scales(DNNL_ARG_SRC, (1<<0) + (1<<1), 
                dnnl::memory::dims{1, k_group_size},
                memory::data_type::f16);
        m_src_scale_md = memory::desc(memory::dims({m_M, m_act_K_groups}), memory::data_type::f16, memory::format_tag::ab);
        return *this;
    }
    onednn_matmul& fpmath_f16() {
        attr.set_fpmath_mode(fpmath_mode::f16, true);
        return *this;
    }
    onednn_matmul& post_op_silu() {
        float alpha = 1.0f;
        float beta = 0.0f;
        postops.append_eltwise(algorithm::eltwise_swish, alpha, beta);
        return *this;
    }
    onednn_matmul& post_op_bin_mul(bool per_oc = true) {
        memory::dim batch_size = m_M;
        if (batch_size == DNNL_RUNTIME_DIM_VAL)
            batch_size = 1024*1024; // big enough fake static batch

        m_bin_mul_md = memory::desc(memory::dims({batch_size, per_oc ? m_N : 1}), memory::data_type::f16, memory::format_tag::ab);
        bin_post_id = postops.len();
        postops.append_binary(algorithm::binary_mul, m_bin_mul_md);
        return *this;
    }

    onednn_matmul& post_op_sum(float scale = 1.f, int32_t zero_point = 0) {
        postops.append_sum(scale, zero_point, memory::data_type::undef);
        return *this;
    }

    void create() {
        if (postops.len() > 0) {
            attr.set_post_ops(postops);
        }
        //memory::desc wei_md = memory::desc(memory::dims({m_K, m_N}), m_w_type, memory::format_tag::any);

        // use fixed weight-layout to prevent shape-dependent weight-layout changes
        memory::desc wei_md = memory::desc(memory::dims({m_K, m_N}), m_w_type, memory::format_tag::ba);

        m_engine = onednn_context::engine();
        m_stream = onednn_context::stream();

        // Create primitive descriptor.
        auto matmul_pd = (m_bias_type == memory::data_type::undef) ? 
                   matmul::primitive_desc(m_engine, m_src_md, wei_md, m_dst_md, attr)
                 : matmul::primitive_desc(m_engine, m_src_md, wei_md, m_bias_md, m_dst_md, attr);;

        // Pre-packed weights stored as int8_t
        m_wei_md = matmul_pd.weights_desc();

        // Create the primitive.
        m_prim = matmul(matmul_pd);
    }

    void forward(tensor& dst,
                 const std::vector<tensor>& src,
                 const std::vector<tensor>& weights,
                 tensor& bias,
                 tensor& bin_input) {
        memory::dim M = src[0].get_numel()/m_K;

        ASSERT(m_M == 0 || m_M == M, "m_M=", m_M, " M=", M);

        auto src_mem = dnnl::ocl_interop::make_memory(m_src_md, m_engine, ocl_interop::memory_kind::usm, (void *)(src[0]));
        auto dst_mem = dnnl::ocl_interop::make_memory(m_dst_md, m_engine, ocl_interop::memory_kind::usm, (void *)(dst));
        auto weight_mem = dnnl::ocl_interop::make_memory(m_wei_md, m_engine, ocl_interop::memory_kind::usm, static_cast<void*>(weights[0]));

        std::unordered_map<int, memory> args;
        args.insert({DNNL_ARG_SRC, src_mem});
        args.insert({DNNL_ARG_DST, dst_mem});
        args.insert({DNNL_ARG_WEIGHTS, weight_mem});
        if (!m_bias_md.is_zero()) {
            auto mem = dnnl::ocl_interop::make_memory(m_dst_md, m_engine, ocl_interop::memory_kind::usm, (void *)(bias));
            args.insert({DNNL_ARG_BIAS, mem});
        }
        if (!m_wei_scale_md.is_zero()) {
            auto scale_mem = dnnl::ocl_interop::make_memory(m_wei_scale_md, m_engine, ocl_interop::memory_kind::usm, static_cast<void*>(weights[1]));
            args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, scale_mem});
        }
        if (!m_wei_zp_md.is_zero()) {
            auto zp_mem = dnnl::ocl_interop::make_memory(m_wei_zp_md, m_engine, ocl_interop::memory_kind::usm, static_cast<void*>(weights[2]));
            args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, zp_mem});
        }
        if (!m_src_scale_md.is_zero()) {
            auto scale_mem = dnnl::ocl_interop::make_memory(m_src_scale_md, m_engine, ocl_interop::memory_kind::usm, static_cast<void*>(src[1]));
            args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_0, scale_mem});
        }
        if (!m_bin_mul_md.is_zero()) {
            const auto& bshape = bin_input.get_shape();
            //ASSERT(bshape[0] == m_M || bshape[0] == 1);
            //ASSERT(bshape[1] == m_N || bshape[1] == m_N);
            auto bin_mem = dnnl::ocl_interop::make_memory(m_bin_mul_md, m_engine, ocl_interop::memory_kind::usm, (void *)(bin_input));
            args.insert({DNNL_ARG_ATTR_MULTIPLE_POST_OP(bin_post_id) | DNNL_ARG_SRC_1, bin_mem});
        }
        auto event = dnnl::ocl_interop::execute(m_prim, m_stream, args);
        g_queue.events.push_back(event);
        //m_prim.execute(m_stream, args);
    }

    // this creator is for predefined matmul primitive types
    enum class type {
        none,
        with_bin_mul,
        with_bin_mul_per_row,
        with_bin_mul_per_row_sum,
        with_silu,
        with_silu_bin_mul,
    };
    int bin_post_id = -1;
    bool bin_per_row = false;
    onednn_matmul(memory::data_type act_dtype,
                  memory::data_type weight_dtype,
                  memory::data_type bias_dtype,
                  int batch, int ic, int oc, int ic_group_size,
                  type t)
             : onednn_matmul(act_dtype, weight_dtype, bias_dtype, batch, ic, oc, ic_group_size) {
        if (t == type::with_bin_mul) {
            bin_post_id = 0;
            post_op_bin_mul(true);
        }
        if (t == type::with_bin_mul_per_row) {
            bin_post_id = 0;
            bin_per_row = true;
            post_op_bin_mul(false);
        }
        if (t == type::with_bin_mul_per_row_sum) {
            bin_post_id = 0;
            bin_per_row = true;
            post_op_bin_mul(false);
            post_op_sum();
        }
        if (t == type::with_silu)
            post_op_silu();
        if (t == type::with_silu_bin_mul) {
            bin_post_id = 1;
            post_op_silu();
            post_op_bin_mul(true);
        }

        create();
    }
    std::vector<uint64_t> finish() {
#ifdef DNNL_EXPERIMENTAL_PROFILING
        ASSERT(clFinish(g_queue.queue) == CL_SUCCESS);
        std::vector<uint64_t> nsecs = dnnl::get_profiling_data(m_stream, profiling_data_kind::time);
        // Reset profiler's state.
        dnnl::reset_profiling(m_stream);
        return nsecs;
#else
        return {};
#endif
    }
};

struct onednn_linear {
    std::shared_ptr<const onednn_matmul> mm;
    memory weight;
    memory scale;
    memory zp;
    memory bias;
    matmul m_prim;
    memory::dim m_K;
    memory::dim m_N;
    memory::dim m_batch;
    memory::data_type m_a_type;
    int bin_post_id;
    dnnl::engine m_engine;
    dnnl::stream m_stream;

    static onednn_linear create(
              memory::data_type act_dtype,
              memory::data_type weight_dtype,
              memory::data_type bias_dtype,
              int batch, int ic, int oc, int ic_group_size,
              onednn_matmul::type t,
              memory::data_type dtype,
              tensor& data, // external weight
              tensor& scale,
              tensor& zp,
              tensor& bias) {
        auto mm = make_cacheable<onednn_matmul>(act_dtype, weight_dtype, bias_dtype, batch, ic, oc, ic_group_size, t);
        onednn_linear linear;
        linear.mm = mm;
        linear.bin_post_id = mm->bin_post_id;
        linear.m_prim = mm->m_prim;
        linear.m_K = mm->m_K;
        linear.m_N = mm->m_N;
        linear.m_batch = batch;
        linear.m_a_type = mm->m_a_type;
        linear.m_engine = mm->m_engine;
        linear.m_stream = mm->m_stream;
        // assume raw weights are nn.Linear
        memory::desc raw_wei_md = memory::desc(memory::dims({mm->m_K, mm->m_N}), dtype, memory::format_tag::ba);

        if (raw_wei_md != mm->m_wei_md) {
            ASSERT(0);
            /*
            linear.weight = memory(mm->m_wei_md, mm->m_engine);
            std::cout << ">>>>>>>>>>>>>>>>>> weight layout changed : reorder is called (seems to be not working)" << std::endl;
            auto src_wei_mem = dnnl::ocl_interop::make_memory(
                                        raw_wei_md,
                                        mm->m_engine,
                                        ocl_interop::memory_kind::usm,
                                        static_cast<void*>(data));
            reorder cvt(src_wei_mem, linear.weight);
            cvt.execute(linear.m_stream, src_wei_mem, linear.weight);
            linear.m_stream.wait();
            */
        } else {
            linear.weight = dnnl::ocl_interop::make_memory(
                                        raw_wei_md,
                                        linear.m_engine,
                                        ocl_interop::memory_kind::usm,
                                        static_cast<void*>(data));
        }

        if (scale) {
            // https://uxlfoundation.github.io/oneDNN/page_weights_decompression_matmul_cpp.html
            // Quantization Group size for scales. Must be divisible by 32.
            auto wei_scale_md = memory::desc(memory::dims({mm->m_K_groups, mm->m_N}),
                                             memory::data_type::f16,
                                             memory::format_tag::ab);
            linear.scale = dnnl::ocl_interop::make_memory(wei_scale_md, linear.m_engine, ocl_interop::memory_kind::usm, scale);
            if (zp) {
                auto wei_zp_md = memory::desc(memory::dims({mm->m_K_groups, mm->m_N}),
                                              mm->m_w_type,
                                              memory::format_tag::ab);
                linear.zp = dnnl::ocl_interop::make_memory(wei_zp_md, linear.m_engine, ocl_interop::memory_kind::usm, zp);
            }
        }

        if (bias) {
            auto bias_md = memory::desc(memory::dims({1, mm->m_N}),
                                                        mm->m_bias_type,
                                                        memory::format_tag::ab);            
            linear.bias = dnnl::ocl_interop::make_memory(bias_md, linear.m_engine, ocl_interop::memory_kind::usm, bias);
        }
        return linear;
    }

    void forward(const tensor& a, tensor& c, tensor& bin_input) {
        memory::dim M = a.get_numel()/m_K;

        ASSERT(m_batch == 0 || m_batch == M, "m_batch=", m_batch, " M=", M);

        memory::desc rt_src_md = memory::desc(memory::dims({M, m_K}), m_a_type, memory::format_tag::ab);
        memory::desc rt_dst_md = memory::desc(memory::dims({M, m_N}), m_a_type, memory::format_tag::ab);
        memory::desc rt_bin_md;
        if (mm->bin_per_row) {
            rt_bin_md = memory::desc(memory::dims({M, 1}), m_a_type, memory::format_tag::ab);
        } else {
            rt_bin_md = memory::desc(memory::dims({M, m_N}), m_a_type, memory::format_tag::ab);
        }
        auto src_mem = dnnl::ocl_interop::make_memory(rt_src_md, m_engine, ocl_interop::memory_kind::usm, (void *)(a));
        //auto weights_mem = dnnl::ocl_interop::make_memory(m_weights_md, m_engine, ocl_interop::memory_kind::usm, (void *)(w));
        auto dst_mem = dnnl::ocl_interop::make_memory(rt_dst_md, m_engine, ocl_interop::memory_kind::usm, (void *)(c));
        //auto bias_mem = memory(bias_md, m_engine);

        std::unordered_map<int, memory> args;
        args.insert({DNNL_ARG_SRC, src_mem});
        args.insert({DNNL_ARG_WEIGHTS, weight});
        if (bias) {
            args.insert({DNNL_ARG_BIAS, bias});
        }
        args.insert({DNNL_ARG_DST, dst_mem});

        if (scale) {
            args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, scale});
        }
        if (zp) {
            args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, zp});
        }
        /*
        if (with_dyn_quant) {
            const auto& bin_shape = bin_input.get_shape();
            auto bin_md = memory::desc(memory::dims({bin_shape[0], bin_shape[1]}), memory::data_type::f16, memory::format_tag::ab);
            auto bin_mem = dnnl::ocl_interop::make_memory(bin_md, m_engine, ocl_interop::memory_kind::usm, (void *)(bin_input));
            args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_0, bin_mem});
        } else 
        */
        if (bin_input) {
            auto bin_mem = dnnl::ocl_interop::make_memory(rt_bin_md, m_engine, ocl_interop::memory_kind::usm, (void *)(bin_input));
            args.insert({DNNL_ARG_ATTR_MULTIPLE_POST_OP(bin_post_id) | DNNL_ARG_SRC_1, bin_mem});
        }
        auto event = dnnl::ocl_interop::execute(m_prim, m_stream, args);
        g_queue.events.push_back(event);
        //m_prim.execute(m_stream, args);
    }

    std::vector<uint64_t> finish() {
#ifdef DNNL_EXPERIMENTAL_PROFILING
        ASSERT(clFinish(g_queue.queue) == CL_SUCCESS);
        std::vector<uint64_t> nsecs = dnnl::get_profiling_data(m_stream, profiling_data_kind::time);
        // Reset profiler's state.
        dnnl::reset_profiling(m_stream);
        return nsecs;
#else
        return {};
#endif
    }
};

#if 0
memory to_memory(const py::array& b, memory::data_type dtype) {
    // returns an instance of A that you made using B
    py::buffer_info info = b.request();
    memory::dims dims;
    size_t numel = 1;

    for(int i = 0; i < info.ndim; i++) {
        numel *= info.shape[i];
        dims.push_back(info.shape[i]);
    }

    auto host_dt = b.dtype();
    auto* p_host = reinterpret_cast<uint8_t*>(info.ptr);

    memory::format_tag fmt;
    if (info.ndim == 1) fmt = memory::format_tag::a;
    else if (info.ndim == 2) fmt = memory::format_tag::ab;
    else if (info.ndim == 3) fmt = memory::format_tag::abc;
    else if (info.ndim == 4) fmt = memory::format_tag::abcd;
    else ASSERT(0);

    memory::desc md = memory::desc(dims, dtype, fmt);
    memory ret = dnnl::ocl_interop::make_memory(md, onednn_context::engine(), ocl_interop::memory_kind::usm);

    g_queue.memcpy_DtoH(ret.get_data_handle(), p_host, numel * host_dt.itemsize());
    return ret;
}

py::array tensor::to_numpy_f16(const memory& mem) {
    // this shouldn't be a very frequent operation which requires optimizations
    // so we just allocate
    py::array ret(dt, shape);
    py::buffer_info info = ret.request();
    auto* p_host = reinterpret_cast<uint8_t*>(info.ptr);

    // make sure data is ready
    g_queue.memcpy_DtoH(ret.get_data_handle(), p_host, numel * host_dt.itemsize());
    
    sycl_queue.submit([&](sycl::handler& h) {
        h.memcpy(p_host, p_buff.get(), numel * dt.itemsize());
    });
    sycl_queue.wait();
    return ret;
}
#endif



void init_ops_onednn(py::module_& m) {
    py::class_<onednn_linear>(m, "onednn_linear")
        .def(py::init())
        .def(py::init(&onednn_linear::create))
        .def("forward", &onednn_linear::forward)
        .def("finish", &onednn_linear::finish);
    //py::class_<memory>(m, "onednn_memory")
    //    .def(py::init())
    //    .def(py::init(&to_memory));


    py::enum_<onednn_matmul::type>(m, "onednn_matmul_type", py::arithmetic())
        .value("none", onednn_matmul::type::none)
        .value("with_bin_mul", onednn_matmul::type::with_bin_mul)
        .value("with_bin_mul_per_row", onednn_matmul::type::with_bin_mul_per_row)
        .value("with_bin_mul_per_row_sum", onednn_matmul::type::with_bin_mul_per_row_sum)
        .value("with_silu", onednn_matmul::type::with_silu)
        .value("with_silu_bin_mul", onednn_matmul::type::with_silu_bin_mul);

    py::enum_<memory::data_type>(m, "onednn_dtype", py::arithmetic())
        .value("s4", memory::data_type::s4)
        .value("u4", memory::data_type::u4)
        .value("s8", memory::data_type::s8)
        .value("u8", memory::data_type::u8)
        .value("f16", memory::data_type::f16)
        .value("f32", memory::data_type::f32)
        .value("undef", memory::data_type::undef);

    py::class_<onednn_matmul>(m, "onednn_matmul")
        .def(py::init<memory::data_type, memory::data_type, memory::data_type, int, int, int, int>())
        .def("a_scale", &onednn_matmul::a_scale)
        .def("w_scale", &onednn_matmul::w_scale)
        .def("w_zp", &onednn_matmul::w_zp)
        .def("fpmath_f16", &onednn_matmul::fpmath_f16)
        .def("post_op_silu", &onednn_matmul::post_op_silu)
        .def("post_op_bin_mul", &onednn_matmul::post_op_bin_mul)
        .def("post_op_sum", &onednn_matmul::post_op_sum)
        .def("create", &onednn_matmul::create)
        .def("forward", &onednn_matmul::forward)
        .def("finish", &onednn_matmul::finish);
}

#else
void init_ops_onednn(py::module_& m) {}
#endif