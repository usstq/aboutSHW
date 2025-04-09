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

struct onednn_matmul {
    matmul m_prim;
    memory::desc m_wei_md;
    memory::data_type m_w_type;
    memory::data_type m_a_type; // activation dtype
    memory::dim m_K;
    memory::dim m_N;
    memory::dim m_M;
    memory::dim m_K_groups;
    dnnl::engine m_engine;
    dnnl::stream m_stream;

    primitive_attr attr;
    post_ops postops;

    onednn_matmul(memory::data_type act_dtype, memory::data_type weight_dtype, int batch_size, int ic, int oc, int ic_group_size = -1) {
        m_a_type = act_dtype;
        m_w_type = weight_dtype;
        m_K_groups = 0;
        m_K = ic;
        m_N = oc;
        m_M = DNNL_RUNTIME_DIM_VAL;
        if (batch_size > 0) {
            // jit-gemm kernel only support static batch size
            m_M = batch_size;
        }
        if (ic_group_size >= 0) {
            w_scale(ic_group_size).w_zp(ic_group_size).fpmath_f16();
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

        memory::desc bin_mul_md = memory::desc(memory::dims({batch_size, per_oc ? m_N : 1}), m_a_type, memory::format_tag::ab);
        postops.append_binary(algorithm::binary_mul, bin_mul_md);
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

        memory::desc src_md = memory::desc(memory::dims({m_M, m_K}), m_a_type, memory::format_tag::ab);
        memory::desc dst_md = memory::desc(memory::dims({m_M, m_N}), m_a_type, memory::format_tag::ab);
        //memory::desc wei_md = memory::desc(memory::dims({m_K, m_N}), m_w_type, memory::format_tag::any);

        // use fixed weight-layout to prevent shape-dependent weight-layout changes
        memory::desc wei_md = memory::desc(memory::dims({m_K, m_N}), m_w_type, memory::format_tag::ba);

        m_engine = onednn_context::engine();
        m_stream = onednn_context::stream();

        // Create primitive descriptor.
        auto matmul_pd = matmul::primitive_desc(m_engine, src_md, wei_md, dst_md, attr);

        // Pre-packed weights stored as int8_t
        m_wei_md = matmul_pd.weights_desc();

        // Create the primitive.
        m_prim = matmul(matmul_pd);
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
    onednn_matmul(memory::data_type act_dtype, memory::data_type weight_dtype, int batch, int ic, int oc, int ic_group_size, type t) : onednn_matmul(act_dtype, weight_dtype, batch, ic, oc, ic_group_size) {
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
};

struct onednn_linear {
    std::shared_ptr<const onednn_matmul> mm;
    memory weight;
    memory scale;
    memory zp;
    matmul m_prim;
    memory::dim m_K;
    memory::dim m_N;
    memory::dim m_batch;
    memory::data_type m_a_type;
    int bin_post_id;
    dnnl::engine m_engine;
    dnnl::stream m_stream;

    static onednn_linear create(
              memory::data_type act_dtype, memory::data_type weight_dtype, int batch, int ic, int oc, int ic_group_size, onednn_matmul::type t,
              memory::data_type dtype,
              tensor& data, // external weight
              tensor& scale,
              tensor& zp) {
        auto mm = make_cacheable<onednn_matmul>(act_dtype, weight_dtype, batch, ic, oc, ic_group_size, t);
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
        return linear;
    }

    void forward(const tensor& a, tensor& c, tensor& bin_input) {
        memory::dim M = a.get_shape()[0];

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
        //args.insert({DNNL_ARG_BIAS, bias_mem});
        args.insert({DNNL_ARG_DST, dst_mem});

        if (scale) {
            args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, scale});
        }
        if (zp) {
            args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, zp});
        }
        if (bin_input) {
            auto bin_mem = dnnl::ocl_interop::make_memory(rt_bin_md, m_engine, ocl_interop::memory_kind::usm, (void *)(bin_input));
            args.insert({DNNL_ARG_ATTR_MULTIPLE_POST_OP(bin_post_id) | DNNL_ARG_SRC_1, bin_mem});
        }
        m_prim.execute(m_stream, args);
    }
};


#if 0
struct MoE {
    onednn_matmul mlp_up;
    onednn_matmul mlp_gate;
    onednn_matmul mlp_down;
    cl_kernels ocl_kernels;

    cl_kernel slicing2d;
    cl_kernel index_add_;

    int m_hidden_size;

    MoE() = default;
    static MoE create(memory::data_type act_dtype,
                      memory::data_type weight_dtype,
                        int hidden_size,
                        int moe_intermediate_size,
                        int k_group_size) {
        MoE moe;
        mode.m_hidden_size = hidden_size;
        bool is_quantized = (weight_dtype != memory::data_type::f16);
        moe.mlp_up = onednn_matmul::create(act_dtype, weight_dtype, hidden_size, moe_intermediate_size, k_group_size, is_quantized, is_quantized, false, true);
        moe.mlp_gate = onednn_matmul::create(act_dtype, weight_dtype, hidden_size, moe_intermediate_size, k_group_size, is_quantized, is_quantized, true, false);
        moe.mlp_down = onednn_matmul::create(act_dtype, weight_dtype, moe_intermediate_size, hidden_size, k_group_size, is_quantized, is_quantized, false, false);
        // load OCL code
        std::ifstream t("moe_ocl.cl");
        std::stringstream buffer;
        buffer << t.rdbuf();
        moe.ocl_kernels.setup(buffer.str(), "", "");
        moe.slicing2d = moe.ocl_kernels.get_kernel("slicing2d");
        moe.index_add_ = moe.ocl_kernels.get_kernel("index_add_");
        return moe;
    }

    struct ExpertMLP {
        onednn_linear up;
        onednn_linear gate;
        onednn_linear down;

        // down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        void forward(tensor& x, tensor& dst, tensor& temp_up, tensor& temp_gate) {
            gate.forward(x, temp_gate, tensor{});
            up.forward(x, temp_up, temp_gate);
            down.forward(temp_up, dst);
        }
    };
    std::vector<ExpertMLP> experts;

    void set_expert(int index, memory::data_type dtype,
                    tensor& gate_data, tensor& gate_scale, tensor& gate_zp,
                    tensor& up_data, tensor& up_scale, tensor& up_zp
                    tensor& down_data, tensor& down_scale, tensor& down_zp) {
        if (experts.size() < index)
            experts.resize(index);
        expert[index].up = mlp_up.get_linear(dtype, gate_data, gate_scale, gate_zp);
        expert[index].gate = mlp_gate.get_linear(dtype, up_data, up_scale, up_zp);
        expert[index].down = mlp_down.get_linear(dtype, down_data, down_scale, down_zp);
        return expert;
    }
/*

    # Loop over all available experts in the model and perform the computation on each expert
    for expert_idx in range(self.num_experts):
        expert_layer = self.experts[expert_idx]
        # idx    is the top-index,(top1, top2, ...) 
        # top_x  is the token-index
        #
        idx, top_x = torch.where(expert_mask[expert_idx])
        if idx.shape[0] != 0:
            print("================expert_idx=", expert_idx)
            print("expert_mask[expert_idx] ::::: ", expert_mask[expert_idx].shape)
            print(expert_mask[expert_idx])
            print("idx ::::: ", idx.shape, list(idx))
            print("top_x ::::: ", top_x.shape, list(top_x))

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)

            # hidden_states[i,:]  [batch*seq, 2048]

            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
*/
    memory alloc_mem() {
        memory::desc md = memory::desc(dims, dtype, fmt);
        memory ret = dnnl::ocl_interop::make_memory(md, onednn_context::engine(), ocl_interop::memory_kind::usm);
        
    }

    void forward(tensor& hidden_states, tensor& expert_mask, tensor& routing_weights, tensor& final_hidden_states) {
        //hidden_states
        memory temp_up;
        memory temp_gate;
        memory current_state;
        memory current_routing_w;
        memory final_hidden_states;
        for(int expert_idx = 0; expert_idx < experts.size(); expert_idx++) {
            // check expert mask using CPU to rule-out experts
            // expert_mask[expert_id, topk_id, token_id]
            // expert_mask[i,j,k] == 1 means expert i is selected by token k as the top-j

            // current_state: [total_tokens, m_hidden_size]
            //      extract current_state from hidden_states according to expert_mask[i,:,:]
            // current_routing_w : [total_tokens, 1]
            //      extract current_routing_w from routing_weights
            ocl_kernels.set_args(slicing2d, );
            ocl_kernels.enqueue(slicing2d, {}, {});
            experts[expert_idx].forward(current_state, current_state, temp_up, temp_gate);

            ocl_kernels.set_args(index_add_, );
            ocl_kernels.enqueue(index_add_, {}, {});
            // scatter update current_state back to final_hidden_states
        }
    }
};

#endif

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

    sycl_queue.submit([&](sycl::handler& h) {
        h.memcpy(ret.get_data_handle(), p_host, numel * host_dt.itemsize());
    });
    sycl_queue.wait();
    return ret;
}

#if 0
py::array tensor::to_numpy_f16(const memory& mem) {
    // this shouldn't be a very frequent operation which requires optimizations
    // so we just allocate
    py::array ret(dt, shape);
    py::buffer_info info = ret.request();
    auto* p_host = reinterpret_cast<uint8_t*>(info.ptr);

    // make sure data is ready
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
        .def("forward", &onednn_linear::forward);

    py::class_<memory>(m, "onednn_memory")
        .def(py::init())
        .def(py::init(&to_memory));


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
        .value("f32", memory::data_type::f32);

    //py::class_<onednn_matmul>(m, "onednn_matmul")
    //    .def(py::init<>(&onednn_matmul::create))
    //    .def("get_linear", &onednn_matmul::get_linear);
}

