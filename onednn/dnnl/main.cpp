#include <iostream>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;


#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_ocl.hpp"

using namespace dnnl;

struct context {
    engine eng;
    stream strm;
    context() : eng(engine::kind::cpu, 0), strm(eng){
    }
};

context g_ctx;

template <typename... TS>
void _write_all(std::ostream& os, TS&&... args) {
    int dummy[sizeof...(TS)] = {(os << std::forward<TS>(args), 0)...};
    (void)dummy;
}
#define OPENVINO_ASSERT(cond, ...)                                                      \
    if (!(cond)) {                                                                      \
        std::stringstream ss;                                                           \
        _write_all(ss, __FILE__, ":", __LINE__, " ", #cond, " failed:", ##__VA_ARGS__); \
        std::cout << "\033[31m" << ss.str() << "\033[0m" << std::endl;                  \
        throw std::runtime_error(ss.str());                                             \
    }

struct onednn_matmul {
    primitive m_prim;
    memory::desc m_input_md;
    memory::desc m_output_md;
    memory::desc m_wei_md;
    memory::desc m_sc_md;
    memory::desc m_zp_md;
    memory::desc m_bin_md;
    memory::data_type m_w_type;
    memory::data_type m_a_type; // activation dtype
    memory::data_type m_sc_dtype;
    memory::data_type m_zp_dtype;
    memory::dim m_K;
    memory::dim m_N;
    memory::dim m_M;
    memory::dim m_K_groups;
    primitive_attr attr;
    post_ops postops;
    int bin_post_id = -1;

    const bool m_use_ip = true;
    onednn_matmul() = default;

    onednn_matmul& init(memory::data_type act_dtype, memory::data_type weight_dtype, int batch_size, int ic, int oc, int ic_group_size = -1) {
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
            w_scale(ic_group_size).w_zp(ic_group_size);//.fpmath_f16();
        }
        m_input_md = memory::desc(memory::dims({m_M, m_K}), act_dtype, memory::format_tag::ab);
        m_output_md = memory::desc(memory::dims({m_M, m_N}), act_dtype, memory::format_tag::ab);        
        return *this;
    }

    onednn_matmul& w_scale(int k_group_size) {
        if (m_use_ip) {
            m_sc_dtype = memory::data_type::f32;
            if (k_group_size <= 0) {
                m_K_groups = 1;
            } else {
                OPENVINO_ASSERT((k_group_size % 32) == 0);
                OPENVINO_ASSERT((m_K % k_group_size) == 0);
                m_K_groups = m_K / k_group_size;
            }
            attr.set_scales_dims(DNNL_ARG_WEIGHTS, {m_N, m_K_groups}, m_sc_dtype);
            //m_sc_md = memory::desc({m_N, m_K_groups}, m_sc_dtype, memory::format_tag::ba);
            m_sc_md = memory::desc({m_K_groups, m_N}, m_sc_dtype, memory::format_tag::ab);
        } else {
            m_sc_dtype = memory::data_type::f32;
            if (k_group_size <= 0) {
                m_K_groups = 1;
                // per-OC, no grouping in K dimension
                attr.set_scales(DNNL_ARG_WEIGHTS, (0 << 0) + (1 << 1), {1}, m_sc_dtype);
            } else {
                OPENVINO_ASSERT((k_group_size % 32) == 0);
                OPENVINO_ASSERT((m_K % k_group_size) == 0);
                m_K_groups = m_K / k_group_size;
                attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) + (1 << 1), {k_group_size, 1}, m_sc_dtype);
            }
            m_sc_md = memory::desc({m_K_groups, m_N}, m_sc_dtype, memory::format_tag::ba);

        }
        return *this;
    }

    onednn_matmul& w_zp(int k_group_size) {
        if (m_use_ip) {
            m_zp_dtype = memory::data_type::u8;
            if (k_group_size <= 0) {
                OPENVINO_ASSERT(m_K_groups == 1);
                attr.set_zero_points(DNNL_ARG_WEIGHTS, (0 << 0) + (1 << 1), {1}, m_zp_dtype);
            } else {
                OPENVINO_ASSERT(m_K_groups = (m_K / k_group_size));
                attr.set_zero_points(DNNL_ARG_WEIGHTS, (1 << 0) + (1 << 1), {k_group_size, 1}, m_zp_dtype);
            }
            attr.set_zero_points_dims(DNNL_ARG_WEIGHTS,  {m_N, m_K_groups}, m_zp_dtype);
            // dtype & layout must be choosen according to kernel's capability
            //m_zp_md = memory::desc({m_N, m_K_groups}, m_zp_dtype, memory::format_tag::ba);
            m_zp_md = memory::desc({m_K_groups, m_N}, m_zp_dtype, memory::format_tag::ab);
        } else {
            m_zp_dtype = memory::data_type::s8;
            if (k_group_size <= 0) {
                OPENVINO_ASSERT(m_K_groups == 1);
                attr.set_zero_points(DNNL_ARG_WEIGHTS, (0 << 0) + (1 << 1), {1}, m_zp_dtype);
            } else {
                OPENVINO_ASSERT(m_K_groups = (m_K / k_group_size));
                attr.set_zero_points(DNNL_ARG_WEIGHTS, (1 << 0) + (1 << 1), {k_group_size, 1}, m_zp_dtype);
            }
            // dtype & layout must be choosen according to kernel's capability
            m_zp_md = memory::desc({m_K_groups, m_N}, m_zp_dtype, memory::format_tag::ab);
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
        OPENVINO_ASSERT(bin_post_id < 0);
        memory::dim batch_size = m_M;
        if (batch_size == DNNL_RUNTIME_DIM_VAL)
            batch_size = 1024*1024; // big enough fake static batch

        m_bin_md = memory::desc(memory::dims({batch_size, per_oc ? m_N : 1}), m_a_type, memory::format_tag::ab);
        postops.append_binary(algorithm::binary_mul, m_bin_md);
        bin_post_id = postops.len() - 1;
        return *this;
    }

    onednn_matmul& post_op_sum(float scale = 1.f, int32_t zero_point = 0) {
        postops.append_sum(scale, zero_point, memory::data_type::undef);
        return *this;
    }

    void create(memory::desc exist_wei_md = {}) {
        const engine &aengine = g_ctx.eng;
        if (postops.len() > 0) {
            attr.set_post_ops(postops);
        }
        memory::desc src_md = memory::desc(memory::dims({m_M, m_K}), m_a_type, memory::format_tag::ab);
        memory::desc dst_md = memory::desc(memory::dims({m_M, m_N}), m_a_type, memory::format_tag::ab);

        // use fixed weight-layout to prevent shape-dependent weight-layout changes
        //memory::desc wei_md = memory::desc(memory::dims({m_K, m_N}), m_w_type, memory::format_tag::ba);

        if (m_use_ip) {
            bool use_exist_wei_md = exist_wei_md && exist_wei_md.get_ndims() > 0;
            memory::desc wei_md = use_exist_wei_md ? exist_wei_md : memory::desc(memory::dims({m_N, m_K}), m_w_type, memory::format_tag::any);
            //memory::desc wei_md = memory::desc(memory::dims({m_N, m_K}), m_w_type, memory::format_tag::any);
            //std::cout << __LINE__ << "," << m_M << "," << m_N<< "," << m_K << "," << std::endl;
            auto ip_md = inner_product_forward::primitive_desc(aengine, dnnl::prop_kind::forward_inference,
                                                               src_md, wei_md, dst_md, attr);
            
            m_wei_md = ip_md.weights_desc();
            m_prim = inner_product_forward(ip_md);
        } else {
            memory::desc wei_md = memory::desc(memory::dims({m_K, m_N}), m_w_type, memory::format_tag::ba);
            auto matmul_pd = matmul::primitive_desc(aengine, src_md, wei_md, dst_md, attr);
            m_wei_md = matmul_pd.weights_desc();
            m_prim = matmul(matmul_pd);
        }
    }

    void exec(memory& src_mem,
              memory& dst_mem,
              memory& weight,
              memory& scale,
              memory& zp,
              memory& bin_mem) {
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
        if (bin_mem) {
            /*
            memory::desc rt_bin_md;
            if (mm->bin_per_row) {
                rt_bin_md = memory::desc(memory::dims({m_M, 1}), m_a_type, memory::format_tag::ab);
            } else {
                rt_bin_md = memory::desc(memory::dims({m_M, m_N}), m_a_type, memory::format_tag::ab);
            }            
            auto bin_mem = memory(rt_bin_md, m_engine, (void *)(bin_input));
            */
            args.insert({DNNL_ARG_ATTR_MULTIPLE_POST_OP(bin_post_id) | DNNL_ARG_SRC_1, bin_mem});
        }
        m_prim.execute(g_ctx.strm, args);
    }
};


memory memory_from_numpy(const py::array& b) {
    py::buffer_info info = b.request();
    /*
    std::cout << "info.format : " << info.format
              << " char_:" << b.dtype().char_()
              << " kind:" << b.dtype().kind() << std::endl;
    */
    memory::dims shape;
    memory::dims strides;
    int numel = 1;
    int expect_stride = 1;
    shape.resize(info.ndim);
    strides.resize(info.ndim);
    for (int i = info.ndim - 1; i >= 0; --i) {
        numel *= info.shape[i];
        shape[i] = info.shape[i];
        strides[i] = info.strides[i] / info.itemsize;
        OPENVINO_ASSERT(strides[i] == expect_stride, "Only plain & compact layout is supported");
        expect_stride *= shape[i];
    }
    memory::data_type dtype;
    switch(b.dtype().char_()) {
        case 'f' : dtype = memory::data_type::f32; break;
        case 'e' : dtype = memory::data_type::f16; break;
        case 'i' : dtype = memory::data_type::s32; break;
        case 'b' : dtype = memory::data_type::s8; break;
        case 'B' : dtype = memory::data_type::u8; break;
        default: OPENVINO_ASSERT(false, "Unsupported numpy array dtype : ", b.dtype().char_())
    }
    //dt = b.dtype();
    memory::desc md(shape, dtype, strides);
    memory mem(md, g_ctx.eng);
    memcpy(mem.get_data_handle(), info.ptr, info.size * info.itemsize);
    return mem;
}

memory memory_from_numpy2(const memory::desc &md, const engine & eng, const py::array& b) {
    py::buffer_info info = b.request();
    //std::cout << "info.format : " << info.format << std::endl;
    return memory(md, eng, info.ptr);
}

py::array as_numpy(memory& mem) {
    pybind11::dtype dt;
    auto md = mem.get_desc();
    switch(md.get_data_type()) {
        case memory::data_type::s8: dt = pybind11::dtype("b"); break;
        case memory::data_type::u8: dt = pybind11::dtype("B"); break;
        case memory::data_type::f32: dt = pybind11::dtype("f"); break;
        case memory::data_type::s32: dt = pybind11::dtype("i"); break;
        case memory::data_type::f16: dt = pybind11::dtype("e"); break;
        default: OPENVINO_ASSERT(false);
    }
    auto strides = md.get_strides();
    for(auto& s : strides) s *= dt.itemsize();
    //std::cout << ">>>>>>>>> " << reinterpret_cast<uintptr_t>(mem.get_data_handle()) << std::endl;
    return py::array(dt, md.get_dims(), strides, mem.get_data_handle());
}

memory to_u4(memory& src) {
    auto src_md = src.get_desc();
    OPENVINO_ASSERT(src_md.get_data_type() == memory::data_type::u8);
    OPENVINO_ASSERT(src_md.get_inner_nblks() == 0);
    OPENVINO_ASSERT(src_md.get_padded_dims() == src_md.get_dims());
    memory dst({src_md.get_dims(), memory::data_type::u4, src_md.get_strides()}, g_ctx.eng);
    auto* psrc = reinterpret_cast<uint8_t*>(src.get_data_handle());
    auto* pdst = reinterpret_cast<uint8_t*>(dst.get_data_handle());
    int cnt = 1;
    for(auto& v : src_md.get_dims()) cnt *= v;
    for(int i = 0; i < cnt; i+=2) {
        *pdst++ = (psrc[i] & 0xF) | (psrc[i+1] << 4);
    }
    return dst;
}

memory reorder_to(memory& src, const memory::desc& md) {
    memory dst(md, g_ctx.eng);
    dnnl::reorder(src, dst).execute(g_ctx.strm, src, dst);
    g_ctx.strm.wait();
    return dst;
}

std::ostream& operator<<(std::ostream& os, const memory::dims& dims) {
    const char *sep = "";
    os << "[";
    for (auto& d : dims) {
        os << sep << d;
        sep = ",";
    }
    os << "]";
    return os;
}
std::ostream& operator<<(std::ostream& os, const memory::data_type& dt) {
    switch(dt) {
        case memory::data_type::f32: os << "f32"; break;
        case memory::data_type::f16: os << "f16"; break;
        case memory::data_type::bf16: os << "bf16"; break;
        case memory::data_type::s32: os << "s32"; break;
        case memory::data_type::u8: os << "u8"; break;
        case memory::data_type::s8: os << "s8"; break;
        case memory::data_type::u4: os << "u4"; break;
        case memory::data_type::s4: os << "s4"; break;
    }
    return os;
}
std::ostream& operator<<(std::ostream& os, const memory::desc& md) {
    auto ndims = md.get_ndims();

    auto dims = md.get_dims();
    os << "memory::desc(" << dims << "," << md.get_data_type();    
    auto padded = md.get_padded_dims();
    if (padded != dims)
        os << ",padded_dims:" << padded;
    if (md.get_inner_nblks()) {
        os << ",inner_blks:";
        auto idx = md.get_inner_idxs();
        auto dims = md.get_inner_blks();
        for(int i = 0; i < idx.size(); i++) {
            os << static_cast<char>(idx[i] + 'a') << dims[i];
        }
    }
    os << ",strides:" << md.get_strides();
    os << ")";    
    return os;
}

PYBIND11_MODULE(csrc, m) {
    // construct memory from numpy with specified memory-descriptor
    // 
    py::enum_<memory::data_type>(m, "data_type", py::arithmetic())
        .value("s4", memory::data_type::s4)
        .value("u4", memory::data_type::u4)
        .value("s8", memory::data_type::s8)
        .value("u8", memory::data_type::u8)
        .value("s32", memory::data_type::s32)
        .value("f16", memory::data_type::f16)
        .value("f32", memory::data_type::f32);

    py::enum_<memory::format_tag>(m, "format_tag", py::arithmetic())
        .value("any", memory::format_tag::any)
        .value("ab", memory::format_tag::ab)
        .value("ba", memory::format_tag::ba);

    py::class_<memory::desc>(m, "memory_desc")
        .def(py::init<>())
        .def(py::init<const memory::dims&, memory::data_type, const memory::dims &, bool>(),
             py::arg(), py::arg(), py::arg(), py::arg() = false)
        .def(py::init<const memory::dims&, memory::data_type, memory::format_tag, bool>(),
            py::arg(), py::arg(), py::arg(), py::arg() = false)
        .def("__repr__", [](const memory::desc & md) {
            std::stringstream ss;
            ss << md;
            return ss.str();
        });

    py::class_<memory>(m, "memory")
#if 0
        .def_buffer([](memory &m) -> py::buffer_info {
            auto md = m.get_desc();
            auto strides = md.get_strides();
            auto dtype_size = 4;
            for(auto& s : strides) s *= dtype_size;
            return py::buffer_info(
                m.get_data_handle(),                    /* Pointer to buffer */
                sizeof(float),                          /* Size of one scalar */
                py::format_descriptor<float>::format(), /* Python struct-style format descriptor */
                2,                                      /* Number of dimensions */
                md.get_dims(),                           /* Buffer dimensions */
                strides
            );
        })
#endif
        .def(py::init<>())
        .def(py::init<>([](const memory::desc &md){
            return memory(md, g_ctx.eng);
         }))
        .def(py::init<>(&memory_from_numpy))
        .def_property_readonly("md", &memory::get_desc)
        .def("numpy", &as_numpy)
        .def("reorder", &reorder_to)
        .def("to_u4", &to_u4)
        .def("__repr__", [](const memory & mem) {
            std::stringstream ss;
            ss << "memory:" << reinterpret_cast<uintptr_t>(mem.get_data_handle())
               << " " << mem.get_desc();
            return ss.str();
        });

    py::enum_<engine::kind>(m, "engine_kind", py::arithmetic())
        .value("any", engine::kind::any)
        .value("gpu", engine::kind::gpu)
        .value("cpu", engine::kind::cpu);

    py::class_<engine>(m, "engine")
        .def(py::init<engine::kind, size_t>());
    
    py::class_<onednn_matmul>(m, "onednn_matmul")
        .def(py::init<>())
        .def("init", &onednn_matmul::init)
        .def("w_scale", &onednn_matmul::w_scale)
        .def("w_zp", &onednn_matmul::w_zp)
        .def("post_op_silu", &onednn_matmul::post_op_silu)
        .def("post_op_bin_mul", &onednn_matmul::post_op_bin_mul)
        .def("post_op_sum", &onednn_matmul::post_op_sum)
        .def("create", &onednn_matmul::create)
        .def("exec", &onednn_matmul::exec)
        .def_readonly("input_md", &onednn_matmul::m_input_md)
        .def_readonly("output_md", &onednn_matmul::m_output_md)
        .def_readonly("wei_md", &onednn_matmul::m_wei_md)
        .def_readonly("sc_md", &onednn_matmul::m_sc_md)
        .def_readonly("zp_md", &onednn_matmul::m_zp_md)
        .def_readonly("bin_md", &onednn_matmul::m_bin_md)
        .def_readonly("w_type", &onednn_matmul::m_w_type)
        .def_readonly("a_type", &onednn_matmul::m_a_type)
        .def_readonly("sc_dtype", &onednn_matmul::m_sc_dtype)
        .def_readonly("zp_type", &onednn_matmul::m_zp_dtype)
        .def_readonly("a_type", &onednn_matmul::m_a_type)
        .def_readonly("ic", &onednn_matmul::m_K)
        .def_readonly("oc", &onednn_matmul::m_N)
        .def_readonly("batch", &onednn_matmul::m_M)
        .def_readonly("ic_groups", &onednn_matmul::m_K_groups)
        .def("__repr__", [](const onednn_matmul & mm) {
            std::stringstream ss;
            ss << "onednn_matmul: batch,oc,ic=" << mm.m_M << "," << mm.m_N << "," << mm.m_K << "g" << mm.m_K_groups << "\n"
               << "\t input:" << mm.m_input_md << "\n"
               << "\t output:" << mm.m_output_md << "\n"
               << "\t weight:" << mm.m_wei_md << "\n"
               << "\t scale:" << mm.m_sc_md << "\n"
               << "\t zp:" << mm.m_zp_md << "\n"
               ;
            return ss.str();
        });
}