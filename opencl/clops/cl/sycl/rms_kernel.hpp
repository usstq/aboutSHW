#pragma once

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/math.hpp>

namespace cldnn::sycl::details {

template<typename Type>
inline ::sycl::event rms_kernel(::sycl::queue& queue,
    const Type* in_buf, const Type* weight, Type* out_buf, const size_t bs, const size_t channel_num, float epsilon) {
    auto p = std::getenv("USE_SYCL");
    if (p && p[0] == '2') {
    return queue.submit([=](::sycl::handler& h) {
            namespace ns_s = ::sycl;
            namespace ns_e = ::sycl::ext::intel::esimd;
            ns_s::stream out(65536, 128, h);
            const int VL = 32;
            const int sg_num = 8;

            h.parallel_for(::sycl::nd_range<1>(bs * sg_num, sg_num), [=](::sycl::nd_item<1> index) [[intel::sycl_explicit_simd]] {
                // __local float variance[16];
                ns_e::slm_init<sg_num * sizeof(float)>();
                auto g = index.get_group();
                int row = g[0];
                int id_local = index.get_local_id();
                auto sg = index.get_sub_group();
                int id_sg = sg.get_group_id();
                auto input = in_buf + row * channel_num;
                auto output = out_buf + row * channel_num;
                ns_e::simd<float, VL> local_var(0);
    
                int i = id_sg * VL;
                for (; i < channel_num; i += VL * sg_num) {
                    ns_e::simd<Type, VL> a = ns_e::convert<float>(ns_e::block_load<Type, VL>(input + i));
                    local_var += a * a;
                }
                // TODO: handle tails(channel_num % VL != 0)
                float b = ns_e::reduce<float>(local_var, std::plus<>());
                ns_e::slm_scalar_store(id_sg * sizeof(float), b);
                ns_e::barrier();
                ns_e::simd<float, sg_num> local_var2 = ns_e::slm_block_load<float, sg_num>(0);
                float all_variance = ns_e::reduce<float>(local_var2, std::plus<>());
                float scale = ns_e::rsqrt((all_variance / channel_num) + epsilon);
                for (i = id_sg * VL; i < channel_num; i += VL * sg_num) {
                    ns_e::simd<Type, VL> o = ns_e::block_load<Type, VL>(input + i) * scale * ns_e::block_load<Type, VL>(weight + i);
                    o.copy_to(output + i);
                }
        });
    });
    } else {
    const size_t WG_SIZE = 256;
    return queue.submit([=](::sycl::handler& h) {
            // __local float variance[16];
            ::sycl::local_accessor<float, 1> variance(::sycl::range(WG_SIZE / 32), h);
            ::sycl::stream out(65536, 128, h);

            h.parallel_for(::sycl::nd_range<1>(bs * WG_SIZE, WG_SIZE), [=](::sycl::nd_item<1> index) [[intel::reqd_sub_group_size(32)]] {
            auto g = index.get_group();
            int row = g[0];
            int id_local = index.get_local_id();
            auto sg = index.get_sub_group();
            int id_sg = sg.get_group_id();
            int id_sg_local = sg.get_local_id();
            auto input = in_buf + row * channel_num;
            auto output = out_buf + row * channel_num;
            float local_var = 0;

            for (int i = id_local; i < channel_num; i += WG_SIZE) {
                // half/fp16 has very limited range: ±65,504
                // which may cause square result to overflow,
                // the square must be done in fp32
                float x = (float)input[i];
                local_var += x * x;
            }

            local_var = ::sycl::reduce_over_group(sg, local_var, ::sycl::plus<>());
            // local_var = sub_group_reduce_add(local_var);
            if (id_sg_local == 0)
                variance[id_sg] = local_var;
            // barrier(CLK_LOCAL_MEM_FENCE);
            index.barrier(::sycl::access::fence_space::local_space);
            if (id_sg_local < WG_SIZE / 32)
                local_var = variance[id_sg_local];
            else
                local_var = 0;
            // float all_variance = sub_group_reduce_add(local_var);
            float all_variance = ::sycl::reduce_over_group(sg, local_var, ::sycl::plus<>());
            float scale = ::sycl::rsqrt((all_variance / channel_num) + epsilon);

            for (int i = id_local; i < channel_num; i += WG_SIZE) {
                output[i] = input[i] * scale * weight[i];
            }
        });
    });
    }
}

}