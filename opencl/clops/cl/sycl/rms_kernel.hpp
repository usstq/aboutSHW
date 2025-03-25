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
    constexpr size_t SG_SIZE = 32;
    return queue.submit([=](::sycl::handler& h) {
            // __local float variance[16];
            //::sycl::local_accessor<float, 1> variance(::sycl::range(WG_SIZE / 32), h);
            //::sycl::stream out(65536, 128, h);

            // following lambda passed to h.parallel_for is a SYCL kernel, it can also be a named functor
            //
            //  https://github.khronos.org/SYCL_Reference/iface/defining-kernels.html#defining-kernels
            //  https://github.khronos.org/SYCL_Reference/iface/nd_range.html#sycl-nd-range
            //  https://github.khronos.org/SYCL_Reference/iface/nd_item.html#sycl-nd-item
            //
            //  each work-group handles one row
            h.parallel_for(::sycl::nd_range<1>(bs * WG_SIZE, WG_SIZE), [=](::sycl::nd_item<1> index) [[intel::reqd_sub_group_size(SG_SIZE)]] {
            auto g = index.get_group();
            int row = g[0];
            int id_local = index.get_local_id();
            auto sg = index.get_sub_group();
            // https://github.khronos.org/SYCL_Reference/iface/sub_group.html#get-group-id
            //      Return an id representing the index of the sub-group within the work-group.
            //      Since the work-items that compose a sub-group are chosen in an implementation defined way,
            //      the returned sub-group id cannot be used to identify a particular work-item in the global nd-range.
            //      Rather, the returned sub-group id is merely an abstract identifier of the sub-group containing
            //      this work-item.
            int id_sg = sg.get_group_id();
            // https://github.khronos.org/SYCL_Reference/iface/sub_group.html#get-local-id
            //      Return a SYCL id representing the calling work-item’s position within the sub-group.
            int id_sg_local = sg.get_local_id();
            auto input = in_buf + row * channel_num;
            auto output = out_buf + row * channel_num;
            float local_var = 0;

            // some item in WG tail may loop one more time than other item
            //  https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#_work_group_data_parallel_kernels
            //
            // Portable device code must not assume that :
            //  - work-items within a sub-group execute in any particular order,
            //  - that work-groups are subdivided into sub-groups in a specific way,
            //  - that the work-items within a sub-group provide specific forward progress guarantees.
            //
            for (int i = id_local; i < channel_num; i += WG_SIZE) {
                // half/fp16 has very limited range: ±65,504
                // which may cause square result to overflow,
                // the square must be done in fp32
                float x = (float)input[i];
                local_var += x * x;
            }

#if 1
            // https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:group-functions
            // https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:algorithms
            //
            // call Group algorithms library function : reduce_over_group directly on group-level
            // which has performance equal to following complex manual version.
            // which do not use shared local memory at all
            //
            float all_variance = ::sycl::reduce_over_group(g, local_var, ::sycl::plus<>());
#else
            // no need to sync on sub-group level
            // https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#_reduce
            //   reduce_over_group combines values held directly by the work-items in a group (can be work-group or sub-group).
            local_var = ::sycl::reduce_over_group(sg, local_var, ::sycl::plus<>());
            // local_var = sub_group_reduce_add(local_var);

            //leader: The leader of the sub-group is guaranteed to be the work-item with a local id of 0.
            //if (id_sg_local == 0)
            if (sg.leader())
                variance[id_sg] = local_var;

            // barrier(CLK_LOCAL_MEM_FENCE);
            // https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:what-changed-between
            //      The barrier and mem_fence member functions of the nd_item class have been removed.
            //      The barrier member function has been replaced by the group_barrier() function, which
            //      can be used to block work-items in either work-groups or sub-groups until all work-items
            //      in the group arrive at the barrier. The mem_fence member function has been replaced by the
            //      atomic_fence function, which is more closely aligned with std::atomic_thread_fence and
            //      offers control over memory ordering and scope.
            // 
            // index.barrier(::sycl::access::fence_space::local_space);

            // g.fence_scope is static member of group class (sycl/group.hpp) defined as sycl::memory_scope::work_group
            // the fence specifies all changed made before this barrier should be observed by all items within work-group
            // after this barrier.
            group_barrier(g, g.fence_scope);

            // all sub-groups within current WG independently reduce sum to get overall all_variance;
            // the reducing is redundant since only 8 (WG_SIZE/32) value to sum. but it saves additional barrier
            //
            if (id_sg_local < WG_SIZE / SG_SIZE)
                local_var = variance[id_sg_local];
            else
                local_var = 0;

            // float all_variance = sub_group_reduce_add(local_var);
            float all_variance = ::sycl::reduce_over_group(sg, local_var, ::sycl::plus<>());
#endif
            float scale = ::sycl::rsqrt((all_variance / channel_num) + epsilon);

            for (int i = id_local; i < channel_num; i += WG_SIZE) {
                output[i] = input[i] * scale * weight[i];
            }
        });
    });
    }
}

}