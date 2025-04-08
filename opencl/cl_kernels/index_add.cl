#ifdef cl_khr_int64_base_atomics
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#endif

// Atomic add for floats using compare-and-swap
inline void atomic_add_float(volatile __global DATATYPE *addr, DATATYPE val) {
    union {
        uint u32;
        DATATYPE f32;
    } next, expected, current;
    current.f32 = *addr;
    do {
        expected.f32 = current.f32;
        next.f32 = expected.f32 + val;
        current.u32 = atom_cmpxchg((volatile __global uint *)addr,
                                   expected.u32, next.u32);
    } while (current.u32 != expected.u32);
}

// Example: Emulating float16 atomic_add using 32-bit atomics
inline void atomic_add_half(volatile __global ushort *addr, half value) {
    union { ushort u16; half f16; } old_val, new_val, current;
    current.f16 = *((volatile __global half *)addr);
    do {
        old_val.f16 = current.f16;
        new_val.f16 = old_val.f16 + value;
        uint result = atom_cmpxchg((volatile __global uint *)addr, 
                                      (uint)old_val.u16, (uint)new_val.u16);
        current.u16 = (ushort)result;
    } while (current.u16 != old_val.u16);
}

__kernel void index_add(
    __global DATATYPE *dst,
    __global const DATATYPE *src,
    __global const int *index,
    int dim,
    int num_dims,
    __constant const int *dst_sizes,
    __constant const int *dst_strides,
    __constant const int *src_sizes,
    __constant const int *src_strides)
{
    int gid = get_global_id(0);
    
    // Calculate source coordinates
    int src_coords[8]; // Supports up to 8 dimensions
    int remaining = gid;
    for (int d = 0; d < num_dims; ++d) {
        src_coords[d] = (remaining / src_strides[d]) % src_sizes[d];
        remaining = remaining % src_strides[d];
    }

    // if (gid==3)printf("index=%d,%d,%d\n", index[0], index[1], index[2]);
    // if (gid==3)printf("src_coords=%d,%d\n", src_coords[0], src_coords[1]);
    
    // Fetch index value
    // self[index[i], :, :] += alpha * src[i, :, :]  # if dim == 0
    // self[:, index[i], :] += alpha * src[:, i, :]  # if dim == 1
    // self[:, :, index[i]] += alpha * src[:, :, i]  # if dim == 2
    int idx = index[src_coords[dim]];
    
    // Compute destination coordinates
    int dst_coords[8];
    for (int d = 0; d < num_dims; ++d) {
        dst_coords[d] = (d == dim) ? idx : src_coords[d];
    }

    // if (gid==3)printf("dst_coords=%d,%d, idx=%d,%d\n", dst_coords[0], dst_coords[1], idx, src_coords[dim]);
    
    // Bounds checking
    bool out_of_bounds = false;
    for (int d = 0; d < num_dims; ++d) {
        if (dst_coords[d] < 0 || dst_coords[d] >= dst_sizes[d]) {
            out_of_bounds = true;
            break;
        }
    }
    if (out_of_bounds) return;
    
    // Calculate destination linear index
    int dst_gid = 0;
    for (int d = 0; d < num_dims; ++d) {
        dst_gid += dst_coords[d] * dst_strides[d];
    }
    
    // Perform atomic addition
    DATATYPE val = src[gid];
    dst[dst_gid] += val;
    // atomic_add_half(&dst[dst_gid], val);
    // printf("[%d %d] %.2f\n ", gid, dst_gid, dst[dst_gid]);
}