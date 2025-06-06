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


/*
    dst[index[i], :] += src[i, :]  # for DIM == 0 only && 2-dimensions only
*/
__kernel void index_add(
    __global DATATYPE *dst,
    __global const DATATYPE *src,
    __global const int *index,
    __constant const int *dst_sizes,
    __constant const int *src_sizes)
{   
    // Calculate source coordinates
    int src_coords[2] = { get_global_id(0), get_global_id(1) };
    
    // Fetch index value and compute destination coordinates
    int dst_coords[2] = { index[src_coords[0]], src_coords[1] };

    // if (src_idx==3)printf("dst_coords=%d,%d, idx=%d,%d\n", dst_coords[0], dst_coords[1], idx, src_coords[DIM]);
    
    // Bounds checking
    if (dst_coords[0] < 0 || dst_coords[0] >= dst_sizes[0] ||
        dst_coords[1] < 0 || dst_coords[1] >= dst_sizes[1]) {
        return;
    }

    // Calculate destination & source linear index
    int dst_idx = dst_coords[0] * dst_sizes[1] + dst_coords[1];
    int src_idx = src_coords[0] * src_sizes[1] + src_coords[1];
    
    // Perform atomic addition
    DATATYPE val = src[src_idx];
    dst[dst_idx] += val;
    // atomic_add_float(&dst[dst_idx], val);
    // printf("[%d %d] %.2f\n ", src_idx, dst_idx, dst[dst_idx]);
}