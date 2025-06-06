/*******************************************************************************
* Copyright (c) 2022-2023 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/// @file
/// C++ API

#pragma once

#include "common/core/common.hpp"

namespace gpu::xetla {

/// @addtogroup xetla_core_base_types
/// @{

/// @brief xetla bf16 data type.
/// The difference between bf16 and fp32 is:
///
/// fp32: 0_00000000_00000000000000000000000
///
/// bf16: 0_00000000_0000000
/// @note
/// The member function in bf16 class is only used in host side.
/// For device side, we will automatically convert it to its native type.
/// @see native_type_t
///
using bf16 = sycl::ext::oneapi::bfloat16;

/// @brief xetla fp16 data type.
using fp16 = sycl::half;

/// @brief xetla tf32 data type.
/// The difference between tf32 and fp32 is:
///
/// fp32: 0_00000000_00000000000000000000000
///
/// tf32: 0_00000000_0000000000
/// @note
/// The member function in tf32 class is only used in host side.
/// For device side, we will automatically convert it to its native type.
/// @see native_type_t
///
using tf32 = sycl::ext::intel::experimental::esimd::tfloat32;

template <typename T, typename = void>
struct is_host_callable : std::false_type {};
template <typename T>
struct is_host_callable<T, std::enable_if_t<T::host_callable == true>>
    : std::true_type {};

/// @brief Used to check if the type is xetla internal data type
/// @tparam T is the data type
template <typename T>
struct is_internal_type {
    static constexpr bool value = std::is_same<remove_const_t<T>, bf16>::value
            || std::is_same<remove_const_t<T>, tf32>::value;
};

/// @brief Used to check if the type is floating_point.
/// @tparam T is the data type
template <typename T>
struct is_floating_point {
    static constexpr bool value = std::is_same<remove_const_t<T>, bf16>::value
            || std::is_same<remove_const_t<T>, fp16>::value
            || std::is_same<remove_const_t<T>, tf32>::value
            || std::is_same<remove_const_t<T>, float>::value
            || std::is_same<remove_const_t<T>, double>::value;
};

/// @brief Used to check if the type is floating_point.
/// @tparam T is the data type
template <typename T>
struct is_integral {
    static constexpr bool value = std::is_same<remove_const_t<T>, int8_t>::value
            || std::is_same<remove_const_t<T>, uint8_t>::value
            || std::is_same<remove_const_t<T>, int16_t>::value
            || std::is_same<remove_const_t<T>, uint16_t>::value
            || std::is_same<remove_const_t<T>, int32_t>::value
            || std::is_same<remove_const_t<T>, uint32_t>::value
            || std::is_same<remove_const_t<T>, int64_t>::value
            || std::is_same<remove_const_t<T>, uint64_t>::value;
};

/// @brief Set the native data type of T
/// @tparam T is the data type
template <typename T>
struct native_type {
    using type = T;
};

/// @brief Return the native data type of T
template <typename T>
using native_type_t = typename native_type<T>::type;

/// @brief Get the unit representation of type T
template <typename T>
struct uint_type {
    static constexpr bool is_uint8 = sizeof(T) == 1;
    static constexpr bool is_uint16 = sizeof(T) == 2;
    static constexpr bool is_uint32 = sizeof(T) == 4;
    static constexpr bool is_uint64 = sizeof(T) == 8;
    using type = typename std::conditional<is_uint8, uint8_t,
            typename std::conditional<is_uint16, uint16_t,
                    typename std::conditional<is_uint32, uint32_t,
                            typename std::conditional<is_uint64, uint64_t,
                                    void>::type>::type>::type>::type;
};

/// @brief Get the unit representation based on Size
template <int Size>
struct get_uint_type {
    static constexpr bool is_uint8 = Size == 1;
    static constexpr bool is_uint16 = Size == 2;
    static constexpr bool is_uint32 = Size == 4;
    static constexpr bool is_uint64 = Size == 8;
    using type = typename std::conditional<is_uint8, uint8_t,
            typename std::conditional<is_uint16, uint16_t,
                    typename std::conditional<is_uint32, uint32_t,
                            typename std::conditional<is_uint64, uint64_t,
                                    void>::type>::type>::type>::type;
};
/// @brief Return the uint representation of type T
template <typename T>
using uint_type_t = typename uint_type<T>::type;

/// @brief Return the uint representation based on Size
template <int Size>
using get_uint_type_t = typename get_uint_type<Size>::type;

/// @brief wrapper for xetla_vector.
/// Alias to ESIMD `__ESIMD_NS::simd`;
/// @tparam Ty data type in xetla_vector.
/// @tparam N  data length in xetla_vector.
///
template <typename Ty, uint32_t N>
using xetla_vector = __ESIMD_NS::simd<native_type_t<Ty>, N>;

///
/// @brief Description of nd tensor descriptor for load and store.
/// Structure is defined in [here](https://gfxspecs.intel.com/Predator/Home/Index/63986).
///
using xetla_tdescriptor = xetla_vector<uint32_t, 16>;

/// @brief Alias to xetla_vector<uint32_t, 16> reference.
#define xetla_tdescriptor_ref xetla_vector<uint32_t, 16> /* __REF__*/

/// @brief wrapper for xetla_mask.
/// Alias to ESIMD `__ESIMD_NS::simd_mask`.
/// @tparam N  data length in xetla_mask.
///
template <uint32_t N>
using xetla_mask = __ESIMD_NS::simd_mask<N>;

/// @brief wrapper for xetla_mask_int.
/// Alias to ESIMD `__ESIMD_NS::simd_mask`.
/// @tparam N data length in xetla_mask_int.
///
template <uint32_t N>
using xetla_mask_int = __ESIMD_NS::simd_mask<N>;

/// @brief Workaround for ESIMD reference usage.
/// Alias to `auto` if go with ESIMD path.
/// @see gpu::xetla::core::xetla_matrix_ref gpu::xetla::core::xetla_vector_ref
#define __REF__ auto

/// @brief Workaround for ESIMD vector(1D) ref type.
/// Use C++20 [concept](https://en.cppreference.com/w/cpp/language/constraints) to constrains the scope of auto.
/// @note Need to be used together with `__REF__`, i.e. `"xetla_vector_ref __REF__"` is the full declaration of xetla vector reference.
/// @tparam Ta first tparam is reserved for auto.
/// @tparam Ty data type in xetla_vector_ref.
/// @tparam N  data length in xetla_vector_ref.
///
// template <typename Ta, typename Ty, uint32_t N>
// concept xetla_vector_ref = __ESIMD_NS::detail::is_simd_view_type_v<
//         Ta> && std::is_same_v<typename Ta::element_type, native_type_t<Ty>> &&(N
//         == __ESIMD_NS::shape_type<Ta>::type::Size_x
//                 * __ESIMD_NS::shape_type<Ta>::type::Size_y);
#define xetla_vector_ref xetla_vector

/// @brief Workaround for ESIMD matrix(2D) ref type.
/// Use C++20 [concept](https://en.cppreference.com/w/cpp/language/constraints) to constrains the scope of auto.
/// @note Need to be used together with `__REF__`, i.e. `"xetla_matrix_ref __REF__"` is the full declaration of xetla matrix reference.
/// @tparam Ta first tparam is reserved for auto.
/// @tparam Ty data type in xetla_matrix_ref.
/// @tparam N1 row num in xetla_matrix_ref.
/// @tparam N2 col num in xetla_matrix_ref.
///
// template <typename Ta, typename Ty, uint32_t N1, uint32_t N2>
// concept xetla_matrix_ref
//         = __ESIMD_NS::detail::is_simd_view_type_v<Ta> && std::is_same_v<
//                   typename Ta::element_type, native_type_t<Ty>> &&(N1
//                   == __ESIMD_NS::shape_type<Ta>::type::Size_y)
//         && (N2 == __ESIMD_NS::shape_type<Ta>::type::Size_x);
#define xetla_matrix_ref xetla_vector

/// @} xetla_core_base_types

} // namespace gpu::xetla

#if (__LIBSYCL_MAJOR_VERSION > 7) \
        || ((__LIBSYCL_MAJOR_VERSION == 7) && (__LIBSYCL_MINOR_VERSION >= 1))

namespace sycl::detail {
template <typename T>
struct is_device_copyable_impl<T,
        std::enable_if_t<gpu::xetla::is_host_callable<T>::value>>
    : std::true_type {};
} // namespace sycl::detail

#else

namespace sycl {
template <typename T>
struct is_device_copyable<T,
        std::enable_if_t<gpu::xetla::is_host_callable<T>::value>>
    : std::true_type {};
} // namespace sycl
#endif
