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

/// @addtogroup xetla_core_barrier
/// @{

/// @brief Initialize the number of named barrier index for a kernel.
/// Available only on PVC. Only need to initialize once at the beginning.
///
/// @tparam NbarCount  - number of named barriers.
template <uint8_t NbarCount>
__XETLA_API void xetla_nbarrier_init() {
    if constexpr (NbarCount != 0) {
#if (__LIBSYCL_MAJOR_VERSION >= 8)
        __ESIMD_NS::named_barrier_init<NbarCount>();
#else
        __ESIMD_ENS::named_barrier_init<NbarCount>();
#endif
    }
}

/// @brief Perform signal operation for the given named barrier id.
/// Available only on PVC.
///
/// @param barrier_id  [in] is the named barrier id.
///
/// @param producer_consumer_mode  [in] is 2-bit flag to indicate if it's
/// producer-consumer mode(0x0) or producer mode (0x1) or consumer mode (0x2).
/// User must ensure the input value is set correctly and higher order bits are cleared.
///
/// @param num_producers  [in] is the number of producers.
///
/// @param num_consumers  [in] is the number of consumers.
__XETLA_API void named_barrier_signal(uint8_t barrier_id,
        uint8_t producer_consumer_mode, uint32_t num_producers,
        uint32_t num_consumers) {
#if (__LIBSYCL_MAJOR_VERSION >= 8)
    __ESIMD_NS::named_barrier_signal(
#else
    __ESIMD_ENS::named_barrier_signal(
#endif
            barrier_id, producer_consumer_mode, num_producers, num_consumers);
}

/// @brief Wait on a named barrier.
/// Available only on PVC
///
/// @param barrier_id  [in] is the named barrier id.
/// It’s value cannot exceed the total count of initialized named barriers.
__XETLA_API void named_barrier_wait(uint8_t barrier_id) {
#if (__LIBSYCL_MAJOR_VERSION >= 8)
    __ESIMD_NS::named_barrier_wait(barrier_id);
#else
    __ESIMD_ENS::named_barrier_wait(barrier_id);
#endif
}

/// @} xetla_core_barrier

} // namespace gpu::xetla
