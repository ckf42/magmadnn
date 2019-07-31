/**
 * @file relu_internal.h
 * @author Daniel Nichols
 * @version 1.0
 * @date 2019-05-01
 *
 * @copyright Copyright (c) 2019
 */
#pragma once

#include "tensor/tensor.h"

namespace magmadnn {
namespace internal {

/** computes the RELU function element-wise over x
 * @tparam T
 * @param x input tensor
 */
template <typename T>
magmadnn_error_t relu_full(Tensor<T> *x, Tensor<T> *out);

#if defined(_HAS_CUDA_)
template <typename T>
void relu_full_device(Tensor<T> *x, Tensor<T> *out);
#endif

}  // namespace internal
}  // namespace magmadnn