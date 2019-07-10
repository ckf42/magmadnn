
#pragma once

#include <cmath>
#include "tensor/tensor.h"

namespace magmadnn {
namespace internal {

template <typename T>
void log_full(Tensor<T> *x, Tensor<T> *out);

#if defined(_HAS_CUDA_)
template <typename T>
void log_full_device(Tensor<T> *x, Tensor<T> *out);
#endif

template <typename T>
void log_grad(Tensor<T> *x, Tensor<T> *grad, Tensor<T> *out);

#if defined(_HAS_CUDA_)
template <typename T>
void log_grad_device(Tensor<T> *x, Tensor<T> *grad, Tensor<T> *out);
#endif

}  // namespace internal
}  // namespace magmadnn
