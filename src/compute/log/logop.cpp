
#include "compute/log/logop.h"

namespace magmadnn {
namespace op {

template <typename T>
LogOp<T>::LogOp(Operation<T> *x, bool stable, bool copy, bool needs_grad)
    : Operation<T>::Operation({x}, needs_grad), x(x), stable(stable), copy(copy) {
    this->output_shape = x->get_output_shape();
    this->mem_type = x->get_memory_type();

    if (copy) {
        this->output_tensor = new Tensor<T>(this->output_shape, {NONE, {}}, this->mem_type);
    } else {
        std::fprintf(stderr, "no-copy logop not supported.\n");
    }
}

template <typename T>
Tensor<T> *LogOp<T>::_eval(bool recompute) {
    x_tensor = x->eval(recompute);

    if (!copy) this->output_tensor = x_tensor;

    internal::log_full(x_tensor, this->output_tensor, stable);

    return this->output_tensor;
}

template <typename T>
Tensor<T> *LogOp<T>::_grad(Operation<T> *consumer, Operation<T> *var, Tensor<T> *grad) {
    /* TODO : grad * (1/x) */
    Tensor<T> *out;

    this->x_tensor = x->eval(false); /* don't recompute x if we don't have to */

    out = this->_grad_cache[(uintptr_t) var];
    if (out == NULL) {
        out = new Tensor<T>(this->output_shape, {NONE, {}}, this->mem_type);
        this->_grad_cache[(uintptr_t) var] = out;
    }

    internal::log_grad(x_tensor, grad, out, stable);

    return out;
}

template class LogOp<int>;
template class LogOp<float>;
template class LogOp<double>;

template <typename T>
LogOp<T> *log(Operation<T> *x, bool stable, bool copy, bool needs_grad) {
    return new LogOp<T>(x, stable, copy, needs_grad);
}
template LogOp<int> *log(Operation<int> *x, bool stable, bool copy, bool needs_grad);
template LogOp<float> *log(Operation<float> *x, bool stable, bool copy, bool needs_grad);
template LogOp<double> *log(Operation<double> *x, bool stable, bool copy, bool needs_grad);

}  // namespace op
}  // namespace magmadnn
