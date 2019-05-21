/**
 * @file add_op.cpp
 * @author Daniel Nichols
 * @version 0.0.1
 * @date 2019-02-20
 * 
 * @copyright Copyright (c) 2019
 */
#include "compute/add/addop.h"

namespace magmadnn {
namespace op {

template <typename T>
AddOp<T>::AddOp(Operation<T>* a, Operation<T>* b, bool copy) : 
	Operation<T>::Operation({a,b}), a(a), b(b), copy(copy) {
	
	assert( a->get_memory_type() == b->get_memory_type() );
	assert( a->get_output_size() == b->get_output_size() );

	this->output_shape = a->get_output_shape();
	this->mem_type = a->get_memory_type();

	/* Go ahead and create copy tensor if we can */
	if (copy)
		this->ret = new Tensor<T> (this->output_shape, this->mem_type);
}

template <typename T>
Tensor<T>* AddOp<T>::eval() {
	a_tensor = a->eval();
	b_tensor = b->eval();

	if (!copy) this->ret = b_tensor;

	internal::geadd_full((T)1, a_tensor, (T)1, b_tensor, this->ret);
	
	return this->ret;
} 

template <typename T>
Operation<T> *AddOp<T>::grad(Operation<T> *consumer, Operation<T> *var, Operation<T> *grad) {
	return grad;
}
template class AddOp<int>;
template class AddOp<float>;
template class AddOp<double>;


template <typename T>
AddOp<T>* add(Operation<T> *a, Operation<T> *b, bool copy) {
    return new AddOp<T> (a, b, copy);
}
template AddOp<int>* add(Operation<int> *a, Operation<int> *b, bool copy);
template AddOp<float>* add(Operation<float> *a, Operation<float> *b, bool copy);
template AddOp<double>* add(Operation<double> *a, Operation<double> *b, bool copy);

} // namespace op
} // namespace magmadnn