#include "layer/gcvconv/gcnconvlayer.h"

namespace magmadnn {
namespace layer {

template <typename T>
void GCNConvLayer<T>::init(void) {
	this->name = "GCNConv";
	T bound = static_cast<T>(sqrt(2.0 / this->input->get_output_shape(1)));
	this->weights_tensor = new Tensor<T>({ this->input->get_output_shape(1), this->hidden_units },
		{ UNIFORM, {-bound, bound} }, this->input->get_memory_type());
	this->weights = op::var("__" + this->name + "_layer_weights", this->weights_tensor);

}
template <typename T>
GCNConvLayer<T>::GCNConvLayer(op::Operation<T>* input, graph<T>* G, unsigned hidden_units):Layer<T>(input->get_output_shape(), input), hidden_units(hidden_units) {
	init();
}
template <typename T>
GCNConvLayer<T>::~GCNConvLayer(void) {
	delete weight_tensor;
}
template <typename T>
std::vector<op::Operation<T>*> GCNConvLayer<T>::get_weights(void) {
	return { this->weights };
}

}  //  namespace layer
}  //  namespace magmadnn
