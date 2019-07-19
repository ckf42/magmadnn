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
	this->transition_matrix = this->struct_graph->get_GCNConv_mat(\, \);  //  todo: what format/mem_type?
	this->output = op::linearforward(op::spgemm(this->transition_matrix, this->input), this->weights);
}
template <typename T>
GCNConvLayer<T>::GCNConvLayer(op::Operation<T>* input, graph<T>* struct_graph, unsigned hidden_units, bool copy, bool needs_grad):Layer<T>(input->get_output_shape(), input), struct_graph(struct_graph), hidden_units(hidden_units), copy(copy), needs_grad(needs_grad) {
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

template <typename T>
GCNConvLayer<T>* gcnconv(op::Operation<T>* input, graph<T>* struct_graph, unsigned hidden_units, bool copy = true, bool needs_grad = true) {
	return new GCNConvLayer<T>(input, struct_graph, hidden_units, copy, needs_grad);
}
template GCNConvLayer<int>* gcnconv(op::Operation<int>* input, graph<int>* struct_graph, unsigned hidden_units, bool copy = true, bool needs_grad = true);
template GCNConvLayer<float>* gcnconv(op::Operation<float>* input, graph<float>* struct_graph, unsigned hidden_units, bool copy = true, bool needs_grad = true);
template GCNConvLayer<double>* gcnconv(op::Operation<double>* input, graph<double>* struct_graph, unsigned hidden_units, bool copy = true, bool needs_grad = true);

}  //  namespace layer
}  //  namespace magmadnn
