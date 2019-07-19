#include "layer/gcvconv/kipfgcnlayer.h"

namespace magmadnn {
namespace layer {

template <typename T>
void KipfGcnLayer<T>::init(void) {
    this->name = "GCNConv";
    T bound = static_cast<T>(sqrt(2.0 / this->input->get_output_shape(1)));
    this->weights_tensor = new Tensor<T>({this->input->get_output_shape(1), this->output_channel},
                                         {UNIFORM, {-bound, bound}}, this->input->get_memory_type());
    this->weights = op::var("__" + this->name + "_layer_weights", this->weights_tensor);
    this->transition_matrix =
        this->struct_graph->get_GCNConv_mat(this->struct_graph->get_adj_format(), this->struct_graph->get_data_type());
    this->outout = op::gcnconv(this->transition_matrix, this->input, this->weights);
}
template <typename T>
KipfGcnLayer<T>::KipfGcnLayer(op::Operation<T>* input, graph<T>* struct_graph, unsigned output_channel, bool copy,
                              bool needs_grad)
    : Layer<T>(input->get_output_shape(), input),
      struct_graph(struct_graph),
      output_channel(output_channel),
      copy(copy),
      needs_grad(needs_grad) {
    assert(this->get_input_shape().size == 3);
    assert(this->struct_graph->get_order() == this->get_input_shape(1));
    init();
}
template <typename T>
KipfGcnLayer<T>::~KipfGcnLayer(void) {
    delete weight_tensor;
}
template <typename T>
std::vector<op::Operation<T>*> KipfGcnLayer<T>::get_weights(void) {
    return {this->weights};
}

template <typename T>
KipfGcnLayer<T>* kipfgcn(op::Operation<T>* input, graph<T>* struct_graph, unsigned output_channel, bool copy = true,
                         bool needs_grad = true) {
    return new KipfGcnLayer<T>(input, struct_graph, output_channel, copy, needs_grad);
}
template KipfGcnLayer<int>* kipfgcn(op::Operation<int>* input, graph<int>* struct_graph, unsigned output_channel,
                                    bool copy = true, bool needs_grad = true);
template KipfGcnLayer<float>* kipfgcn(op::Operation<float>* input, graph<float>* struct_graph, unsigned output_channel,
                                      bool copy = true, bool needs_grad = true);
template KipfGcnLayer<double>* kipfgcn(op::Operation<double>* input, graph<double>* struct_graph,
                                       unsigned output_channel, bool copy = true, bool needs_grad = true);

}  //  namespace layer
}  //  namespace magmadnn
