#include "layer/gcvconv/kwgcnlayer.h"

namespace magmadnn {
namespace layer {

template <typename T>
KWGCNLayer<T>::KWGCNLayer(op::Operation<T>* input, graph<T>* struct_graph, unsigned output_channel, bool copy,
                          bool needs_grad)
    : Layer<T>(input->get_output_shape(), input),
      struct_graph(struct_graph),
      output_channel(output_channel),
      copy(copy),
      needs_grad(needs_grad) {
    assert(this->get_input_shape().size() == 3);
    assert(this->struct_graph->get_order() == this->get_input_shape(1));
    this->name = "KWGCNLayer / " + std::to_string(output_channel);
    T bound = static_cast<T>(sqrt(2.0 / this->input->get_output_shape(2)));
    this->weights_tensor = new Tensor<T>({this->input->get_output_shape(2), this->output_channel},
                                         {UNIFORM, {-bound, bound}}, this->input->get_memory_type());
    this->weights = op::var("__" + this->name + "_layer_weights", this->weights_tensor);
    this->transition_matrix =
        this->struct_graph->get_KW_transit_mat(this->struct_graph->get_adj_format(), this->struct_graph->get_data_type());
    bool is_sparse = true;
    switch (this->struct_graph->get_adj_format()) {
        case SPARSEMATRIX_FORMAT_HOST_CSR:
#if defined(_HAS_CUDA_)
    case SPARSEMATRIX_FORMAT_CUSPARSE_CSR:
#endif
        is_sparse = true;
        break;
    case SPARSEMATRIX_FORMAT_HOST_DENSE:
#if defined(_HAS_CUDA_)
    case SPARSEMATRIX_FORMAT_CUSPARSE_DENSE:
#endif
        is_sparse = false;
        break;
    default:
        std::fprintf(stderr, "Input graph format for KWGCN layer is not recongnized.\n");
        break;
    }
    this->output = op::gcnconv(this->transition_matrix, this->input, this->weights, is_sparse);
}
template <typename T>
KWGCNLayer<T>::~KWGCNLayer(void) {
    delete weights_tensor;
}
template <typename T>
std::vector<op::Operation<T>*> KWGCNLayer<T>::get_weights(void) {
    return {this->weights};
}

template <typename T>
KWGCNLayer<T>* kwgcn(op::Operation<T>* input, graph<T>* struct_graph, unsigned output_channel, bool copy, bool needs_grad) {
    return new KWGCNLayer<T>(input, struct_graph, output_channel, copy, needs_grad);
}
template KWGCNLayer<int>* kwgcn(op::Operation<int>* input, graph<int>* struct_graph, unsigned output_channel,
                                  bool, bool needs_grad);
template KWGCNLayer<float>* kwgcn(op::Operation<float>* input, graph<float>* struct_graph, unsigned output_channel,
                                    bool copy, bool needs_grad);
template KWGCNLayer<double>* kwgcn(op::Operation<double>* input, graph<double>* struct_graph, unsigned output_channel,
                                     bool copy, bool needs_grad);

}  //  namespace layer
}  //  namespace magmadnn
