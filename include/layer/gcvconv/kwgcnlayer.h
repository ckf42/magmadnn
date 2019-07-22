#pragma once

#include <vector>

#include "compute/gcnconv/gcnconvop.h"
#include "compute/operation.h"
#include "compute/tensor_operations.h"
#include "graph/graph.h"
#include "layer/layer.h"
#include "sparseMatrix/sparseMatrix.h"

namespace magmadnn {
namespace layer {

//  layer for doing GCNConv as defined by Kipf and Welling in "Semi-Supervised Classification with Graph Convolutional
//  Networks" (arXiv:1609.02907)
//  input:           operation pointer, output shape batch_size * n_vertex * n_channelIn
//  struct_graph:    graph object, of order n_vertex, in sparse format
//  output_channel:  number of output channels, n_channelOut
//  output shape:    batch_size * n_vertex * n_channelOut
template <typename T>
class KWGCNLayer : public Layer<T> {
   protected:
    Tensor<T>* weights_tensor;
    graph<T>* struct_graph;
    op::Operation<T>* weights;
    spMatrix::sparseMatrix<T>* transition_matrix;
    unsigned output_channel;
    bool copy;
    bool needs_grad;

   public:
    KWGCNLayer(op::Operation<T>* input, graph<T>* struct_graph, unsigned output_channel, bool copy = true,
               bool needs_grad = true);
    virtual ~KWGCNLayer(void);
    inline op::Operation<T>* get_weight(void) { return weights; }
    std::vector<op::Operation<T>*> get_weights(void);
};

template <typename T>
KWGCNLayer<T>* kipfgcn(op::Operation<T>* input, graph<T>* struct_graph, unsigned output_channel, bool copy = true,
                       bool needs_grad = true);

}  //  namespace layer
}  //  namespace magmadnn
