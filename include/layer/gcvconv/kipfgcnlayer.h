#pragma once

#include <vector>

#include "compute/operation.h"
#include "graph/graph.h"
#include "layer/layer.h"
#include "sparseMatrix/sparseMatrix.h"
#include "compute/tensor_operations.h"
#include "compute/gcnconv/gcnconvop.h"

namespace magmadnn {
namespace layer {

//  layer for doing GCNConv as defined by Kipf and Welling in "Semi-Supervised Classification with Graph Convolutional Networks" (arXiv:1609.02907)
//  input: operation pointer, output shape batch_size * n_vertex * n_channelIn
//  struct_graph: graph object, of order n_vertex, in sparse format
//  output_channel: number of output channels, n_channelOut
//  output shape: batch_size * n_vertex * n_channelOut
template <typename T>
class KipfGcnLayer :public Layer<T> {
protected:
	Tensor<T>* weight_tensor;
	graph<T>* struct_graph;
	op::Operation<T>* weights;
	spMatrix::sparseMatrix<T>* transition_matrix;
	void init(void);
	unsigned output_channel;
	bool copy;
	bool needs_grad;

public:
	KipfGcnLayer(op::Operation<T>* input, graph<T>* struct_graph, unsigned output_channel, bool copy = true, bool needs_grad = true);
	virtual ~KipfGcnLayer(void);
	inline op::Operation<T>* get_weight(void) { return weights; }
	std::vector<op::Operation<T>*> get_weights(void);
};

template <typename T>
KipfGcnLayer<T>* kipfgcn(op::Operation<T>* input, graph<T>* struct_graph, unsigned output_channel, bool copy = true, bool needs_grad = true);

}  //  namespace layer
}  //  namespace magmadnn
