#pragma once

#include <vector>

#include "compute/operation.h"
#include "graph/graph.h"
#include "layer/layer.h"
#include "sparseMatrix/sparseMatrix.h"
#include "compute/tensor_operations.h"
#include "compute/spgemm/spgemmop.h"

namespace magmadnn {
namespace layer {

template <typename T>
class GCNConvLayer :public Layer<T> {
protected:
	Tensor<T>* weight_tensor;
	graph<T>* struct_graph;
	op::Operation<T>* weights;
	spMatrix::sparseMatrix<T>* transition_matrix;
	void init(void);
	bool copy;
	bool needs_grad;

public:
	GCNConvLayer(op::Operation<T>* input, graph<T>* struct_graph, unsigned hidden_units, bool copy = true, bool needs_grad = true);
	virtual ~GCNConvLayer(void);
	inline op::Operation<T>* get_weight(void) { return weights; }
	std::vector<op::Operation<T>*> get_weights(void);
};

template <typename T>
GCNConvLayer<T>* gcnconv(op::Operation<T>* input, graph<T>* struct_graph, unsigned hidden_units, bool copy = true, bool needs_grad = true);

}  //  namespace layer
}  //  namespace magmadnn
