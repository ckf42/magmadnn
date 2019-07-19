#pragma once

#include <vector>

#include "compute/operation.h"
#include "graph/graph.h"
#include "layer/layer.h"
#include "compute/spgemm/spgemmop.h"

namespace magmadnn {
namespace layer {

template <typename T>
class GCNConvLayer :public Layer<T> {
protected:
	Tensor<T>* weight_tensor;
	op::Operation<T>* weights;
	void init(void);

public:
	GCNConvLayer(op::Operation<T>* input, graph<T>* G, unsigned hidden_units);
	virtual ~GCNConvLayer(void);
	inline op::Operation<T>* get_weight(void) { return weights; }
	std::vector<op::Operation<T>*> get_weights(void);
};


}  //  namespace layer
}  //  namespace magmadnn
