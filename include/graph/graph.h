#pragma once

#include <cassert>
#include <vector>
#include "sparseMatrix/sparseMatrix.h"
#include "tensor/tensor.h"
#include "utilities_internal.h"

#if defined(_HAS_CUDA_)
#include "cusparse.h"
#endif

namespace magmadnn {

template <typename T>
class graph {
   private:
    unsigned V;                            //  number of nodes in the graph
    unsigned E;                            //  number of edges in the graph
    spMatrix::sparseMatrix<T>* adjMatrix;  //  adjacency matrix of the graph, stored as a sparse matrix
    static unsigned getE(const Tensor<T>* adjMatrixTensorPtr);

   public:
    graph(const Tensor<T>* adjMatrixTensorPtr, spMatrix_format format, memory_t mem_type = HOST,
          bool checkInput = false);
    ~graph(void);
    //  returns the number of nodes in the graph
    inline unsigned get_order(void) { return V; }
    //  returns the number of edges in the graph
    inline unsigned get_size(void) { return E; }
    //  writes the laplacian matrix into output
    //  output must have be a V * V Tensor
    void get_laplancian(Tensor<T>* output) const;
    //  writes the normalized laplacian matrix into output
    //  output must have be a V * V Tensor
    void get_norm_laplancian(Tensor<T>* output) const;
};

}  // namespace magmadnn
