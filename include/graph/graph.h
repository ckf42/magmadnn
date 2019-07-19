#pragma once

#include <cassert>
#include <vector>
#include <cmath>
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
    //  static helper method to count E
    static unsigned getE(const Tensor<T>* adjMatrixTensorPtr);

   public:
    graph(const Tensor<T>* adjMatrixTensorPtr, spMatrix_format format, memory_t mem_type = HOST,
          bool checkInput = false);
    //  todo: other constructor
    ~graph(void);
    //  returns the number of nodes in the graph
    inline unsigned get_order(void) { return V; }
    //  returns the number of edges in the graph
    inline unsigned get_size(void) { return E; }
    //  returns the pointer to the (sparse) adjMatrix
    inline spMatrix::sparseMatrix<T>* get_adj_ptr(void) const { return *adjMatrix; }
    //  returns the format of the adjacency matrix
    inline spMatrix_format get_adj_format(void) const { return adjMatrix->get_data_format(); }
    //  returns the memory type of the adjacency matrix
    inline memory_t get_data_type(void) const { return adjMatrix->get_memory_type(); }
    //  returns the approximated and renormalized transition matrix used in arXiv:1609.02907 by Kipf and Welling, K = \tilde{D}^{-1/2} * \tilde{A} * \tilde{D}^{-1/2}, where \tilde{A} = A + I_{V} is the adjacency matrix of the arugmented graph with a self-loop of unit weight inserted to each vertex, \tilde{D}_{ii} = \sum_{k}\tilde{A}_{ik} is the (digonal) degree matrix \tilde{A}
    //  K_{ij} = \tilde{D}_{ii}^{-1/2} * \tilde{D}_{jj}^{-1/2} * ( A_{ij} + \delta_{ij} )
    //  workaround before appropriate math routines of sparse matrix addition/modification are implemented
    //  todo: rewrite with proper math routines
    spMatrix::sparseMatrix<T>* get_GCNConv_mat(spMatrix_format return_format, memory_t return_mem_type = HOST) const;
};

}  // namespace magmadnn
