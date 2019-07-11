#pragma once

#include <cassert>
#include <memory>
#include <vector>
#include "tensor/tensor.h"
#include "utilities_internal.h"

#if defined(_HAS_CUDA_)
#include "cusparse.h"
#endif

namespace magmadnn {

/* graph object
adjacency matrix stored as sparse matrix in CSR format  */

template <typename T>
class graph {
   private:
    unsigned V;                                //  number of nodes
    unsigned nnz;                              //  num of nonzero element
    std::shared_ptr<std::vector<T>> valPtr;    //  ptr to data array, length nnz
    std::shared_ptr<std::vector<int>> rowPtr;  //  ptr to row array, length V + 1, first one is zero
    std::shared_ptr<std::vector<int>> colIdx;  //  ptr to col index array, length nnz

#if defined(_HAS_CUDA_)
    T *valPtr_dev;
    int *rowPtr_dev;
    int *colIdx_dev;
    cusparseSpMatDescr_t cusparseDesc;
#endif

   public:
    graph(const Tensor<T> *);

    // get the order (number of vertices) of the graph
    inline unsigned get_order(void) { return V; }

    // get the size (number of edges) of the graph
    inline unsigned get_size(void) { return nnz / 2; }
    inline T *get_val_ptr(void) { return valPtr->data(); }
    inline int *get_row_ptr(void) { return rowPtr->data(); }
    inline int *get_col_ptr(void) { return colIdx->data(); }

#if defined(_HAS_CUDA_)
    void init_cusparse_desc(void);
    inline cusparseSpMatDescr_t &get_mat_desc(void) { return cusparseDesc; }
    void free_cusparse_desc(void);
#endif
    ~graph(void);
};

}  // namespace magmadnn