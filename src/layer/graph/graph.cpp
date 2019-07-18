#include "graph/graph.h"
namespace magmadnn {

template <typename T>
unsigned graph<T>::getE(const Tensor<T>* adjMatrixTensorPtr) {
    unsigned counter = 0;
    T edge_weight, opposite_edge_weight;
    for (unsigned i = 0; i < V - 1; ++i) {
        assert(adjMatrixTensorPtr->get({i, i}) == 0 && "No self loop allowed");
        for (unsigned j = i + 1; j < V; ++j) {
            edge_weight = adjMatrixTensorPtr->get({i, j});
            assert(edge_weight >= 0 && "Edge weight must all be nonnegative");
            opposite_edge_weight = adjMatrixTensorPtr->get({j, i});
            assert(edge_weight == opposite_edge_weight && "Graph must be undirected");
            if (edge_weight != T(0)) {
                counter++;
            }
        }
    }
    return counter;
}

template <typename T>
graph<T>::graph(const Tensor<T>* adjMatrixTensorPtr, spMatrix_format format, memory_t mem_type, bool checkInput)
    : V(0), E(0), adjMatrix(nullptr) {
    //  can optimize by assuming adjMatrix to be symmetric, reduce half io
    //  how?
    assert(T_IS_MATRIX(adjMatrixTensorPtr));
    assert(adjMatrixTensorPtr->get_shape(0) == adjMatrixTensorPtr->get_shape(1));
    V = adjMatrixTensorPtr->get_shape(0);
    E = 0;
    if (checkInput) {
        E = graph<T>::getE(adjMatrixTensorPtr);
    }
    switch (format) {
        case SPARSEMATRIX_FORMAT_HOST_DENSE:
            adjMatrix = new spMatrix::hostSpMatrix_DENSE<T>(adjMatrixTensorPtr, true);
            if (!checkInput) {
                E = graph<T>::getE(adjMatrixTensorPtr);
            }
            break;
        case SPARSEMATRIX_FORMAT_HOST_CSR:
            adjMatrix = new spMatrix::hostSpMatrix_CSR<T>(adjMatrixTensorPtr);
            if (!checkInput) {
                E = AS_TYPE(spMatrix::hostSpMatrix_CSR<T>*, adjMatrix)->get_nnz() / 2;
            }
            break;
#if defined(_HAS_CUDA_)
        case SPARSEMATRIX_FORMAT_CUSPARSE_DENSE:
            adjMatrix = new spMatrix::cusparseSpMatrix_DENSE<T>(adjMatrixTensorPtr, mem_type, true);
            if (!checkInput) {
                E = graph<T>::getE(adjMatrixTensorPtr);
            }
            break;
        case SPARSEMATRIX_FORMAT_CUSPARSE_CSR:
            adjMatrix = new spMatrix::cusparseSpMatrix_CSR<T>(adjMatrixTensorPtr, mem_type);
            if (!checkInput) {
                E = AS_TYPE(spMatrix::cusparseSpMatrix_CSR<T>*, adjMatrix)->get_nnz() / 2;
            }
            break;
#endif
        default:
            std::fprintf(stderr, "Graph constructor for requested sparse matrix format is not implemented.\n");
            break;
    }
}

template <typename T>
graph<T>::~graph(void) {
    delete adjMatrix;
}

template <typename T>
spMatrix::sparseMatrix<T>* graph<T>::get_GCNConv_mat(spMatrix_format return_format, memory_t return_mem_type) const {
    std::vector<T> Dtilde(V, (T) 0);
    for (unsigned idx = 0; idx < V; ++idx) {
        Dtilde[idx] = 1 + this->adjMatrix->rowSum(idx);
    }

}

template class graph<int>;
template class graph<float>;
template class graph<double>;

}  // namespace magmadnn
