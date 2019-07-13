#include "graph/graph.h"
namespace magmadnn {

template <typename T>
graph<T>::graph(const Tensor<T> *adjMatrixTensorPtr, sparseMat_format format, memory_t mem_type) {
    //  can optimize by assuming adjMatrix to be symmetric, reduce half io
    //  how?
    assert(T_IS_MATRIX(adjMatrixTensorPtr));
    assert(adjMatrixTensorPtr->get_shape(0) == adjMatrixTensorPtr->get_shape(1));
    V = adjMatrixTensorPtr->get_shape(0);

    switch (format) {
        case HOST_CSR:
            adjMatrix = new hostSparseMat_CSR<T>(adjMatrixTensorPtr);
            E = adjMatrix->get_nnz();
            break;
#if defined(_HAS_CUDA_)
        case CUSPARSE_CSR:
            adjMatrix = new cusparseMat_CSR<T>(adjMatrixTensorPtr, mem_type);
            E = adjMatrix->get_nnz();
            break;
#endif
        default:
            std::fprintf(stderr, "Constructor for requested format is not implemented.\n");
            break;
    }
}

template <typename T>
graph<T>::~graph(void) {
    delete adjMatrix;
}

#if defined(_HAS_CUDA_)

#endif

template class graph<int>;
template class graph<float>;
template class graph<double>;

}  // namespace magmadnn