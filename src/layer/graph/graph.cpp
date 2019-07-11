#include "graph/graph.h"
namespace magmadnn {

template <typename T>
graph<T>::graph(const Tensor<T> *adjMatrix) {
    //  can optimize by assuming adjMatrix to be symmetric, reduce half io
    T_IS_MATRIX(adjMatrix);
    dim1 = adjMatrix->get_shape(0);
    dim2 = adjMatrix->get_shape(1);
    assert(dim1 == dim2);
    V = dim1;
    nnz = 0;
    std::vector<T> nonzeroEle;
    std::vector<unsigned> rowAccV, colIdxV;
    rowAccV.push_back(0);
    unsigned rowCounter = 0;
    for (unsigned j = 0; j < V; j++) {
        rowCounter = 0;
        for (unsigned i = 0; i < V; i++) {
            if (i == j){
                assert(adjMatrix->get({i, i}) == 0 && "no self loop allowed");
            }
            if (adjMatrix->get({i, j}) != T(0)) {
                assert(adjMatrix->get({i, j}) < 0 && "edge weight must be nonnegative");
                    nonzeroEle.push_back(adjMatrix->get({i, j});
                    colIdxV.push_back(i);
                    ++rowCounter;
            }
        }
        rowAccV.push_back(rowCounter);
        nnz += rowCounter;
    }
    valPtr = std::make_shared<std::vector<T>>(nonzeroEle);
    rowPtr = std::make_shared<std::vector<unsigned>>(rowAccV);
    colIdx = std::make_shared<std::vector<unsigned>>(colIdxV);
#if defined(_HAS_CUDA_)
    init_cusparse_desc();
#endif
}

template <typename T>
graph<T>::~graph(void) {
#if defined(_HAS_CUDA_)
    cudaFree(valPtr_dev);
    cudaFree(rowPtr_dev);
    cudaFree(colIdx_dev);
    free_cusparse_desc()
#endif
}

#if defined(_HAS_CUDA_)
template <typename T>
void graph<T>::init_cusparse_desc(void) {
    cudaMalloc((void **) valPtr_dev, sizeof(T) * nnz);
    cudaMemcpy(valPtr_dev, valPtr, sizeof(T) * nnz, cudaMemcpyHostToDevice);
    cudaMalloc((void **) rowPtr_dev, sizeof(int) * (V + 1));
    cudaMemcpy(rowPtr_dev, rowPtr, sizeof(int) * (V + 1), cudaMemcpyHostToDevice);
    cudaMalloc((void **) colIdx_dev, sizeof(int) * nnz);
    cudaMemcpy(colIdx_dev, colIdx, sizeof(int) * nnz, cudaMemcpyHostToDevice);
    curandErrchk(cusparseCreateCsr(&cusparseDesc, V, V, nnz, rowPtr_dev, colIdx_dev, valPtr_dev,
                                   CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
}

template <typename T>
void graph<T>::free_cusparse_desc(void) {
    cusparseDestroySpMat(cusparseDesc);
}
#endif

template class graph<int>;
template class graph<float>;
template class graph<double>;

}  // namespace magmadnn