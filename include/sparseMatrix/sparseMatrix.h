#include <vector>

#include "tensor/tensor.h"
#include "utilities_internal.h"

#if defined(_HAS_CUDA_)
#include "cusparse.h"
//#include "magma.h"
#endif

namespace magmadnn {

typedef enum {
    HOST_CSR,
#if defined(_HAS_CUDA_)
    CUSPARSE_CSR
#endif
} sparseMat_format;

template <typename T>
class sparseMatrix {
   protected:
    sparseMat_format format;
    unsigned dim1, dim2;  //  matrix: dim1 * dim2
    void* descripter;
    bool descripter_is_set;

   public:
    sparseMatrix(Tensor<T>* adjMatrixTensorPtr);
    inline sparseMat_format get_format(void) const { return format; }
    inline std::vector<unsigned> get_shape(void) { return std::vector<unsigned>{dim1, dim2}; }
    inline unsigned get_shape(unsigned idx) {
        assert(idx >= 0 && idx < 2);
        return (idx == 0 ? dim1 : dim2);
    }
    virtual ~sparseMatrix(void) = 0;
    virtual memory_t get_memory_type(void) const;
    void* get_desc(void) const {return descripter};
};
template <typename T>
sparseMatrix<T>::sparseMatrix(Tensor<T>* adjMatrixTensorPtr) : descripter(nullptr), descripter_is_set(false) {
    assert(T_IS_MATRIX(adjMatrixTensorPtr));
    dim1 = adjMatrixTensorPtr->get_shape(0);
    dim2 = adjMatrixTensorPtr->get_shape(1);
}
template <typename T>
sparseMatrix<T>::~sparseMatrix(void) {
    if (descripter_is_set) {
        delete descripter;
    }
}

template <typename T>
class baseSparseMat_CSR : public sparseMatrix<T> {
   protected:
    unsigned nnz;  //  number of nonzero elements
    Tensor<T>* valPtr;
    Tensor<int>* rowPtr;
    Tensor<int>* colIdx;

   public:
    baseSparseMat_CSR(const Tensor<T>* adjMatrixTensorPtr, memory_t mem_type);
    virtual ~baseSparseMat_CSR(void) = 0;
    inline unsigned get_nnz(void) { return nnz; }
    inline Tensor<T>* get_val_ptr(void) const { return valPtr; }
    inline Tensor<int>* get_row_ptr(void) const { return rowPtr; }
    inline Tensor<int>* get_col_ptr(void) const { return colIdx; }
    inline memory_t get_memory_type(void) const { return valPtr->get_memory_type(); }
    // ?? get_uncompressed_adjMat(void) const;
};

template <typename T>
class hostSparseMat_CSR : public baseSparseMat_CSR<T> {
   public:
    hostSparseMat_CSR(const Tensor<T>* adjMatrixTensorPtr);
};

#if defined(_HAS_CUDA_)
template <typename T>
class cusparseMat_CSR : public baseSparseMat_CSR<T> {
   private:
    cusparseSpMatDescr_t cusparseDesc;
    bool cusparseDesc_valid = false;

   public:
    cusparseMat_CSR(const Tensor<T>* adjMatrixTensorPtr, memory_t mem_type = DEVICE);
    ~cusparseMat_CSR(void);
};
#endif

}  // namespace magmadnn