#pragma once
#include "sparseMatrix/sparseMatrix.h"

namespace magmadnn {
namespace spMatrix {

#if defined(_HAS_CUDA_)
#if defined(USE_CUSPARSE_NEW_API)
//  Concrete class for sparse matrix in cusparse dense format in GPU memory
//  Wrapper for cusparseDnMatDescr_t object in CUDA version 10.1+.
//  Since cusparse does not support row-major dense matrix while Tensor stores data in row-major order, internal _data
//  are stored as column-major. Hense whenever a object is created from a Tensor or whenever get_uncompressed_data() is
//  called, extra time will be used to convert order
//  todo: find a method to reduce the computation time spent on
//  converting order
//  todo: wait for cusparse support on row-major order
template <typename T>
class cusparseSpMatrix_DENSE_10010 : public spMatrix_DENSE<T> {
   private:
    cusparseSpMatrix_DENSE_10010(const cusparseSpMatrix_DENSE_10010<T>& that) = delete;
    cusparseSpMatrix_DENSE_10010<T>& operator=(const cusparseSpMatrix_DENSE_10010<T>& that) = delete;
    void createDesc(cudaDataType_t cuda_data_type);

   public:
    cusparseSpMatrix_DENSE_10010(const Tensor<T>* spMatrixTensorPtr, memory_t mem_type = MANAGED, bool copy = false);
    cusparseSpMatrix_DENSE_10010(const std::vector<T>& diag, memory_t mem_type = MANAGED);
    cusparseSpMatrix_DENSE_10010(unsigned dim0, unsigned dim1, memory_t mem_type = MANAGED);
    ~cusparseSpMatrix_DENSE_10010(void);
    //  need to specialize due the the difference in order
    inline void get_uncompressed_mat(Tensor<T>* output, T alpha = (T) 1) const {
        assert(T_IS_MATRIX(output));
        assert(output->get_shape(0) == this->_dim0);
        assert(output->get_shape(1) == this->_dim1);
        output->copy_from(*this->_data);
        internal::transpose_full_device<T>(output, output);
        if (alpha != (T) 1) {
            math::scalar_tensor_product(alpha, output, output);
        }
    }
    //  same as above
    inline T get(unsigned i, unsigned j) const { return spMatrix_DENSE<T>::get(j, i); }
	//  same as above
    inline void set(unsigned i, unsigned j, T val) { spMatrix_DENSE<T>::set(j, i, val); }
};

//  concrete class for sparse matrix in cusparse CSR format in GPU memory
//  wrapper for cusparseSpMatDescr_t with format CSR
template <typename T>
class cusparseSpMatrix_CSR_10010 : public spMatrix_CSR<T> {
   private:
    cusparseSpMatrix_CSR_10010(const cusparseSpMatrix_CSR_10010<T>& that) = delete;
    cusparseSpMatrix_CSR_10010<T>& operator=(const cusparseSpMatrix_CSR_10010<T>& that) = delete;
    void createDesc(cudaDataType_t cuda_data_type);

   public:
    cusparseSpMatrix_CSR_10010(const Tensor<T>* spMatrixTensorPtr, memory_t mem_type = MANAGED);
    cusparseSpMatrix_CSR_10010(const std::vector<T>& diag, memory_t mem_type = MANAGED);
    cusparseSpMatrix_CSR_10010(unsigned dim0, unsigned dim1, const std::vector<T>& valList,
                               const std::vector<int>& rowAccum, const std::vector<int>& colIdx,
                               memory_t mem_type = MANAGED);
    ~cusparseSpMatrix_CSR_10010(void);
};
#endif
#endif
}  //  namespace spMatrix
}  //  namespace magmadnn
