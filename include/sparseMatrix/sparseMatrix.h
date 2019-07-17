#pragma once

#include <memory>
#include <string>
#include <vector>

#include "compute/transpose/transpose_internal.h"  //  for converting order for cusparseSpMatrix_DENSE
#include "math/scalar_tensor_product.h"
#include "tensor/tensor.h"
#include "utilities_internal.h"

#if defined(_HAS_CUDA_)
#include "cusparse.h"
//#include "magma.h"
#endif

namespace magmadnn {
namespace spMatrix {
//  Abstract base class for all sparse matrix classes
//  Use sparseMatrix* for generic sparse matrix pointer and cast to corresponding class to access class-specific members
//  todo: rewrite in templates to deal with void*
template <typename T>
class sparseMatrix {
   private:
    sparseMatrix(const sparseMatrix& that) = delete;                   //  no copy for abstract class
    sparseMatrix<T>& operator=(const sparseMatrix<T>& that) = delete;  //  no assignment for abstract class
   protected:
    spMatrix_format _format;  //  data format for sparse matrix
    unsigned _dim0, _dim1;    //  matrix: _dim0 * _dim1
    void* _descriptor;        //  descriptor for the sparse matrix, need to cast to corresponding class when needed
    bool _descriptor_is_set;  //  whether the _descriptor is pointing to a valid object
    memory_t _mem_type;       //  where the actural matrix data are stored
   public:
#if defined(DEBUG)
    sparseMatrix(void);
#endif
    sparseMatrix(const Tensor<T>* adjMatrixTensorPtr, memory_t mem_type, spMatrix_format format);
    /** returns the data format of the sparse matrix
     */
    inline spMatrix_format get_data_format(void) const { return _format; }
    /** returns the shape of the sparse matrix
     */
    inline std::vector<unsigned> get_shape(void) { return std::vector<unsigned>{_dim0, _dim1}; }
    /** returns the axis size at idx of shape
     *  @param
     */
    inline unsigned get_shape(unsigned idx) {
        assert(idx >= 0 && idx < 2);
        return (idx == 0 ? _dim0 : _dim1);
    }
    virtual ~sparseMatrix(void) = 0;
    //  returns whether the descriptor is set
    bool has_descriptor(void) const { return _descriptor_is_set; }
    //  returns a void pointer to the descriptor
    void* get_descriptor(void) const { return _descriptor; }
    //  returns the memory type of the matrix data
    inline memory_t get_memory_type(void) { return _mem_type; }
    //  writes the (uncompressed) matrix into output and multiplies it with alpha
    //  output must have the same shape with this object
    virtual void get_uncompressed_mat(Tensor<T>* output, T alpha) const = 0;

    std::string to_string(void) const { return "spMat(" + std::to_string(_dim0) + " x " + std::to_string(_dim1) + ")"; }
};

//  abstract base class for sparse matrix in dense format
//  data are stored as a tensor
//  basically wrapper for class Tensor<T>
template <typename T>
class spMatrix_DENSE : public sparseMatrix<T> {
   private:
    spMatrix_DENSE(const spMatrix_DENSE& that) = delete;                   //  no copy for abstract class
    spMatrix_DENSE<T>& operator=(const spMatrix_DENSE<T>& that) = delete;  //  no assignment for abstract class
   protected:
    Tensor<T>* _data;  //  a copy of the whole adjacency matix
    //  returns the pointer to the data Tensor
    inline Tensor<T>* get_data_ptr(void) const { return _data; }

   public:
#if defined(DEBUG)
    spMatrix_DENSE(void);
#endif
    //  allocate a copy
    spMatrix_DENSE(const Tensor<T>* adjMatrixTensorPtr, memory_t mem_type, spMatrix_format format, bool copy);
    virtual ~spMatrix_DENSE(void) = 0;
    //  writes the (uncompressed) matrix into output and multiplies it with alpha
    //  output must have the same shape with this object
    inline void get_uncompressed_mat(Tensor<T>* output, T alpha) const {
        assert(T_IS_MATRIX(output));
        assert(output->get_shape(0) == this->_dim0);
        assert(output->get_shape(1) == this->_dim1);
        output->copy_from(*_data);
        if (alpha != (T) 1) {
            math::scalar_tensor_product(alpha, output, output);
        }
    }
    //  copies the content from input and writes it to _data, must have same shape
    inline virtual void set_mat(const Tensor<T>* input) {
        assert(T_IS_MATRIX(input));
        assert(input->get_shape(0) == this->_dim0);
        assert(input->get_shape(1) == this->_dim1);
        _data->copy_from(*input);
    }
};

//  abstract base class for sparse matrix in CSR format
template <typename T>
class spMatrix_CSR : public sparseMatrix<T> {
   private:
    spMatrix_CSR(const spMatrix_CSR& that) = delete;                   //  no copy for abstract class
    spMatrix_CSR<T>& operator=(const spMatrix_CSR<T>& that) = delete;  //  no assignment for abstract class
   protected:
    unsigned _nnz;           //  number of nonzero elements
    Tensor<T>* _valList;     //  nonzero elements in row order, size 1 * _nnz
    Tensor<int>* _rowCount;  //  number of nonzero elements appear in row[0:idx), zero-based, size 1 * (row + 1)
    Tensor<int>* _colIdx;    //  column indices of the nonzero elements in val, size 1 * _nnz
   public:
#if defined(DEBUG)
    spMatrix_CSR(void);
#endif
    spMatrix_CSR(const Tensor<T>* adjMatrixTensorPtr, memory_t mem_type, spMatrix_format format);
    virtual ~spMatrix_CSR(void) = 0;
    //  returns the number of nonzero elements
    inline unsigned get_nnz(void) { return _nnz; }
    //  returns a const pointer to valList
    inline const Tensor<T>* get_val_ptr(void) const { return _valList; }
    //  returns a const pointer to rowCount
    inline const Tensor<int>* get_row_ptr(void) const { return _rowCount; }
    //  returns a const pointer to colIdx
    inline const Tensor<int>* get_col_ptr(void) const { return _colIdx; }
    //  returns the memory type used to store the data
    inline memory_t get_memory_type(void) const { return this->_mem_type; }
    //  writes the (uncompressed) adjacency matrix into output
    //  output must have the same shape with this object and assumed to be all zero
    void get_uncompressed_mat(Tensor<T>* output, T alpha) const;
};

//  concrete class for sparse matrix in dense format on host memory
//  wrapper for Tensor<T>
template <typename T>
class hostSpMatrix_DENSE : public spMatrix_DENSE<T> {
   private:
    hostSpMatrix_DENSE(const hostSpMatrix_DENSE<T>& that) = delete;
    hostSpMatrix_DENSE<T>& operator=(const hostSpMatrix_DENSE<T>& that) = delete;

   public:
    hostSpMatrix_DENSE(const Tensor<T>* adjMatrixTensorPtr, bool copy = false);
    ~hostSpMatrix_DENSE(void);
};

//  concrete class for sparse matrix in CSR format on host memory
template <typename T>
class hostSpMatrix_CSR : public spMatrix_CSR<T> {
   private:
    hostSpMatrix_CSR(const hostSpMatrix_CSR& that) = delete;
    hostSpMatrix_CSR<T>& operator=(const hostSpMatrix_CSR<T>& that) = delete;

   public:
    hostSpMatrix_CSR(const Tensor<T>* adjMatrixTensorPtr);
    ~hostSpMatrix_CSR(void);
};

#if defined(_HAS_CUDA_)

//  concrete class for sparse matrix in cusparse dense format in GPU memory
//  wrapper for cusparseDnMatDescr_t object
//  since cusparse does not support row-major dense matrix while Tensor stores data in row-major order, internal _data are stored as column-major
//  hense whenever a object is created from a Tensor or whenever get_uncompressed_data() is called, extra time will be used to convert order 
//  todo: find a method to reduce the computation time spent on converting order 
//  todo: wait for cusparse support on row-major order
template <typename T>
class cusparseSpMatrix_DENSE : public spMatrix_DENSE<T> {
   private:
    cusparseSpMatrix_DENSE(const cusparseSpMatrix_DENSE<T>& that) = delete;
    cusparseSpMatrix_DENSE<T>& operator=(const cusparseSpMatrix_DENSE<T>& that) = delete;

   public:
    cusparseSpMatrix_DENSE(const Tensor<T>* adjMatrixTensorPtr, memory_t mem_type = MANAGED, bool copy = false);
    ~cusparseSpMatrix_DENSE(void);
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
};

//  concrete class for sparse matrix in cusparse CSR format in GPU memory
//  wrapper for cusparseSpMatDescr_t with format CSR
template <typename T>
class cusparseSpMatrix_CSR : public spMatrix_CSR<T> {
   private:
    cusparseSpMatrix_CSR(const cusparseSpMatrix_CSR<T>& that) = delete;
    cusparseSpMatrix_CSR<T>& operator=(const cusparseSpMatrix_CSR<T>& that) = delete;

   public:
    cusparseSpMatrix_CSR(const Tensor<T>* adjMatrixTensorPtr, memory_t mem_type = MANAGED);
    ~cusparseSpMatrix_CSR(void);
};

#endif

}  // namespace spMatrix
}  //  namespace magmadnn
