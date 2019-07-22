#pragma once

#include <memory>
#include <string>
#include <vector>

#include "compute/transpose/transpose_internal.h"  //  for transpose in spMatrix_DENSE
#include "math/scalar_tensor_product.h"
#include "tensor/tensor.h"
#include "utilities_internal.h"

#if defined(_HAS_CUDA_)

#include "cusparse.h"
//#include "magma.h"  //  todo: add support
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
    //  construct with given (dense) matrix
    sparseMatrix(const Tensor<T>* spMatrixTensorPtr, memory_t mem_type, spMatrix_format format);
    //  constructor only to define the dim
    sparseMatrix(unsigned dim0, unsigned dim1, memory_t mem_type, spMatrix_format format);
    // returns the data format of the sparse matrix
    inline spMatrix_format get_data_format(void) const { return _format; }
    // returns the shape of the sparse matrix
    inline std::vector<unsigned> get_shape(void) { return std::vector<unsigned>{_dim0, _dim1}; }
    // returns the axis size at idx of shapedigonal
    inline unsigned get_shape(unsigned idx) {
        assert(idx >= 0 && idx < 2);
        return (idx == 0 ? _dim0 : _dim1);
    }
    virtual ~sparseMatrix(void) = 0;
    //  returns whether the descriptor is set
    bool has_descriptor(void) const { return _descriptor_is_set; }
    //  returns a void pointer to the descriptor
    virtual void* get_descriptor(void) const { return _descriptor; }
    //  returns the memory type of the matrix data
    inline memory_t get_memory_type(void) { return _mem_type; }
    //  writes the (uncompressed) matrix into output and multiplies it with alpha
    //  output must have the same shape with this object
    virtual void get_uncompressed_mat(Tensor<T>* output, T alpha) const = 0;
    //  returns the string the matrix that describes the matrix shape
    inline std::string to_string(void) const {
        return "spMat(" + std::to_string(_dim0) + " x " + std::to_string(_dim1) + ")";
    }
    //  returns the value at location (i, j)
    virtual T get(unsigned i, unsigned j) const = 0;
    //  returns a vector storing all the values at row rowIdx
    virtual std::vector<T> getRowSlice(unsigned rowIdx) const = 0;
    //  changes the value at location (i, j) to val
    virtual void set(unsigned i, unsigned j, T val) = 0;
    //  returns the sum of elements in row rowIdx
    virtual T rowSum(unsigned rowIdx) const;
};

//  abstract base class for sparse matrix in dense format
//  data are stored as a tensor
//  basically wrapper for class Tensor<T> (possibly with additional descriptor)
template <typename T>
class spMatrix_DENSE : public sparseMatrix<T> {
   private:
    spMatrix_DENSE(const spMatrix_DENSE& that) = delete;                   //  no copy for abstract class
    spMatrix_DENSE<T>& operator=(const spMatrix_DENSE<T>& that) = delete;  //  no assignment for abstract class

   protected:
    Tensor<T>* _data;  //  a copy of the whole adjacency matix
                       //  returns the pointer to the data Tensor

   public:
#if defined(DEBUG)
    spMatrix_DENSE(void);
#endif
    //  allocate a copy
    spMatrix_DENSE(const Tensor<T>* spMatrixTensorPtr, memory_t mem_type, spMatrix_format format, bool copy);
    //  allocate a diagonal dense matrix
    spMatrix_DENSE(const std::vector<T>& diag, memory_t mem_type, spMatrix_format format);
    //  allocate shape only
    spMatrix_DENSE(unsigned dim0, unsigned dim1, memory_t mem_type, spMatrix_format format);
    virtual ~spMatrix_DENSE(void) = 0;
    //  returns a pointer to the data ptr
    inline Tensor<T>* get_data_ptr(void) const { return _data; }
    //  writes the matrix into output and multiplies it with alpha
    //  output must have the same shape with this object
    inline void get_uncompressed_mat(Tensor<T>* output, T alpha = (T) 1) const {
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
    //  returns the value at location (i, j)
    inline virtual T get(unsigned i, unsigned j) const { return _data->get({i, j}); }
    //  returns a vector storing all the values at row rowIdx
    virtual std::vector<T> getRowSlice(unsigned rowIdx) const;
    //  changes the value at location (i, j) to val
    inline virtual void set(unsigned i, unsigned j, T val) { _data->set({i, j}, val); }
    //  returns the sum of elements in row rowIdx
    virtual T rowSum(unsigned rowIdx) const;
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
    spMatrix_CSR(const Tensor<T>* spMatrixTensorPtr, memory_t mem_type, spMatrix_format format);
    spMatrix_CSR(const std::vector<T>& diag, memory_t mem_type, spMatrix_format format);
    spMatrix_CSR(unsigned dim0, unsigned dim1, const std::vector<T>& valList, const std::vector<int>& rowAccum,
                 const std::vector<int>& colIdx, memory_t mem_type, spMatrix_format format);
    virtual ~spMatrix_CSR(void) = 0;
    //  returns the number of nonzero elements
    inline unsigned get_nnz(void) { return _nnz; }
    //  returns a const pointer to valList
    inline Tensor<T>* get_val_ptr(void) const { return _valList; }
    //  returns a const pointer to rowCount
    inline Tensor<int>* get_row_ptr(void) const { return _rowCount; }
    //  returns a const pointer to colIdx
    inline Tensor<int>* get_col_ptr(void) const { return _colIdx; }
    //  writes the (uncompressed) adjacency matrix into output
    //  output must have the same shape with this object and is assumed to be all zero
    void get_uncompressed_mat(Tensor<T>* output, T alpha = (T) 1) const;
    //  return the value at location (i, j)
    T get(unsigned i, unsigned j) const;
    //  returns a vector storing all the values at row rowIdx
    std::vector<T> getRowSlice(unsigned rowIdx) const;
    //  sets the value at location (i, j) to val
    //  if nnz changes, need to reallocate _valList, _colIdx (and possibly descriptors)
    //  currently can only change nonzero value to nonzero value
    //  todo: change implementation to allow adding/removing nonzero element
    void set(unsigned i, unsigned j, T val);
    //  returns the sum of elements in row rowIdx
    T rowSum(unsigned rowIdx) const;
};

//  concrete class for sparse matrix in dense format on host memory
//  wrapper for Tensor<T>
template <typename T>
class hostSpMatrix_DENSE : public spMatrix_DENSE<T> {
   private:
    hostSpMatrix_DENSE(const hostSpMatrix_DENSE<T>& that) = delete;
    hostSpMatrix_DENSE<T>& operator=(const hostSpMatrix_DENSE<T>& that) = delete;

   public:
    hostSpMatrix_DENSE(const Tensor<T>* spMatrixTensorPtr, bool copy = false);
    hostSpMatrix_DENSE(const std::vector<T>& diag);
    hostSpMatrix_DENSE(unsigned dim0, unsigned dim1);
    ~hostSpMatrix_DENSE(void);
};

//  concrete class for sparse matrix in CSR cusparseSpMatrix_DENSEformat on host memory
template <typename T>
class hostSpMatrix_CSR : public spMatrix_CSR<T> {
   private:
    hostSpMatrix_CSR(const hostSpMatrix_CSR& that) = delete;
    hostSpMatrix_CSR<T>& operator=(const hostSpMatrix_CSR<T>& that) = delete;

   public:
    hostSpMatrix_CSR(const Tensor<T>* spMatrixTensorPtr);
    hostSpMatrix_CSR(const std::vector<T>& diag);
    hostSpMatrix_CSR(unsigned dim0, unsigned dim1, const std::vector<T>& valList, const std::vector<int>& rowAccum,
                     const std::vector<int>& colIdx);
    ~hostSpMatrix_CSR(void);
};

#if defined(_HAS_CUDA_)
#if (CUDART_VERSION >= 100100)
//  Concrete class for sparse matrix in cusparse dense format in GPU memory
//  Wrapper for cusparseDnMatDescr_t object in CUDA version 10.1+.
//  The interanal _data is stored in row-major (as a normal Tensor) but cusparse currently only takes column-major dense
//  matrices
//  Hence the descriptor stores the dimensions as transposed and an extra transpose would be needed when calling
//  cusparse routines
//  todo: wait for cusparse support on row-major order
template <typename T>
class cusparseSpMatrix_DENSE_10010 : public spMatrix_DENSE<T> {
   private:
    cusparseSpMatrix_DENSE_10010(const cusparseSpMatrix_DENSE_10010<T>& that) = delete;
    cusparseSpMatrix_DENSE_10010<T>& operator=(const cusparseSpMatrix_DENSE_10010<T>& that) = delete;
    cudaDataType_t cuda_data_type;
    void createDesc(void);

   public:
    cusparseSpMatrix_DENSE_10010(const Tensor<T>* spMatrixTensorPtr, memory_t mem_type = MANAGED, bool copy = false);
    cusparseSpMatrix_DENSE_10010(const std::vector<T>& diag, memory_t mem_type = MANAGED);
    cusparseSpMatrix_DENSE_10010(unsigned dim0, unsigned dim1, memory_t mem_type = MANAGED);
    ~cusparseSpMatrix_DENSE_10010(void);
    inline cudaDataType_t get_data_type(void) { return this->cuda_data_type; }
};

//  concrete class for sparse matrix in cusparse CSR format in GPU memory
//  wrapper for cusparseSpMatDescr_t with format CSR
template <typename T>
class cusparseSpMatrix_CSR_10010 : public spMatrix_CSR<T> {
   private:
    cusparseSpMatrix_CSR_10010(const cusparseSpMatrix_CSR_10010<T>& that) = delete;
    cusparseSpMatrix_CSR_10010<T>& operator=(const cusparseSpMatrix_CSR_10010<T>& that) = delete;
    cudaDataType_t cuda_data_type;
    void createDesc(void);

   public:
    cusparseSpMatrix_CSR_10010(const Tensor<T>* spMatrixTensorPtr, memory_t mem_type = MANAGED);
    cusparseSpMatrix_CSR_10010(const std::vector<T>& diag, memory_t mem_type = MANAGED);
    cusparseSpMatrix_CSR_10010(unsigned dim0, unsigned dim1, const std::vector<T>& valList,
                               const std::vector<int>& rowAccum, const std::vector<int>& colIdx,
                               memory_t mem_type = MANAGED);
    ~cusparseSpMatrix_CSR_10010(void);
    inline cudaDataType_t get_data_type(void) { return this->cuda_data_type; }
};
template <typename T>
using cusparseSpMatrix_DENSE = cusparseSpMatrix_DENSE_10010<T>;
template <typename T>
using cusparseSpMatrix_CSR = cusparseSpMatrix_CSR_10010<T>;
#elif (CUDART_VERSION >= 10010)
//  Concrete class for sparse matrix in cusparse dense format in GPU memory
//  For dense matrix used in CUDA version below 10.1
//  wrapper for tensor on device
//  Since cusparse takes column-major dense matrix but Tensor stores in row-major format, internal _data are stored as
//  column-major. Hense whenever a object is uesd in cusparse, an extra transpose and a reshape will be needed
template <typename T>
class cusparseSpMatrix_DENSE_LEGACY : public spMatrix_DENSE<T> {
   private:
    cusparseSpMatrix_DENSE_LEGACY(const cusparseSpMatrix_DENSE_LEGACY<T>& that) = delete;
    cusparseSpMatrix_DENSE_LEGACY<T>& operator=(const cusparseSpMatrix_DENSE_LEGACY<T>& that) = delete;
    cudaDataType_t cuda_data_type;

   public:
    cusparseSpMatrix_DENSE_LEGACY(const Tensor<T>* spMatrixTenorPtr, memory_t mem_type = MANAGED, bool copy = false);
    cusparseSpMatrix_DENSE_LEGACY(const std::vector<T>& diag, memory_t mem_type = MANAGED);
    cusparseSpMatrix_DENSE_LEGACY(unsigned dim0, unsigned dim1, memory_t mem_type = MANAGED);
    ~cusparseSpMatrix_DENSE_LEGACY(void);
    inline cudaDataType_t get_data_type(void) { return this->cuda_data_type; }
};

//  concrete class for sparse matrix in cusparse CSR format
template <typename T>
class cusparseSpMatrix_CSR_LEGACY : public spMatrix_CSR<T> {
    private:
    cusparseSpMatrix_CSR_LEGACY(const cusparseSpMatrix_CSR_LEGACY<T>& that) = delete;
    cusparseSpMatrix_CSR_LEGACY<T> operator=(const cusparseSpMatrix_CSR_LEGACY<T>& that) = delete;
    cudaDataType_t cuda_data_type;
    void createDesc(void);

   public:
    cusparseSpMatrix_CSR_LEGACY(const Tensor<T>* spMatrixTensorPtr, memory_t mem_type = MANAGED);
    cusparseSpMatrix_CSR_LEGACY(const std::vector<T>& diag, memory_t mem_type = MANAGED);
    cusparseSpMatrix_CSR_LEGACY(unsigned dim0, unsigned dim1, const std::vector<T>& valList,
                               const std::vector<int>& rowAccum, const std::vector<int>& colIdx,
                               memory_t mem_type = MANAGED);
    ~cusparseSpMatrix_CSR_LEGACY(void);
    inline cudaDataType_t get_data_type(void) { return this->cuda_data_type; }
};
template <typename T>
using cusparseSpMatrix_DENSE = cusparseSpMatrix_DENSE_LEGACY<T>;
template <typename T>
using cusparseSpMatrix_CSR = cusparseSpMatrix_CSR_LEGACY<T>;
#endif

#endif

//  returns a new sparseMatrix pointer that stores spMatrixTensorPtr in given format
template <typename T>
sparseMatrix<T>* get_spMat(const Tensor<T>* spMatrixTensorPtr, spMatrix_format format, memory_t mem_type = HOST);
//  returns a new sparseMatrix pointer with CSR format data valList, rowAccum, colIdx in given format
template <typename T>
sparseMatrix<T>* get_spMat(unsigned dim0, unsigned dim1, const std::vector<T>& valList,
                           const std::vector<int>& rowAccum, const std::vector<int>& colIdx, spMatrix_format format,
                           memory_t mem_type = HOST);

}  // namespace spMatrix
}  //  namespace magmadnn
