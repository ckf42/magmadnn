#pragma once

#include <memory>
#include <string>
#include <vector>

#include "tensor/tensor.h"
#include "utilities_internal.h"

#if defined(_HAS_CUDA_)
#include "cusparse.h"
//#include "magma.h"
#endif

namespace magmadnn {
namespace spMatrix {
/** Abstract base class for all sparse matrix classes
 *  Use sparseMatrix* for generic sparse matrix pointer and cast to corresponding class to access class-specific members
 */
template <typename T>
class sparseMatrix {
   private:
    sparseMatrix(const sparseMatrix& that) = delete;                   //  no copy for abstract class
    sparseMatrix<T>& operator=(const sparseMatrix<T>& that) = delete;  //  no assignment for abstract class
   protected:
    spMatrix_format format;  //  data format for sparse matrix
    unsigned dim0, dim1;     //  matrix: dim0 * dim1
    void* descriptor;        //  descriptor for the sparse matrix, need to cast to corresponding class when needed
    bool descriptor_is_set;  //  whether the descriptor is pointing to a valid object
    memory_t mem_type;       //  where the actural matrix data are stored
   public:
#if defined(DEBUG)
    sparseMatrix(void);
#endif
    sparseMatrix(Tensor<T>* adjMatrixTensorPtr, memory_t mem_type, spMatrix_format format);
    /** returns the data format of the sparse matrix
     */
    inline spMatrix_format get_data_format(void) const { return format; }
    /** returns the shape of the sparse matrix
     */
    inline std::vector<unsigned> get_shape(void) { return std::vector<unsigned>{dim0, dim1}; }
    /** returns the axis size at idx of shape
     *  @param
     */
    inline unsigned get_shape(unsigned idx) {
        assert(idx >= 0 && idx < 2);
        return (idx == 0 ? dim0 : dim1);
    }
    virtual ~sparseMatrix(void) = 0;
    //  returns whether the descriptor is set
    bool has_descriptor(void) const { return descriptor_is_set; }
    //  returns a void pointer to the descriptor
    void* get_desc(void) const {return descriptor};
    //  returns the memory type of the matrix data
    inline memory_t get_memory_type(void) { return mem_type; }
    //  writes the (uncompressed) adjacency matrix into output
    //  output must have the same shape with this object
    virtual void get_uncompressed_mat(Tensor<T>* output) const = 0;

    std::string to_string(void) const {return "spMat(" + dim0 + " x " + dim1 + ")"};
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
    Tensor<T> data;  //  the whole adjacency matix
	//  returns the pointer to the data
	inline Tensor<T>* get_data_ptr(void) const { return &data; }
   public:
#if defined(DEBUG)
    spMatrix_DENSE(void);
#endif
    spMatrix_DENSE(const Tensor<T>* adjMatrixTensorPtr, memory_t mem_type, spMatrix_format format);
    virtual ~spMatrix_DENSE(void) = 0;
    //  writes the (uncompressed) adjacency matrix into output
    //  output must have the same shape with this object
    inline void get_uncompressed_mat(Tensor<T>* output) const {
        T_IS_MATRIX(output);
        assert(output->get_shape[0] == dim0);
        assert(output->get_shape[1] == dim1);
        output->copy_from(&data);
    }
};

//  abstract base class for sparse matrix in CSR format
template <typename T>
class spMatrix_CSR : public sparseMatrix<T> {
   private:
    spMatrix_CSR(const spMatrix_CSR& that) = delete;                   //  no copy for abstract class
    spMatrix_CSR<T>& operator=(const spMatrix_CSR<T>& that) = delete;  //  no assignment for abstract class
   protected:
    unsigned nnz;          //  number of nonzero elements
    Tensor<T> valList;     //  nonzero elements in row order, size 1 * nnz
    Tensor<int> rowCount;  //  number of nonzero elements appear in row[0:idx), zero-based, size 1 * (row + 1)
    Tensor<int> colIdx;    //  column indices of the nonzero elements in val, size 1 * nnz
   public:
#if defined(DEBUG)
    spMatrix_CSR(void);
#endif
    spMatrix_CSR(const Tensor<T>* adjMatrixTensorPtr, memory_t mem_type, spMatrix_format format);
    virtual ~spMatrix_CSR(void) = 0;
    //  returns the number of nonzero elements
    inline unsigned get_nnz(void) { return nnz; }
    //  returns a const pointer to valList
    inline const Tensor<T>* get_val_ptr(void) const { return &valPtr; }
    //  returns a const pointer to rowCount
    inline const Tensor<int>* get_row_ptr(void) const { return &rowPtr; }
    //  returns a const pointer to colIdx
    inline const Tensor<int>* get_col_ptr(void) const { return &colIdx; }
    //  returns the memory type used to store the data
    inline memory_t get_memory_type(void) const { return mem_type; }
    //  writes the (uncompressed) adjacency matrix into output
    //  output must have the same shape with this object
    void get_uncompressed_mat(Tensor<T>* output) const;
};

//  concrete class for sparse matrix in dense format on host memory
//  wrapper for Tensor<T>
template <typename T>
class hostSpMatrix_DENSE : public spMatrix_DENSE<T> {
   public:
    hostSpMatrix_DENSE(const Tensor<T>* adjMatrixTensorPtr);
    hostSpMatrix_DENSE(const hostSpMatrix_DENSE<T>& that);
	hostSpMatrix_DENSE<T>& operator=(const hostSpMatrix_DENSE<T>& that);
    ~hostSpMatrix_DENSE(void);
};

//  concrete class for sparse matrix in CSR format on host memory
template <typename T>
class hostSpMatrix_CSR : public spMatrix_CSR<T> {
   public:
    hostSpMatrix_CSR(const Tensor<T>* adjMatrixTensorPtr);
    hostSpMatrix_CSR(const hostSpMatrix_CSR& that);
    hostSpMatrix_CSR<T>& operator=(const hostSpMatrix_CSR<T>& that);
    ~hostSpMatrix_CSR(void);
};

#if defined(_HAS_CUDA_)

//  todo: check if cusparse can only use mat on host

//  concrete class for sparse matrix in dense format in GPU memory
//  wrapper for cusparseDnMatDescr_t
template <typename T>
class cusparseSpMatrix_DENSE : public spMatrix_DENSE<T> {
   public:
    cusparseSpMatrix_DENSE(const Tensor<T>* adjMatrixTensorPtr, memory_t mem_type = MANAGED);
    cusparseSpMatrix_DENSE(const cusparseSpMatrix_DENSE<T>& that);
    cusparseSpMatrix_DENSE<T>& operator=(const cusparseSpMatrix_DENSE<T>& that);
    ~cusparseSpMatrix_DENSE(void);
};

//  concrete class for sparse matrix in CSR format in GPU memory
//  wrapper for cusparseSpMatDescr_t with format CSR
template <typename T>
class cusparseSpMatrix_CSR : public spMatrix_CSR<T> {
   public:
    cusparseSpMatrix_CSR(const Tensor<T>* adjMatrixTensorPtr, memory_t mem_type = MANAGED);
    cusparseSpMatrix_CSR(const cusparseSpMatrix_CSR<T>& that);
    cusparseSpMatrix_CSR<T>& operator=(const cusparseSpMatrix_CSR<T>& that);
    ~cusparseSpMatrix_CSR(void);
};

#endif

}  // namespace spMatrix
}  //  namespace magmadnn
