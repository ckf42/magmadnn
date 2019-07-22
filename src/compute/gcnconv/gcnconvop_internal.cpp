#include "compute/gcnconv/gcnconvop_internal.h"
#include <iostream>
namespace magmadnn {
namespace internal {

#if defined(_HAS_CUDA_)
#if (CUDART_VERSION >= 100100)
template <typename T>
void set_cusparse_spmm_settings(void* settings, cudaDataType data_type, const T* alpha, bool spMatDoTrans,
                                spMatrix::sparseMatrix<T>* spMat, bool dnMatDoTrans, spMatrix::spMatrix_DENSE<T>* dnMat,
                                const T* beta, spMatrix::spMatrix_DENSE<T>* dnOut, cusparseSpMMAlg_t alg) {
    settings = new math::spgemm_cusparse_settings{alg, nullptr, 0};
    // std::cout << (spMatDoTrans ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE) << std::endl;
    // std::cout << (dnMatDoTrans ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE) << std::endl;
    
    // std::cout << (*alpha) << std::endl;
    // std::cout << (*beta) << std::endl;


    // long row, col, ld, nnz;
    // cudaDataType dtype;
    // cusparseOrder_t otype;
    // float *dev_ptr;
    // int *rowArr, *colArr;
    // float *valArr;
    // cusparseIndexType_t rowOffsetType, colIdxType;
    // cusparseIndexBase_t idxbase;

    // cusparseFormat_t format;
    // cusparseSpMatGetFormat(*AS_TYPE(cusparseSpMatDescr_t*, spMat->get_descriptor()), &format);
    // std::cout << (format) << std::endl;
    
    // cusparseSpMatGetIndexBase(*AS_TYPE(cusparseSpMatDescr_t*, spMat->get_descriptor()), &idxbase);
    
    // cusparseCsrGet(*AS_TYPE(cusparseSpMatDescr_t*, spMat->get_descriptor()), &row, &col, &nnz, (void**)&rowArr, (void**)&colArr, (void**)&valArr, &rowOffsetType, &colIdxType, &idxbase, &dtype);
    // std::cout<<(row)<<std::endl;
    // std::cout<<(col)<<std::endl;
    // std::cout<<(nnz)<<std::endl;
    // int fooPtr[(nnz>row+1)?nnz:row+1];
    // float fooFPtr[nnz];
    // cudaMemcpy(fooPtr, rowArr, (row + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < row + 1; ++i) {
    //     std::cout << fooPtr[i] << " ";
    // }
    // std::cout << std::endl;

    // cudaMemcpy(fooPtr, colArr, nnz * sizeof(int), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < nnz; ++i) {
    //     std::cout << fooPtr[i] << " ";
    // }
    // std::cout << std::endl;
    // cudaMemcpy(fooFPtr, valArr, nnz * sizeof(float), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < nnz; ++i) {
    //     std::cout << fooFPtr[i] << " ";
    // }
    // std::cout << std::endl;
    // cudaFree(dev_ptr);

    // cusparseSpMatGetValues(*AS_TYPE(cusparseSpMatDescr_t*, dnOut->get_descriptor()), (void**)&dev_ptr);

    // cusparseDnMatGet(*AS_TYPE(cusparseDnMatDescr_t*, dnMat->get_descriptor()), &row, &col, &ld, (void**) &dev_ptr,
    //                  &dtype, &otype);
    // std::cout << (row) << std::endl;
    // std::cout << (col) << std::endl;
    // std::cout << (ld) << std::endl;
    // float arr[100];
    // cudaMemcpy(arr, dev_ptr, 100 * sizeof(float), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < 100; ++i) {
    //     std::cout << arr[i] << " ";
    // }
    // std::cout << std::endl;


    // cusparseDnMatGet(*AS_TYPE(cusparseDnMatDescr_t*, dnOut->get_descriptor()), &row, &col, &ld, (void**) &dev_ptr,
    //                  &dtype, &otype);
    // std::cout << (row) << std::endl;
    // std::cout << (col) << std::endl;
    // std::cout << (ld) << std::endl;
    // cudaMemcpy(arr, dev_ptr, 100 * sizeof(float), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < 100; ++i) {
    //     std::cout << arr[i] << " ";
    // }
    // std::cout << std::endl;
    // std::cout << (data_type) << std::endl;
    // std::cout<<(alg)<<std::endl;

    size_t sizeVar;

    cusparseSpMatDescr_t spDesc;
    spMatrix::cusparseSpMatrix_CSR<T>* castedSpMat = AS_TYPE(spMatrix::cusparseSpMatrix_CSR<T>*, spMat);
    cusparseCreateCsr(&spDesc, 10, 10, 10, castedSpMat->get_row_ptr()->get_ptr(), castedSpMat->get_col_ptr()->get_ptr(),
                      castedSpMat->get_val_ptr()->get_ptr(), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                      CUDA_R_32F);
    cusparseDnMatDescr_t dnDesc1;
    cusparseCreateDnMat(&dnDesc1, 10, 10, 10, dnMat->get_data_ptr()->get_ptr(), CUDA_R_32F, CUSPARSE_ORDER_COL);
    cusparseDnMatDescr_t dnDesc2;
    cusparseCreateDnMat(&dnDesc2, 10, 10, 10, dnOut->get_data_ptr()->get_ptr(), CUDA_R_32F, CUSPARSE_ORDER_COL);

    cusparseErrchk(
        cusparseSpMM_bufferSize(::magmadnn::internal::MAGMADNN_SETTINGS->cusparse_handle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE, (void*)alpha,
                                spDesc,
                                dnDesc1, (void*)beta,
                                dnDesc2, CUDA_R_32F, CUSPARSE_CSRMM_ALG1,
                                &sizeVar));
    printf("gar5a\n");
    cudaErrchk(cudaMalloc((void**) &AS_TYPE(math::spgemm_cusparse_settings*, settings)->workspace,
                          sizeVar));
}
template void set_cusparse_spmm_settings<int>(void* settings, cudaDataType data_type, const int* alpha,
                                              bool spMatDoTrans, spMatrix::sparseMatrix<int>* spMat, bool dnMatDoTrans,
                                              spMatrix::spMatrix_DENSE<int>* dnMat, const int* beta,
                                              spMatrix::spMatrix_DENSE<int>* dnOut, cusparseSpMMAlg_t alg);
template void set_cusparse_spmm_settings<float>(void* settings, cudaDataType data_type, const float* alpha,
                                                bool spMatDoTrans, spMatrix::sparseMatrix<float>* spMat,
                                                bool dnMatDoTrans, spMatrix::spMatrix_DENSE<float>* dnMat,
                                                const float* beta, spMatrix::spMatrix_DENSE<float>* dnOut,
                                                cusparseSpMMAlg_t alg);
template void set_cusparse_spmm_settings<double>(void* settings, cudaDataType data_type, const double* alpha,
                                                 bool spMatDoTrans, spMatrix::sparseMatrix<double>* spMat,
                                                 bool dnMatDoTrans, spMatrix::spMatrix_DENSE<double>* dnMat,
                                                 const double* beta, spMatrix::spMatrix_DENSE<double>* dnOut,
                                                 cusparseSpMMAlg_t alg);
#elif (CUDART_VERSION >= 10010)

#endif
#endif

}  //  namespace internaldnMat
}  //  namespace magmadnndnMat
