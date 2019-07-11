#include "math/spgematmul.h"
namespace magmadnn {
namespace math {

template <>
void spgematmul(int alpha, graph<int> &A, bool transpo_A, Tensor<int> *B, bool transpo_B, Tensor<int> *C) {}

template <>
void spgematmul(float alpha, graph<float> &A, bool transpo_A, Tensor<float> *B, bool transpo_B, Tensor<float> *C) {
    T_IS_MATRIX(B);
    assert(A.get_order() == B->get_shape(0));
#if defined(_HAS_CUDA_)
    cusparseDnMatDescr_t B_desc;
    cusparseCreateDnMat(B_desc, )
#else
    printf("spgematmul is not implemented on CPU yet.\n");
#endif
}

template <>
void spgematmul(double alpha, graph<double> &A, bool transpo_A, Tensor<double> *B, bool transpo_B, Tensor<double> *C) {}

}  // namespace math
}  // namespace magmadnn