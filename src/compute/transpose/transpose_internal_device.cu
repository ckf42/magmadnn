
#include "compute/transpose/transpose_internal.h"

#define TILE_DIM 32
#define BLOCK_ROWS 8

namespace magmadnn {
namespace internal {

/* Transpose code modified from "Efficient Matrix Transpose Cuda" by Mark Harris (Nvidia)
    at https://devblogs.nvidia.com/efficient-matrix-transpose-cuda-cc/ */

template <typename T>
__global__ void kernel_transpose_full_device(T *in, T *out, unsigned int x_rows, unsigned int x_cols) {
    __shared__ T tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

#pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < x_cols && (y + j) < x_rows) {
            tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * x_cols + x];
        }
    }

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

#pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < x_rows && (y + j) < x_cols) {
            out[(y + j) * x_rows + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

template <typename T>
void transpose_full_device(Tensor<T> *x, Tensor<T> *out) {
    unsigned int x_rows = x->get_shape(0);
    unsigned int x_cols = x->get_shape(1);

    dim3 dimGrid(ceil(((float) x_rows) / TILE_DIM), ceil(((float) x_cols) / TILE_DIM), 1);
    dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);

    kernel_transpose_full_device<<<dimGrid, dimBlock>>>(x->get_ptr(), out->get_ptr(), x_rows, x_cols);
}
template void transpose_full_device(Tensor<int> *x, Tensor<int> *out);
template void transpose_full_device(Tensor<float> *x, Tensor<float> *out);
template void transpose_full_device(Tensor<double> *x, Tensor<double> *out);

}  // namespace internal
}  // namespace magmadnn
