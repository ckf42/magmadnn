/**
 * @file tensor_math.cpp
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-02-21
 *
 * @copyright Copyright (c) 2019
 */

/* AFFINE TRANSFORMATION
 * This example shows you how to create tensors A,x, and b. Then
 * calculate the affine transformation Ax+b and print it out. */

#include <stdio.h>

/* Include magmadnn.h to use the magmadnn libraries */
#include "magmadnn.h"

/* all magmadnn functions/classes are located in the magmadnn namespace */
using namespace magmadnn;

int main(int argc, char **argv) {
    /* must be called at the beginning of every program */
    magmadnn_init();

    /* A: MxN, x: Nx1, and b: Mx1   (Ax+b) : Mx1 */
    const unsigned int M = 5;
    const unsigned int N = 4;

    /* initialize the tensor with three parameters:
        shape: a vector of axis sizes
        filler: a struct dictating how to fill the tensor (this is optional)
        memory type: HOST means these tensors will be stored in RAM for the CPU */
    // Tensor<float> *A_tensor = new Tensor<float> ({M, N}, {CONSTANT, {3}}, HOST);
    // Tensor<float> *x_tensor = new Tensor<float> ({N, 1}, {CONSTANT, {2}}, HOST);
    // Tensor<float> *b_tensor = new Tensor<float> ({M, 1}, {ONE, {}}, HOST);

    /* now we wrap the tensors in variables so that we can use them in the
        compute graph. All compute graph methods are in magmadnn::op.
        op::var takes two arguments:
            name: a string name for the variable (functionality is not dependent on this)
            tensor: the tensor value that the variable points to
        op::var returns an allocated pointer (op::Variable<float> *) */
    // auto A = op::var("A", A_tensor);
    // auto x = op::var("x", x_tensor);
    // auto b = op::var("b", b_tensor);

    /* below is a cleaner way of doing the above. The variables automatically create
     the tensors and handle them for us. */
    auto A = op::var<float>("A", {M, N}, {CONSTANT, {3}}, HOST);
    auto x = op::var<float>("x", {N, 1}, {CONSTANT, {2}}, HOST);
    auto b = op::var<float>("b", {M, 1}, {ONE, {}}, HOST);

    /* create the compute tree. nothing is evaluated.
        returns an operation pointer (op::Operation<float> *) */
    auto aff = op::add(op::matmul(A, x), b);

    /* get the final tensor result by evaluating the compute tree */
    Tensor<float> *final_val = aff->eval();

    /* print out the results */
    printf("Ax+b = {");
    for (int i = 0; i < (int) M; i++) {
        /* final_val is Mx1 in dimension.
           final_val->get({m,n}) returns the m,n element of the matrix */
        printf("%.1f%s", final_val->get({i, 0}), (i != (int) M - 1) ? ", " : "}\n");
    }

    /* free up allocated memory. free the head of the tree, will free
        all of the nodes in the tree. */
    // delete A_tensor;
    // delete x_tensor;
    // delete b_tensor;
    delete aff;

    /* must be called at the end of every magmadnn program */
    magmadnn_finalize();
    return 0;
}