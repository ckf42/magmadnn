#### Adding New Operations 
---------------------------
The skeleton of the operation superclass can be found at [include/compute/operation.h](/include/compute/operation.h). There are several things that must be implemented in order to have a working operation. Below the process is layed out for the example operation `sub`.

1. Create a new folder in `include/compute` and `src/compute` named `sub`.
2. Inside the `include/compute/sub` folder create `subop.h` and in `src/compute/sub` create `subop.cpp`. 
3. Define the `SubOp` class, which extends `Operation`, in the header file. Note that the class must be in the namespace `magmadnn::op`. The method `foo` should also be defined here, which returns a new operation class `SubOp`.
4. Implement `SubOp` in `subop.cpp`. The new operation must implement the constructor, _eval, _grad, and to_string methods. `_eval` should return the evaluated tensor up to this point in the tree. (Note, you are allowed to create helper files within the folder `sub/`, but their methods must live within the namespace `magmadnn::internal`). `_grad` is given the consumer operation, variable that gradient is w.r.t., and the grad w.r.t. the output. It should return a tensor pointer, where the tensor contains the gradient computation. Implement the function `sub` in `subop.cpp`. `sub` must work with and be compiled for `int`, `float`, and `double`. `sub` is also expected to work for all memory types. See [constructor](#constructor), [eval](#eval), [grad](#grad), [to_string](#to_string), and [func](#func) for more information on how to implement these.
5. Add `#include "sub/subop.h"` to `include/compute/tensor_operations.h`. This allows the rest of the library to see the new operation.
6. _Optional:_ Add a tester file to the `testing/` folder.

See [include/compute/add/](/include/compute/add) and [src/compute/add/](/src/compute/add) for an actual example.

Operators _can_ implement copy and no-copy options, determining whether to return a newly allocated tensor or write over one of the parameters. However, this is not required.

### constructor
The constructor must call the parent constructor of `Operation<T>` that takes a vector of operations. This sets the children of the operation and is used for memory releasing. `output_shape` and `mem_type` should also be set within the constructor. This allows shape checking and pre-allocation of tensors when the tree is created.

The constructor should also do any preprocessing that is possible, to remove computational burden from the performance critical `eval` function. 

An example of a constructor might look like,

```c++
/* x is the only child here, so pass that to the parent class constructor. */
template <typename T>
TanhOp<T>::TanhOp(Operation<T> *x, bool needs_grad) : Operation<T>::Operation({x}, needs_grad), x(x) {
    
    /* set the output shape and memory type */
    this->output_shape = x->get_output_shape();
    this->mem_type = x->get_memory_type();

    /* create return tensor here to avoid memory allocation at tree execution */
    this->output_tensor = new Tensor<T> (this->output_shape, this->mem_type);
}
```

### eval
The eval method is simply responsible for the evaluation of the operation. It should return a tensor pointer with the same shape and memory type as defined by the operation. An example eval function might look like,

```c++
template <typename T>
Tensor<T>* MatmulOp<T>::_eval(bool recompute) {
    /* evaluate the child nodes */
    a_tensor = a->eval(recompute);    // MxK
    b_tensor = b->eval(recompute);    // KxN
    c_tensor = c->eval(recompute);    // MxN
    
    /* use an external utility function to multiply the matrices */
    /* or use the preferred math::matmul(.) function */
    internal::gemm_full(alpha, a_tensor, b_tensor, beta, this->output_tensor);

    return this->output_tensor;
} 
```


### grad
Grad functions should write into the gradient tensor and return it. Every operation has a `std::map<Operation<T>*, Tensor<T>*> _grad_cache`, which can help store these tensors keyed on input variables. For instance, the `_grad` implementation of a log operation:
```c++
template <typename T>
Tensor<T> *LogOp<T>::_grad(Operation<T> *consumer, Operation<T> *var, Tensor<T> *grad) {
    /* TODO : grad * (1/x) */
    Tensor<T> *out;

    this->x_tensor = x->eval(false);    /* don't recompute x if we don't have to */
    
    out = this->_grad_cache[(uintptr_t)var];
    if (out == NULL) {
        out = new Tensor<T> (this->output_shape, {NONE,{}}, this->mem_type);
        this->_grad_cache[(uintptr_t)var] = out;
    }

    internal::log_grad(x_tensor, grad, out);
    
    return out;
}
```

### to_string
The `to_string` method is fairly simple to implement. It defines a form to print out the operation, using the `to_string` return values of the child operations. For instance, the operation `add(a,b)`'s implementation might look like 

```c++
template <typename T>
std::string AddOp<T>::to_string() {
    return "(" + a->to_string() + " + " + b->to_string() + ")";
    /* OR something like this */
    return "ADD(" + a->to_string() + ", " + b->to_string() + ")";
}
```

### func
Every operation should be paired with a function that returns a new pointer to that operation. This allows for cleaner math expressions for the library user (i.e. `auto x = op::add(op::matmul(A, x),c)`). An example of this function might look like

```c++
template <typename T>
TanhOp<T>* tanh(Operation<T> *x, bool use_gpu) {
    return new TanhOp<T> (x, use_gpu);
}
```