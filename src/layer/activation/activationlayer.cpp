/**
 * @file activationlayer.cpp
 * @author Daniel Nichols
 * @version 1.0
 * @date 2019-02-26
 *
 * @copyright Copyright (c) 2019
 */
#include "layer/activation/activationlayer.h"

namespace magmadnn {
namespace layer {

template <typename T>
ActivationLayer<T>::ActivationLayer(op::Operation<T>* input, activation_t activation_func)
    : Layer<T>::Layer(input->get_output_shape(), input), activation_func(activation_func) {
    init();
}

template <typename T>
ActivationLayer<T>::~ActivationLayer() {}

template <typename T>
std::vector<op::Operation<T>*> ActivationLayer<T>::get_weights() {
    return {};
}

template <typename T>
void ActivationLayer<T>::init() {
    this->name = "Activation / ";

    switch (this->activation_func) {
        case SIGMOID:
            this->output = op::sigmoid(this->input);
            this->name += "Sigmoid";
            break;
        case TANH:
            this->output = op::tanh(this->input);
            this->name += "Tanh";
            break;
        case RELU:
            this->output = op::relu(this->input);
            this->name += "ReLU";
            break;
        case SOFTMAX:
            this->output = op::softmax(this->input);
            this->name += "Softmax";
            break;
        default:
            this->output = op::sigmoid(this->input);
            this->name += "Sigmoid";
            break;
    }
}

template class ActivationLayer<int>;
template class ActivationLayer<float>;
template class ActivationLayer<double>;

template <typename T>
ActivationLayer<T>* activation(op::Operation<T>* input, activation_t activation_func) {
    return new ActivationLayer<T>(input, activation_func);
}
template ActivationLayer<int>* activation(op::Operation<int>*, activation_t);
template ActivationLayer<float>* activation(op::Operation<float>*, activation_t);
template ActivationLayer<double>* activation(op::Operation<double>*, activation_t);

}  // namespace layer
}  // namespace magmadnn