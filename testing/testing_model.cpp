/**
 * @file testing_model.cpp
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-05-30
 *
 * @copyright Copyright (c) 2019
 */

#include <cstdio>
#include <vector>
#include "magmadnn.h"
#include "utilities.h"

using namespace magmadnn;

void test_model_MLP(memory_t mem, unsigned int size);

int main(int argc, char **argv) {
    magmadnn_init();

    test_for_all_mem_types(test_model_MLP, 50);

    magmadnn_finalize();
    return 0;
}

void test_model_MLP(memory_t mem, unsigned int size) {
    unsigned int n_features = 6;
    unsigned int n_classes = 10;
    unsigned int n_samples = 10;
    unsigned int batch_size = 2;
    model::metric_t metrics;

    printf("testing %s MLP...  ", get_memory_type_name(mem));

    Tensor<float> *x = new Tensor<float>({n_samples, n_features}, {CONSTANT, {0.5f}}, mem);
    auto var = op::var<float>("x", {batch_size, n_features}, {NONE, {}}, mem);

    Tensor<float> *y = new Tensor<float>({n_samples, n_classes}, {IDENTITY, {}}, mem);

    auto input = layer::input<float>(var);
    auto fc1 = layer::fullyconnected<float>(input->out(), 10);
    auto act1 = layer::activation<float>(fc1->out(), layer::SIGMOID);
    auto fc2 = layer::fullyconnected<float>(act1->out(), n_classes);
    auto act2 = layer::activation<float>(fc2->out(), layer::SIGMOID);
    auto output = layer::output<float>(act2->out());

    std::vector<layer::Layer<float> *> layers = {input, fc1, act1, fc2, act2, output};

    model::nn_params_t p;
    p.n_epochs = 5;
    p.batch_size = batch_size;
    model::NeuralNetwork<float> model(layers, optimizer::CROSS_ENTROPY, optimizer::SGD, p);

    /* training routing */
    model.fit(x, y, metrics);

    // printf("loss: %.5g acc: %.5g time: %.5g ", metrics.loss, metrics.accuracy, metrics.training_time);

    delete x;
    delete y;

    show_success();
}