#include "perceptron.hpp"
#include <math.h>

Perceptron::Perceptron() {}

float Perceptron::sigmoid(float data) {
    return (1/(1 + exp(-data)));
}

float Perceptron::relu(float data) {
    if (data < 0) {
        return 0;
    }
    return data;
}

float Perceptron::pred(std::vector<float> data) {
    float ret;
    int counter = 0;
    for (float i : data) {
        ret = ret + (i*weights[counter]);
        counter = counter + 1;
    }
    ret = ret + bias;
    return relu(ret);
}