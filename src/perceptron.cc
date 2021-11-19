#include "perceptron.hpp"
#include <math.h>

Perceptron::Perceptron() {}

double Perceptron::Sigmoid(double data) {
    return (1/(1 + exp(-data)));
}

double Perceptron::Relu(double data) {
    if (data < 0) {
        return 0;
    }
    return data;
}

double Perceptron::Pred(std::vector<double> data) {
    double ret;
    int counter = 0;
    for (double i : data) {
        ret = ret + (i*weights[counter]);
        counter = counter + 1;
    }
    ret = ret + bias;
    return Relu(ret);
}

double Perceptron::DSigmoid(double data) {
    double sigmoid = Sigmoid(data);
    return sigmoid * (1 - sigmoid);
}

int Perceptron::DRelu(double data) {
    if (data <= 0) {
        return 0;
    }
    return 1;
}

double Perceptron::BCELoss(std::vector<double> actual, std::vector<double> predict) {
    double loss = 0.0;
    for (size_t i  = 0; i < actual.size(); i++) {
        double diff = -1 * (actual[i] * log10(predict[i]) + (1 - actual[i]) * log10(1 - predict[i]));
        loss += diff;
    }
    loss /= actual.size();
    return loss;
}

double Perceptron::DBCELoss(std::vector<double> inputs, std::vector<double> actual, std::vector<double> predict) {
    double loss = 0.0;
    for (size_t i = 0; i < inputs.size(); i++) {
        double diff = inputs[i] * (predict[i] - actual[i]);
        loss += diff;
    }
    loss /= inputs.size();
    return loss;
}