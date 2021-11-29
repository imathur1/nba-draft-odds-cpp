#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include "dataframe.hpp"
#include <vector>

class Perceptron {
    public:
        Perceptron();
        double Sigmoid(double data);
        double Relu(double data);
        double Pred(std::vector<double> data);
        double DSigmoid(double data);
        int DRelu(double data);
        std::vector<double> BCELoss(std::vector<double> actual, std::vector<double> predict);
        std::vector<double> DBCELoss(std::vector<double> actual, std::vector<double> predict);

    
    private:
        std::vector<double> weights;
        double bias;

};

#endif