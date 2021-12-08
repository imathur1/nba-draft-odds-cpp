#ifndef MLP_H
#define MLP_H

#include <vector>
#include <numeric>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <tuple>
class MLP {
    public:
        MLP();
        MLP(std::vector<std::vector<double>> x_train, std::vector<std::vector<double>> y_train, std::vector<std::vector<double>> x_valid, std::vector<std::vector<double>> y_valid);
        std::vector<std::vector<double>> Train(double lr, int num_epochs);
        std::vector<bool> Predict(std::vector<std::vector<double>> x);
    // private:
        void InitializeMatrices();
        void ForwardPropagation();
        std::tuple<std::vector<std::vector<std::vector<double>>>, std::vector<std::vector<double>>> BackwardPropagation();
        void UpdateWeightsBiases(std::vector<std::vector<std::vector<double>>> weights_gradient, std::vector<std::vector<double>> biases_gradient, double lr);
        
        std::vector<std::vector<double>> Test(std::vector<std::vector<double>> x);
        
        double ComputeLoss(std::vector<std::vector<double>> actual, std::vector<std::vector<double>> predict);
        double ComputeAccuracy(std::vector<std::vector<double>> actual, std::vector<std::vector<double>> predict);

        std::vector<std::vector<double>> Sigmoid(std::vector<std::vector<double>> z);
        std::vector<std::vector<double>> ReLU(std::vector<std::vector<double>> z);
        std::vector<std::vector<double>> SigmoidPrime(std::vector<std::vector<double>> z);
        std::vector<std::vector<double>> ReLUPrime(std::vector<std::vector<double>> z);
        std::vector<std::vector<double>> BCE(std::vector<std::vector<double>> actual, std::vector<std::vector<double>> predict);
        std::vector<std::vector<double>> BCEPrime(std::vector<std::vector<double>> actual, std::vector<std::vector<double>> predict);

        std::vector<std::vector<double>> Transpose(std::vector<std::vector<double>> m);
        std::vector<std::vector<double>> MatMul(std::vector<std::vector<double>> m1, std::vector<std::vector<double>> m2);
        
        std::vector<std::vector<double>> x_train_;
        std::vector<std::vector<double>> y_train_;
        std::vector<std::vector<double>> x_valid_;
        std::vector<std::vector<double>> y_valid_;
        
        std::vector<int> layer_sizes_ = {0, 4, 1};
        std::vector<std::vector<std::vector<double>>> nodes_;
        std::vector<std::vector<std::vector<double>>> activations_;
        std::vector<std::vector<std::vector<double>>> weights_;
        std::vector<std::vector<std::vector<double>>> biases_;
};

#endif