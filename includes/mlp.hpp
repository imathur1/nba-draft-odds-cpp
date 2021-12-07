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
        MLP(std::vector<double> input_data);
        void Train();
        bool Predict(std::vector<double> data);

    private:
        void InitializeMatrices();
        void ForwardPropagation();
        std::tuple<std::vector<std::vector<std::vector<double>>>, std::vector<std::vector<double>>> BackwardPropagation(std::vector<double> actual);
        void UpdateWeightsBiases();

        double Sigmoid(double z);
        double ReLU(double z);
        double SigmoidPrime(double z);
        double ReLUPrime(double z);
        std::vector<double> BCE(std::vector<double> actual, std::vector<double> predict);
        std::vector<double> BCEPrime(std::vector<double> actual, std::vector<double> predict);

        std::vector<std::vector<double>> Transpose(std::vector<std::vector<double>> m);
        std::vector<std::vector<double>> mulMat(std::vector<std::vector<double>> m1, std::vector<std::vector<double>>m2);
        
        std::vector<int> layer_sizes_ = {0, 4, 1};
        std::vector<std::vector<double>> data_;
        std::vector<std::vector<double>> activations_;
        std::vector<std::vector<std::vector<double>>> weights_;
        std::vector<std::vector<double>> biases_;
        double lr = 0.01;
};

#endif