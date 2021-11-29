#ifndef MLP_H
#define MLP_H

#include <vector>
#include <numeric>
#include <iostream>
#include <math.h>

class MLP {
    public:
        MLP();
        MLP(std::vector<double> input_data);
        void Train();
        bool Predict(std::vector<double> data);

    private:
        void InitializeMatrices();
        void ForwardPropagation();
        void BackwardPropagation();
        void UpdateWeightsBiases();

        double Sigmoid(double z);
        double ReLU(double z);
        double SigmoidPrime(double z);
        double ReLUPrime(double z);
        std::vector<double> BCE(std::vector<double> actual, std::vector<double> predict);
        std::vector<double> BCEPrime(std::vector<double> actual, std::vector<double> predict);
        
        std::vector<int> layer_sizes_ = {0, 2, 1, 1};
        std::vector<std::vector<double>> data_;
        std::vector<std::vector<double>> activations_;
        std::vector<std::vector<std::vector<double>>> weights_;
        std::vector<std::vector<double>> biases_;
};

#endif