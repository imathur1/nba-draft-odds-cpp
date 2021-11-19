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
        double BCELoss(std::vector<double> actual, std::vector<double> predict);
        double DBCELoss(std::vector<double> inputs, std::vector<double> actual, std::vector<double> predict);

    
    private:
        std::vector<double> weights;
        double bias;

};