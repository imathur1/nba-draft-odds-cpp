#include "dataframe.hpp"
#include <vector>

class Perceptron {
    public:
        Perceptron();
        float sigmoid(float data);
        float relu(float data);
        float pred(std::vector<float> data);
    
    private:
        std::vector<float> weights;
        float bias;

};