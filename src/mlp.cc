#include "mlp.hpp"
 
MLP::MLP() {}

MLP::MLP(std::vector<double> input_data) {
    layer_sizes_[0] = input_data.size();
    data_.push_back(input_data);
    InitializeMatrices();
}

void MLP::InitializeMatrices() {
    for (size_t i = 1; i < layer_sizes_.size(); i++) {
        std::vector<double> d(layer_sizes_[i], 0);
        data_.push_back(d);
    }

    for (size_t i = 1; i < layer_sizes_.size(); i++) {
        std::vector<double> a(layer_sizes_[i], 0);
        activations_.push_back(a);
    }

    for (size_t i = 0; i < layer_sizes_.size() - 1; i++) {
        std::vector<std::vector<double>> layer_weights;
        for (int j = 0; j < layer_sizes_[i + 1]; j++) {
            std::vector<double> w;
            for (int k = 0; k < layer_sizes_[i]; k++) {
                w.push_back(((double) std::rand() / (RAND_MAX)));
            }
            layer_weights.push_back(w);
        }
        weights_.push_back(layer_weights);
    }

    for (size_t i = 1; i < layer_sizes_.size(); i++) {
        std::vector<double> b;
        for (int j = 0; j < layer_sizes_[i]; j++) {
            b.push_back(((double) std::rand() / (RAND_MAX)));
        }
        biases_.push_back(b);
    }
}

void MLP::Train() {
    ForwardPropagation();
}

bool Predict(std::vector<double> data) {

}

void MLP::ForwardPropagation() {
    for (size_t i = 0; i < layer_sizes_.size() - 1; i++) {
        std::vector<double> z;
        if (i == 0) {
            for (int j = 0; j < layer_sizes_[i + 1]; j++) {
                double prod = std::inner_product(std::begin(data_[i]), std::end(data_[i]),
                                std::begin(weights_[i][j]), 0.0) + biases_[i][j];
                z.push_back(prod);
            }
        } else {
            for (int j = 0; j < layer_sizes_[i + 1]; j++) {
                double prod = std::inner_product(std::begin(activations_[i - 1]), std::end(activations_[i - 1]),
                                std::begin(weights_[i][j]), 0.0) + biases_[i][j];
                z.push_back(prod);
            }
        }
        data_[i + 1] = z;
        if (i == layer_sizes_.size() - 2) {
            for (int j = 0; j < layer_sizes_[i + 1]; j++) {
                double y = Sigmoid(z[j]);
                activations_[i][j] = y;
            }
        } else {
            for (int j = 0; j < layer_sizes_[i + 1]; j++) {
                double y = ReLU(z[j]);
                activations_[i][j] = y;
            }
        }
    }
}

double MLP::Sigmoid(double z) {
    return (1/(1 + exp(-z)));
}

double MLP::ReLU(double z) {
    if (z < 0) {
        return 0;
    }
    return z;
}

double MLP::SigmoidPrime(double z) {
    return z * (1 - z);
}

double MLP::ReLUPrime(double z) {
    if (z <= 0) {
        return 0;
    }
    return 1;
}

std::vector<double> MLP::BCE(std::vector<double> actual, std::vector<double> predict) {
    std::vector<double> losses;
    for (size_t i  = 0; i < actual.size(); i++) {
        double loss = -1 * (actual[i] * log10(predict[i]) + (1 - actual[i]) * log10(1 - predict[i]));
        losses.push_back(loss);
    }
    return losses;
}

std::vector<double> MLP::BCEPrime(std::vector<double> actual, std::vector<double> predict) {
    std::vector<double> losses;
    for (size_t i = 0; i < actual.size(); i++) {
        double loss = -1 * actual[i] / predict[i] + (1 - actual[i]) / (1 - predict[i]);
        losses.push_back(loss);
    }
    return losses;
}