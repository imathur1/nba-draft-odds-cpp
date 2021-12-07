#include "mlp.hpp"
 
MLP::MLP() {}

MLP::MLP(std::vector<double> input_data) {
    layer_sizes_[0] = input_data.size();
    nodes_.push_back(input_data);
    InitializeMatrices();
}

void MLP::InitializeMatrices() {
    // The nodes in the first layer are the input into the model
    // Initialize the nodes in the rest of the layers to 0
    for (size_t i = 1; i < layer_sizes_.size(); i++) {
        std::vector<double> d(layer_sizes_[i], 0);
        nodes_.push_back(d);
    }

    // Initialize the activations of the nodes in all layers besides the 
    // input layer to 0
    for (size_t i = 1; i < layer_sizes_.size(); i++) {
        std::vector<double> a(layer_sizes_[i], 0);
        activations_.push_back(a);
    }

    // Initialize all weighted connections between all layers to a random
    // number between 0 and 1
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

    // Initialize the biases of all nodes in all layers except the input layer to a random
    // number between 0 and 1
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
    if (data.size() == 0) {
        
    }
    return false;
}

std::tuple<std::vector<std::vector<std::vector<double>>>, std::vector<std::vector<double>>> MLP::BackwardPropagation(std::vector<double> actual) {
    //Store the gradient of all weights and biases in nested vectors
    std::vector<std::vector<std::vector<double>>> weightsgrad;
    std::vector<std::vector<double>> biasesgrad;

    //Calculates the partial derivatives of weights and biases
    for (size_t i = weights_.size() -  1; i >= 0; i--) {
        size_t num_partials = 3 + 2 * (weights_.size() - i - 1);
        std::vector<std::vector<std::vector<double>>> wparts;
        std::vector<std::vector<std::vector<double>>> bparts;

        //Finds partial derivatives for each layer
        for (size_t j = 0; j < num_partials; j++) {
            std::vector<std::vector<double>> wpart;
            std::vector<std::vector<double>> bpart;

            // BCE Loss
            if (j == 0) {
                wpart.push_back(BCEPrime(actual, activations_[activations_.size() - 1]));
                bpart = wpart;
            
            // Previous layer activation function output / initial inputs
            } else if (j == num_partials - 1) {
                int index = activations_.size() - ceil(j / 2) - 1;
                std::vector<double> a;
                if (index == -1) {
                    a = nodes_[0];
                } else {
                    a = activations_[index];
                }
                wpart.push_back(a);
                bpart.push_back({1});
            
            // Activation functions
            } else if (j % 2 != 0) {
                std::vector<double> a = activations_[activations_.size() - ceil(j/2)];
                //If it's the last layer, store the derivative of the sigmoid activation function
                if (j == 1) {
                    for (size_t i = 0; i < a.size(); i++) {
                        std::vector<double> data = {SigmoidPrime(a[i])};
                        wpart.push_back(data);
                    }
                //If it's not the last layer, store the derivative of the ReLU activation function
                } else {
                    for (size_t i = 0; i < a.size(); i++) {
                        std::vector<double> data = {ReLUPrime(a[i])};
                        wpart.push_back(data);
                    }
                }
                bpart = wpart;

            // Otherwise transpose specific weights for matrix math
            } else {
                std::vector<std::vector<double>> w = weights_[weights_.size() - ceil(j/2)];
                wpart = Transpose(w);
                bpart = wpart;
            }
            wparts.push_back(wpart);
            bparts.push_back(bpart);
        }

        //Do matrix multiplication and element-wise multiplication for both gradients
        std::vector<std::vector<double>> wgrad;
        std::vector<std::vector<double>> bgrad;
        for (size_t j = 0; j < wparts.size(); j+=2) {
            if (wgrad.empty()) {
                wgrad = wparts[j];
            } else {
                wgrad = MatMul(wgrad, wparts[j]);
            } 
            for (size_t x = 0; x < wgrad.size(); x++) {
                for (size_t y = 0; y < wgrad[x].size(); y++) {
                    wgrad[x][y] *= wparts[j][x][y];
                }
            }

            if (bgrad.empty()) {
                bgrad = bparts[j];
            } else {
                bgrad = MatMul(bgrad, bparts[j]);
            } 
            for (size_t x = 0; x < bgrad.size(); x++) {
                for (size_t y = 0; y < bgrad[x].size(); y++) {
                    bgrad[x][y] *= bparts[j][x][y];
                }
            }
        }
        
        //Finds the average gradient from all of the samples
        std::vector<std::vector<double>> twparts = Transpose(wparts[wparts.size()-1]);
        wgrad = MatMul(twparts, wgrad);
        for (size_t i = 0; i < wgrad.size(); i++) {
            for (size_t j = 0; j < wgrad[i].size(); j++) {
                wgrad[i][j] /= actual.size();
            }
        }
        std::vector<double> bgradient; 
        for (size_t i = 0; i < bgrad[0].size(); i++) {
            double total = 0.0;
            for (size_t j = 0; j < bgrad.size(); j++) {
                total += bgrad[j][i];
            }
            total /= bgrad.size();
            bgradient.push_back(total);
        }
        weightsgrad.push_back(wgrad);
        biasesgrad.push_back(bgradient);
    }

    //Reverse the gradients to match the forward direction of the mlp
    std::reverse(weightsgrad.begin(), weightsgrad.end());
    std::reverse(biasesgrad.begin(), biasesgrad.end());

    //Return a tuple of the gradients
    std::tuple<std::vector<std::vector<std::vector<double>>>, std::vector<std::vector<double>>> items = {weightsgrad, biasesgrad};
    return items;
}

std::vector<std::vector<double>> MLP::Transpose(std::vector<std::vector<double>> m) {
    std::vector<std::vector<double>> transposed(m[0].size(), std::vector<double>());
    for (size_t i = 0; i < m.size(); i++) {
        for (size_t j = 0; j < m[i].size(); j++) {
            transposed[j].push_back(m[i][j]);
        }
    }
    return transposed;
}

std::vector<std::vector<double>> MLP::MatMul(std::vector<std::vector<double>> m1, std::vector<std::vector<double>>m2) {
    std::vector<std::vector<double>> result; 
    for (size_t i = 0; i < m1.size(); i++) {
        std::vector<double> vec;
        for (size_t j = 0; j < m2.at(i).size(); j++) {
            double num = 0;
            for (size_t k = 0; k < m2.size(); k++) {
                num += m1.at(i).at(k) * m2.at(k).at(j);
            }
            vec.push_back(num);
        }
        result.push_back(vec);
    }
    return result;
}

void MLP::ForwardPropagation() {
    for (size_t i = 0; i < layer_sizes_.size() - 1; i++) {
        std::vector<double> z;
        if (i == 0) {
            // If it's the input layer, compute z = x * w + b
            // where z is the output in the next layer, x is the input,
            // w is the weight, and b is the bias.
            // Compute this for all connections between the input and hidden layer
            for (int j = 0; j < layer_sizes_[i + 1]; j++) {
                double prod = std::inner_product(std::begin(nodes_[i]), std::end(nodes_[i]),
                                std::begin(weights_[i][j]), 0.0) + biases_[i][j];
                z.push_back(prod);
            }
        } else {
            // If it's not the input layer, compute z = a * w + b
            // where z is the output in the next layer, a is the activation of the previous layer,
            // w is the weight, and b is the bias.
            // Compute this for all connections between the input and hidden layer
            for (int j = 0; j < layer_sizes_[i + 1]; j++) {
                double prod = std::inner_product(std::begin(activations_[i - 1]), std::end(activations_[i - 1]),
                                std::begin(weights_[i][j]), 0.0) + biases_[i][j];
                z.push_back(prod);
            }
        }
        nodes_[i + 1] = z;
        // Use ReLU for the activation function unless it is the output layer
        // In that case use sigmoid
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
    // Calculates sigmoid value for a given input
    return (1/(1 + exp(-z)));
}

double MLP::ReLU(double z) {
    // Calculates ReLU value for a given input
    if (z < 0) {
        return 0;
    }
    return z;
}

double MLP::SigmoidPrime(double z) {
    // Calculates the derivative of sigmoid for a given input
    return z * (1 - z);
}

double MLP::ReLUPrime(double z) {
    // Calculates the derivative of ReLU for a given input
    if (z <= 0) {
        return 0;
    }
    return 1;
}

std::vector<double> MLP::BCE(std::vector<double> actual, std::vector<double> predict) {
    // Calculates Binary Cross Entropy measures between actual and predicted values
    std::vector<double> losses;
    for (size_t i  = 0; i < actual.size(); i++) {
        double loss = -1 * (actual[i] * log10(predict[i]) + (1 - actual[i]) * log10(1 - predict[i]));
        losses.push_back(loss);
    }
    return losses;
}

std::vector<double> MLP::BCEPrime(std::vector<double> actual, std::vector<double> predict) {
    // Calculates Binary Cross Entropy derivitave measures between actual and predicted values
    std::vector<double> losses;
    for (size_t i = 0; i < actual.size(); i++) {
        double loss = -1 * actual[i] / predict[i] + (1 - actual[i]) / (1 - predict[i]);
        losses.push_back(loss);
    }
    return losses;
}