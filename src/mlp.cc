#include "mlp.hpp"

MLP::MLP() {}

MLP::MLP(std::vector<std::vector<double>> x_train, std::vector<std::vector<double>> y_train, std::vector<std::vector<double>> x_valid, std::vector<std::vector<double>> y_valid) {
    x_train_ = x_train;
    y_train_ = y_train;
    x_valid_ = x_valid;
    y_valid_ = y_valid;

    layer_sizes_[0] = x_train[0].size();
    InitializeMatrices();
}

void MLP::InitializeMatrices() {
    // The nodes in the first layer are the input into the model
    // Initialize the nodes in the rest of the layers to 0
    nodes_.push_back(x_train_);
    for (size_t i = 1; i < layer_sizes_.size(); i++) {
        std::vector<double> v2(layer_sizes_[i], 0);
        std::vector<std::vector<double>> v1(x_train_.size(), v2);
        nodes_.push_back(v1);
    }

    // Initialize the activations of the nodes in all layers besides the 
    // input layer to 0
    for (size_t i = 1; i < layer_sizes_.size(); i++) {
        std::vector<double> v2(layer_sizes_[i], 0);
        std::vector<std::vector<double>> v1(x_train_.size(), v2);
        activations_.push_back(v1);
    }

    // Initialize all weighted connections between all layers to a random
    // number between 0 and 1
    for (size_t i = 0; i < layer_sizes_.size() - 1; i++) {
        std::vector<std::vector<double>> layer_weights;
        for (int j = 0; j < layer_sizes_[i]; j++) {
            std::vector<double> w;
            for (int k = 0; k < layer_sizes_[i + 1]; k++) {
                w.push_back(((double) std::rand() / (RAND_MAX)));
            }
            layer_weights.push_back(w);
        }
        weights_.push_back(layer_weights);
    }

    // Initialize the biases of all nodes in all layers except the input layer to a random
    // number between 0 and 1
    for (size_t i = 1; i < layer_sizes_.size(); i++) {
        std::vector<std::vector<double>> v1;
        for (size_t j = 0; j < x_train_.size(); j++) {
            std::vector<double> v2;
            for (int k = 0; k < layer_sizes_[i]; k++) {
                v2.push_back(((double) std::rand() / (RAND_MAX)));
            }
            v1.push_back(v2);
        }
        biases_.push_back(v1);
    }
}

std::vector<std::vector<double>> MLP::Train(double lr, int num_epochs) {
    std::vector<double> train_losses;
    std::vector<double> train_accuracies;
    std::vector<double> valid_losses;
    std::vector<double> valid_accuracies;
    for (int i = 0; i < num_epochs; i++) {
        ForwardPropagation();
        std::tuple<std::vector<std::vector<std::vector<double>>>, std::vector<std::vector<double>>> gradients = BackwardPropagation();
        UpdateWeightsBiases(std::get<0>(gradients), std::get<1>(gradients), lr);
        
        double train_loss = ComputeLoss(y_train_, activations_[activations_.size() - 1]);
        train_losses.push_back(train_loss);
        double train_accuracy = ComputeAccuracy(y_train_, activations_[activations_.size() - 1]);
        train_accuracies.push_back(train_accuracy);

        std::vector<std::vector<double>> valid_pred = Test(x_valid_);
        double valid_loss = ComputeLoss(y_valid_, valid_pred);
        valid_losses.push_back(valid_loss);
        double valid_accuracy = ComputeAccuracy(y_valid_, valid_pred);
        valid_accuracies.push_back(valid_accuracy);
    }

    std::vector<std::vector<double>> metrics;
    metrics.push_back(train_losses);
    metrics.push_back(train_accuracies);
    metrics.push_back(valid_losses);
    metrics.push_back(valid_accuracies);
    return metrics;
}

std::vector<std::vector<double>> MLP::Test(std::vector<std::vector<double>> x) {
    // Run input through existing weights and biases of model to get output
    std::vector<std::vector<std::vector<double>>> new_nodes;
    new_nodes.push_back(x);
    std::vector<std::vector<std::vector<double>>> new_activations;
    std::vector<std::vector<std::vector<double>>> new_biases;
    for (size_t i = 0; i < layer_sizes_.size(); i++) {
        if (i != 0) {
            std::vector<double> n2(layer_sizes_[i], 0);
            std::vector<std::vector<double>> n1(x.size(), n2);
            new_nodes.push_back(n1);

            std::vector<double> a2(layer_sizes_[i], 0);
            std::vector<std::vector<double>> a1(x.size(), a2);
            new_activations.push_back(a1);

            std::vector<double> b1 = biases_[i - 1][0];
            std::vector<std::vector<double>> b2;
            for (size_t j = 0; j < x.size(); j++) {
                b2.push_back(b1);
            }
            new_biases.push_back(b2);
        }
    }

    for (size_t i = 0; i < layer_sizes_.size() - 1; i++) {
        std::vector<std::vector<double>> z;
        if (i == 0) {
            z = MatMul(new_nodes[i], weights_[i]);
            for (size_t j = 0; j < z.size(); j++) {
                for (size_t k = 0; k < z[j].size(); k++) {
                    z[j][k] += new_biases[i][j][k];
                }
            }
        } else {
            z = MatMul(new_activations[i - 1], weights_[i]);
            for (size_t j = 0; j < z.size(); j++) {
                for (size_t k = 0; k < z[j].size(); k++) {
                    z[j][k] += new_biases[i][j][k];
                }
            }
        }
        new_nodes[i + 1] = z;
        if (i == layer_sizes_.size() - 2) {
            new_activations[i] = Sigmoid(z);
        } else {
            new_activations[i] = ReLU(z);
        }
    }
    return new_activations[new_activations.size() - 1];
}

std::vector<bool> MLP::Predict(std::vector<std::vector<double>> x) {
    std::vector<std::vector<double>> prediction = Test(x);
    std::vector<bool> output;
    for (size_t i = 0; i < prediction.size(); i++) {
        if (prediction[i][0] > 0.5) {
            output.push_back(true);
        } else {
            output.push_back(false);
        }
    }
    return output;
}

std::tuple<std::vector<std::vector<std::vector<double>>>, std::vector<std::vector<double>>> MLP::BackwardPropagation() {
    // Store the gradient of all weights and biases in nested vectors
    std::vector<std::vector<std::vector<double>>> weights_gradient;
    std::vector<std::vector<double>> biases_gradient;
    
    // Calculates the partial derivatives of weights and biases
    for (size_t i = weights_.size() -  1; i >= 0; i--) {
        size_t num_partials = 3 + 2 * (weights_.size() - i - 1);
        std::vector<std::vector<std::vector<double>>> w_partials;
        std::vector<std::vector<std::vector<double>>> b_partials;
        
        // Finds partial derivatives for each layer
        for (size_t j = 0; j < num_partials; j++) {
            std::vector<std::vector<double>> w_partial;
            std::vector<std::vector<double>> b_partial;
            if (j == 0) {
                // BCE Loss
                // dL/da
                w_partial = BCEPrime(y_train_, activations_[activations_.size() - 1]);
                b_partial = w_partial;
                w_partials.push_back(w_partial);
                b_partials.push_back(b_partial);
            } else if (j == num_partials - 1) {
                // Previous layer activation function output / initial inputs
                // dz/dw
                int index = activations_.size() - ceil(j / 2) - 1;
                std::vector<std::vector<double>> a;
                if (index == -1) {
                    a = nodes_[0];
                } else {
                    a = activations_[index];
                }
                w_partial = a;

                // dz/db
                b_partial.push_back({1});
                w_partials.push_back(w_partial);
                b_partials.push_back(b_partial);
            } else if (j % 2 != 0) {
                // If it's the last layer, store the derivative of the sigmoid activation function
                // If it's not the last layer, store the derivative of the ReLU activation function
                // da/dz
                std::vector<std::vector<double>> a = activations_[activations_.size() - ((j + 1) / 2)];
                if (j == 1) {
                    w_partial = SigmoidPrime(a);
                } else {
                    w_partial = ReLUPrime(a);                    
                }
                b_partial = w_partial;
                w_partials.push_back(w_partial);
                b_partials.push_back(b_partial);
            } else {
                // Otherwise transpose specific weights for matrix math
                // da/dz_prev
                std::vector<std::vector<double>> z = nodes_[nodes_.size() - ((j + 1) / 2)];
                std::vector<std::vector<double>> w = weights_[weights_.size() - ((j + 1) / 2)];
                std::vector<std::vector<double>> w_partial = Transpose(w);
                std::vector<std::vector<double>> b_partial = w_partial;
                w_partials.push_back(w_partial);
                b_partials.push_back(b_partial);
            }
        }

        // Do matrix multiplication and element-wise multiplication for both gradients
        std::vector<std::vector<double>> w_gradient;
        std::vector<std::vector<double>> b_gradient;
        for (size_t j = 0; j < w_partials.size() - 1; j+=2) {
            if (w_gradient.empty()) {
                w_gradient = w_partials[j];
            } else {
                w_gradient = MatMul(w_gradient, w_partials[j]);
            }

            for (size_t x = 0; x < w_gradient.size(); x++) {
                for (size_t y = 0; y < w_gradient[x].size(); y++) {
                    w_gradient[x][y] *= w_partials[j + 1][x][y];
                }
            }

            if (b_gradient.empty()) {
                b_gradient = b_partials[j];
            } else {
                b_gradient = MatMul(b_gradient, b_partials[j]);
            } 
            for (size_t x = 0; x < b_gradient.size(); x++) {
                for (size_t y = 0; y < b_gradient[x].size(); y++) {
                    b_gradient[x][y] *= b_partials[j + 1][x][y];
                }
            }
        }


        // Finds the average gradient from all of the samples
        std::vector<std::vector<double>> tw_partials = Transpose(w_partials[w_partials.size()-1]);
        w_gradient = MatMul(tw_partials, w_gradient);

        for (size_t i = 0; i < w_gradient.size(); i++) {
            for (size_t j = 0; j < w_gradient[i].size(); j++) {
                w_gradient[i][j] /= y_train_.size();
            }
        }
        std::vector<double> b_gradient_mean; 
        for (size_t i = 0; i < b_gradient[0].size(); i++) {
            double total = 0.0;
            for (size_t j = 0; j < b_gradient.size(); j++) {
                total += b_gradient[j][i];
            }
            total /= b_gradient.size();
            b_gradient_mean.push_back(total);
        }

        weights_gradient.push_back(w_gradient);
        biases_gradient.push_back(b_gradient_mean);
        if (i == 0) {
            break;
        }
    }

    // Reverse the gradients to match the forward direction of the mlp
    std::reverse(weights_gradient.begin(), weights_gradient.end());
    std::reverse(biases_gradient.begin(), biases_gradient.end());
    
    // Return a tuple of the gradients
    std::tuple<std::vector<std::vector<std::vector<double>>>, std::vector<std::vector<double>>> output = {weights_gradient, biases_gradient};
    return output;
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
        for (size_t j = 0; j < m2[0].size(); j++) {
            double num = 0;
            for (size_t k = 0; k < m2.size(); k++) {
                num += m1[i][k] * m2[k][j];
            }
            vec.push_back(num);
        }
        result.push_back(vec);
    }
    return result;
}

void MLP::ForwardPropagation() {
    for (size_t i = 0; i < layer_sizes_.size() - 1; i++) {
        std::vector<std::vector<double>> z;
        if (i == 0) {
            // If it's the input layer, compute z = x * w + b
            // where z is the output in the next layer, x is the input,
            // w is the weight, and b is the bias.
            // Compute this for all connections between the input and hidden layer
            z = MatMul(nodes_[i], weights_[i]);
            for (size_t j = 0; j < z.size(); j++) {
                for (size_t k = 0; k < z[j].size(); k++) {
                    z[j][k] += biases_[i][j][k];
                }
            }
        } else {
            // If it's not the input layer, compute z = a * w + b
            // where z is the output in the next layer, a is the activation of the previous layer,
            // w is the weight, and b is the bias.
            // Compute this for all connections between the hidden layers
            z = MatMul(activations_[i - 1], weights_[i]);
            for (size_t j = 0; j < z.size(); j++) {
                for (size_t k = 0; k < z[j].size(); k++) {
                    z[j][k] += biases_[i][j][k];
                }
            }
        }
        nodes_[i + 1] = z;
        // // Use ReLU for the activation function unless it is the output layer
        // // In that case use sigmoid
        if (i == layer_sizes_.size() - 2) {
            activations_[i] = Sigmoid(z);
        } else {
            activations_[i] = ReLU(z);
        }
    }
}

void MLP::UpdateWeightsBiases(std::vector<std::vector<std::vector<double>>> weights_gradient, std::vector<std::vector<double>> biases_gradient, double lr) {
    for (size_t i = 0; i < weights_.size(); i++) {
        for (size_t j = 0; j < weights_[i].size(); j++) {
            for (size_t k = 0; k < weights_[i][j].size(); k++) {
                weights_[i][j][k] -= lr * weights_gradient[i][j][k];
            }
        }
    }
    for (size_t i = 0; i < biases_.size(); i++) {
        for (size_t j = 0; j < biases_[i].size(); j++) {
            for (size_t k = 0; k < biases_[i][j].size(); k++) {
                biases_[i][j][k] -= lr * biases_gradient[i][k];
            }
        }
    }
}

std::vector<std::vector<double>> MLP::Sigmoid(std::vector<std::vector<double>> z) {
    std::vector<std::vector<double>> all_results;
    for (size_t i = 0; i < z.size(); i++) {
        std::vector<double> results;
        for (size_t j  = 0; j < z[i].size(); j++) {
            results.push_back((1/(1 + exp(-z[i][j]))));
        }
        all_results.push_back(results);
    }
    return all_results;
}

std::vector<std::vector<double>> MLP::ReLU(std::vector<std::vector<double>> z) {
    std::vector<std::vector<double>> all_results;
    for (size_t i = 0; i < z.size(); i++) {
        std::vector<double> results;
        for (size_t j  = 0; j < z[i].size(); j++) {
            if (z[i][j] < 0) {
                results.push_back(0);
            } else {
                results.push_back(z[i][j]);
            }
        }
        all_results.push_back(results);
    }
    return all_results;
}

std::vector<std::vector<double>> MLP::SigmoidPrime(std::vector<std::vector<double>> z) {
    std::vector<std::vector<double>> all_results;
    for (size_t i = 0; i < z.size(); i++) {
        std::vector<double> results;
        for (size_t j  = 0; j < z[i].size(); j++) {
            results.push_back(z[i][j] * (1 - z[i][j]));
        }
        all_results.push_back(results);
    }
    return all_results;
}

std::vector<std::vector<double>> MLP::ReLUPrime(std::vector<std::vector<double>> z) {
    std::vector<std::vector<double>> all_results;
    for (size_t i = 0; i < z.size(); i++) {
        std::vector<double> results;
        for (size_t j  = 0; j < z[i].size(); j++) {
            if (z[i][j] < 0) {
                results.push_back(0);
            } else {
                results.push_back(1);
            }
        }
        all_results.push_back(results);
    }
    return all_results;
}

std::vector<std::vector<double>> MLP::BCE(std::vector<std::vector<double>> actual, std::vector<std::vector<double>> predict) {
    std::vector<std::vector<double>> all_losses;
    for (size_t i = 0; i < actual.size(); i++) {
        std::vector<double> losses;
        for (size_t j = 0; j < actual[i].size(); j++) {
            double loss = -1 * (actual[i][j] * log10(predict[i][j]) + (1 - actual[i][j]) * log10(1 - predict[i][j]));
            losses.push_back(loss);
        }
        all_losses.push_back(losses);
    }
    return all_losses;
}

std::vector<std::vector<double>> MLP::BCEPrime(std::vector<std::vector<double>> actual, std::vector<std::vector<double>> predict) {
    std::vector<std::vector<double>> all_losses;
    for (size_t i = 0; i < actual.size(); i++) {
        std::vector<double> losses;
        for (size_t j = 0; j < actual[i].size(); j++) {
            double loss = -1 * actual[i][j] / predict[i][j] + (1 - actual[i][j]) / (1 - predict[i][j]);
            losses.push_back(loss);
        }
        all_losses.push_back(losses);
    }
    return all_losses;
}

double MLP::ComputeLoss(std::vector<std::vector<double>> actual, std::vector<std::vector<double>> predict) {
    std::vector<std::vector<double>> losses = BCE(actual, predict);
    double total_loss = 0;
    for (size_t i = 0; i < losses.size(); i++) {
        for (size_t j = 0; j < losses[i].size(); j++) {
            total_loss += losses[i][j];
        }
    }
    return total_loss / (actual.size() * actual[0].size());
}

double MLP::ComputeAccuracy(std::vector<std::vector<double>> actual, std::vector<std::vector<double>> predict) {
    double accuracy = 0;
    for (size_t i = 0; i < predict.size(); i++) {
        for (size_t j = 0; j < predict[i].size(); j++) {
            double prediction = 0.0;
            if (predict[i][j] > 0.5) {
                prediction = 1.0;
            }
            if (prediction == actual[i][j]) {
                accuracy += 1.0;
            }
        }
    }
    return accuracy / (actual.size() * actual[0].size());
}