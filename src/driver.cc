#include "dataframe.hpp"
#include "mlp.hpp"
#include <iostream>
#include <ctime>
#include <iomanip>

int main() {
    // Ensures random numbers are different every time the program runs
    srand ( time(NULL) );

    // Training
    std::string filename = "data.csv";
    DataFrame df = DataFrame(filename);
    df.ConvertEmptyToInt("pick");

    // Undersampling: Too many players don't get drafted which makes for imbalanced classes
    // So drop some of the players who don't get drafted
    df.DropRowsWithColValue("pick", "0", 0.975);
 
    std::vector<std::string> cols_to_drop = {"yr", "ht", "num", "ast/tov", "rimmade", "rimmade+rimmiss", "midmade", "midmade+midmiss", 
             "rimmade/(rimmade+rimmiss)", "midmade/(midmade+midmiss)", "dunksmade", "dunksmiss+dunksmade",
             "dunksmade/(dunksmade+dunksmiss)", "team", "conf", "type", "year", "pid", "player_name", "Unnamed: 65", "Unnamed: 66"};
    df.DropColumns(cols_to_drop);
    df.FillEmpty("Rec Rank", "0");
    df.DropRowsWithEmptyData();

    DataFrame y_df = df.GetColumn("pick");
    y_df.ConvertToNumber();
    df.DropColumns({"pick"});

    df.ConvertToNumber();
    df.Normalize();

    std::vector<std::vector<std::vector<double>>> data = df.GetTrainValidSplit(0.2, y_df);

    MLP mlp = MLP(data[0], data[1], data[2], data[3]);
    double lr = 0.01;
    int num_epochs = 500;
    std::vector<std::vector<double>> metrics = mlp.Train(lr, num_epochs);
    std::cout << "Final Train Loss: " << metrics[0][metrics[0].size() - 1] << "\n";
    std::cout << "Final Train Accuracy: " << metrics[1][metrics[1].size() - 1] << "\n";
    std::cout << "Final Validation Loss: " << metrics[2][metrics[2].size() - 1] << "\n";
    std::cout << "Final Validation Accuracy: " << metrics[3][metrics[3].size() - 1] << "\n";
    
    // Predicting
    std::string test_filename = "prediction.csv";
    DataFrame test_df = DataFrame(test_filename);

    DataFrame player_df = test_df.GetColumn("player_name");
    test_df.DropColumns(cols_to_drop);
    test_df.DropColumns({"pick"});

    test_df.FillEmpty("Rec Rank", "0");

    std::vector<std::string> col_means_str;
    std::vector<double> col_maxes;
    std::vector<double> col_mins;
    for (size_t i = 0; i < df.GetColMeans().size(); i++) {
        col_means_str.push_back(std::to_string(df.GetColMeans()[i]));
        col_maxes.push_back(df.GetColMaxes()[i]);
        col_mins.push_back(df.GetColMins()[i]);
    }
    for (size_t i = 0; i < test_df.GetColNames().size(); i++) {
        test_df.FillEmpty(test_df.GetColNames()[i], col_means_str[i]);
    }
    test_df.ConvertToNumber();
    test_df.Normalize(col_maxes, col_mins);
    std::vector<bool> prediction = mlp.Predict(test_df.GetInputs());

    std::cout << "Drafted Players\n";
    int count = 0;
    for (size_t i = 0; i < prediction.size(); i++) {
        if (prediction[i] == 1) {
            std::cout << "Player: " << player_df.GetData()[i][0] << "\n";
            count += 1;
        }
    }
    std::cout << count << "\n";
    return 0;
}