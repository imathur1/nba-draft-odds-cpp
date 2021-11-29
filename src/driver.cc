#include "dataframe.hpp"
#include "mlp.hpp"
#include <iostream>
#include <string>

int main() {
    // std::string filename = "data.csv";
    // DataFrame df = DataFrame(filename);
    // df.ConvertEmptyToInt("pick");
    // DataFrame output = df.GetColumn("pick");
    // std::vector<std::string> cols_to_drop = {"yr", "ht", "num", "ast/tov", "rimmade", "rimmade+rimmiss", "midmade", "midmade+midmiss", 
    //          "rimmade/(rimmade+rimmiss)", "midmade/(midmade+midmiss)", "dunksmade", "dunksmiss+dunksmade",
    //          "dunksmade/(dunksmade+dunksmiss)", "pick", "team", "conf", "type", "year", "pid", "player_name", "Unnamed: 65", "Unnamed: 66"};
    // df.DropColumns(cols_to_drop);
    // df.FillEmpty("Rec Rank", "0");
    // df.DropRowsWithEmptyData();
    // df.ConvertToNumber();
    // df.Normalize();

    std::vector<double>data = {5, 6, 7, 8};
    // std::vector<std::vector<double>> data = {{5, 6}, {5, 8}, {6, 6.5}, {9, 8}, {3, 8}, {7, 4}, {7, 5}, {9, 4}, {8, 5}, {5.5, 7.5},
    //    {1, 1}, {5, 3}, {5, 2}, {4, 6}, {4, 4}, {1, 7}, {2, 5}, {8, 2}, {10, 1}, {3, 3.5}};
    MLP mlp = MLP(data);
    mlp.Train();
    return 0;
}