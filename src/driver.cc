#include <iostream>
#include <string>
#include "dataframe.hpp"

int main() {
    std::string filename = "data.csv";
    DataFrame df = DataFrame(filename);
    df.ConvertEmptyToInt("pick");
    DataFrame output = df.GetColumn("pick");
    std::vector<std::string> cols_to_drop = {"yr", "ht", "num", "ast/tov", "rimmade", "rimmade+rimmiss", "midmade", "midmade+midmiss", 
             "rimmade/(rimmade+rimmiss)", "midmade/(midmade+midmiss)", "dunksmade", "dunksmiss+dunksmade",
             "dunksmade/(dunksmade+dunksmiss)", "pick", "team", "conf", "type", "year", "pid", "player_name", "Unnamed: 65", "Unnamed: 66"};
    df.DropColumns(cols_to_drop);
    df.FillEmpty("Rec Rank", "0");
    df.DropRowsWithEmptyData();
    df.ConvertToNumber();
    df.Normalize();
    return 0;
}