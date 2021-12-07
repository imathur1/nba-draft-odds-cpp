#ifndef CATCH_CONFIG_MAIN
#  define CATCH_CONFIG_MAIN
#endif

#include "catch.hpp"
#include "dataframe.hpp"
#include "mlp.hpp"
#include <string>
#include <vector>
#include <stdexcept>


TEST_CASE("DataFrame ReadCSV", "[DF_ReadCSV]") {
    SECTION("Reading csv with valid name") {
        std::string filename = "data.csv";
        DataFrame df = DataFrame(filename);
        REQUIRE(df.GetColNames().size() == 65);
        REQUIRE(df.GetData().size() == 61061);
        REQUIRE(df.GetData().at(0).size() == 65);
    }
    SECTION("Reading invalid csv name") {
        std::string filename = "dataf.csv";
        REQUIRE_THROWS_AS(DataFrame(filename), std::runtime_error);
    }
}

TEST_CASE("DataFrame constructor from 2D vector", "[DF_constructor]") {
    SECTION("Initializing with data from another DataFrame") {
        std::string filename = "data.csv";
        DataFrame df = DataFrame(filename);
        std::vector<std::string> col_names = {"a", "b"};
        std::vector<std::string> row1 = {"1", "2"};
        std::vector<std::string> row2 = {"s", "a"};
        std::vector<std::vector<std::string>> new_data;
        new_data.push_back(col_names);
        new_data.push_back(row1);
        new_data.push_back(row2);
        DataFrame new_df = DataFrame(new_data);
        REQUIRE(new_df.GetColNames().size() == 2);
        REQUIRE(new_df.GetData().size() == 2);
        REQUIRE(new_df.GetData().at(0).size() == 2);
    }
    SECTION("Initializing with empty vector") {
        std::vector<std::vector<std::string>> v;
        REQUIRE_THROWS_AS(DataFrame(v), std::runtime_error);
    }
}

TEST_CASE("DataFrame ConvertEmptyToInt", "[DF_ConvertEmptyToInt]") {
    SECTION("Converting empty data to ints in column that doesn't exist") {
        std::string filename = "data.csv";
        DataFrame df = DataFrame(filename);
        REQUIRE_THROWS_AS(df.ConvertEmptyToInt("fff"), std::runtime_error);
    }
}

TEST_CASE("DataFrame FillEmpty", "[DF_FillEmpty]") {
    SECTION("Filling empty data in column that doesn't exist") {
        std::string filename = "data.csv";
        DataFrame df = DataFrame(filename);
        REQUIRE_THROWS_AS(df.FillEmpty("fff", "0"), std::runtime_error);
    }
}


TEST_CASE("DataFrame DropColumns", "[DF_DropColumns]") {
    SECTION("Drop columns that do exist") {
        std::string filename = "data.csv";
        DataFrame df = DataFrame(filename);
        std::vector<std::string> cols_to_drop = {"yr", "ht", "num", "ast/tov", "rimmade", "rimmade+rimmiss", "midmade", "midmade+midmiss", 
                "rimmade/(rimmade+rimmiss)", "midmade/(midmade+midmiss)", "dunksmade", "dunksmiss+dunksmade",
                "dunksmade/(dunksmade+dunksmiss)", "pick", "team", "conf", "type", "year", "pid", "player_name", "Unnamed: 65"};    
        df.DropColumns(cols_to_drop);
        REQUIRE(df.GetColNames().size() == 44);
        REQUIRE(df.GetData().size() == 61061);
        REQUIRE(df.GetData().at(0).size() == 44);
    }
    SECTION("Drop column that doesn't exist") {
        std::string filename = "data.csv";
        DataFrame df = DataFrame(filename);
        std::vector<std::string> cols_to_drop = {"yr, fff"};
        REQUIRE_THROWS_AS(df.DropColumns(cols_to_drop), std::runtime_error);
    }
}

TEST_CASE("DataFrame DropRowsWithEmptyData", "[DF_DropRowsWithEmptyData]") {
    std::string filename = "data.csv";
    DataFrame df = DataFrame(filename);
    std::vector<std::string> cols_to_drop = {"yr", "ht", "num", "ast/tov", "rimmade", "rimmade+rimmiss", "midmade", "midmade+midmiss", 
             "rimmade/(rimmade+rimmiss)", "midmade/(midmade+midmiss)", "dunksmade", "dunksmiss+dunksmade",
             "dunksmade/(dunksmade+dunksmiss)", "pick", "team", "conf", "type", "year", "pid", "player_name", "Unnamed: 65"};
    df.DropColumns(cols_to_drop);
    df.FillEmpty("Rec Rank", "0");
    df.DropRowsWithEmptyData();
    REQUIRE(df.GetColNames().size() == 44);
    REQUIRE(df.GetData().size() == 56367);
    REQUIRE(df.GetData().at(0).size() == 44);

}

TEST_CASE("DataFrame GetColumn", "[DF_GetColumn]") {
    SECTION("Get valid column") {
        std::string filename = "data.csv";
        DataFrame df = DataFrame(filename);
        DataFrame pick_df = df.GetColumn("pick");
        REQUIRE(pick_df.GetColNames().size() == 1);
        REQUIRE(pick_df.GetData().size() == 61061);
        REQUIRE(pick_df.GetData().at(0).size() == 1);
        REQUIRE(df.GetColNames().size() == 65);
        REQUIRE(df.GetData().size() == 61061);
        REQUIRE(df.GetData().at(0).size() == 65);
    }
    SECTION("Get invalid column") {
        std::string filename = "data.csv";
        DataFrame df = DataFrame(filename);
        REQUIRE_THROWS_AS(df.GetColumn("irrelevant"), std::runtime_error);
    }
}


TEST_CASE("DataFrame ConvertToNumber", "[DF_ConvertToNumber]") {
    std::string filename = "data.csv";
    DataFrame df = DataFrame(filename);
    std::vector<std::string> cols_to_drop = {"yr", "ht", "num", "ast/tov", "rimmade", "rimmade+rimmiss", "midmade", "midmade+midmiss", 
             "rimmade/(rimmade+rimmiss)", "midmade/(midmade+midmiss)", "dunksmade", "dunksmiss+dunksmade",
             "dunksmade/(dunksmade+dunksmiss)", "pick", "team", "conf", "type", "year", "pid", "player_name", "Unnamed: 65"};
    df.DropColumns(cols_to_drop);
    df.FillEmpty("Rec Rank", "0");
    df.DropRowsWithEmptyData();
    df.ConvertToNumber();
    REQUIRE(df.GetInputs().size() == 56367);
    REQUIRE(df.GetInputs().at(0).size() == 42);
}