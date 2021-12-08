#ifndef DATAFRAME_H
#define DATAFRAME_H

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <random>
#include <limits>
#include <algorithm>

class DataFrame {
    public:
        DataFrame();
        DataFrame(std::string filename);
        DataFrame(std::vector<std::vector<std::string>> v);

        void ReadCSV(std::string filename);
        void ConvertEmptyToInt(std::string column);
        void FillEmpty(std::string column, std::string value);
        void DropColumns(std::vector<std::string> cols);
        void DropRowsWithEmptyData();
        void DropRowsWithColValue(std::string column, std::string value, double frac);
        std::vector<std::vector<std::vector<double>>> GetTrainValidSplit(double frac, DataFrame y_df);
        DataFrame GetColumn(std::string column);
        void ConvertToNumber();
        void Normalize();
        void Normalize(std::vector<double> maxes, std::vector<double> mins);

        std::vector<std::string> GetColNames() { return col_names_; }
        std::vector<std::vector<std::string>> GetData() { return data_; }
        std::vector<std::vector<double>> GetInputs() { return inputs_; }

        std::vector<double> GetColMeans() { return col_means_; }
        std::vector<double> GetColMaxes() { return col_maxes_; }
        std::vector<double> GetColMins() { return col_mins_; }

    private:
        int ColIndexOf(std::vector<std::string> col_names, std::string col);

        std::vector<std::string> col_names_;
        std::vector<std::vector<std::string>> data_;
        std::vector<std::vector<double>> inputs_;

        std::vector<double> col_means_;
        std::vector<double> col_maxes_;
        std::vector<double> col_mins_;
};

#endif