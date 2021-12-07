#include "dataframe.hpp"
#include <stdexcept>

DataFrame::DataFrame() {}

DataFrame::DataFrame(std::string filename) {
    ReadCSV(filename);
}

DataFrame::DataFrame(std::vector<std::vector<std::string>> v) {
    // Initializing DataFrame from the 2D vector of another DataFrame
    if (v.empty()) {
        throw std::runtime_error("Invalid data format");
    }
    col_names_ = v.at(0);
    for (size_t i = 1; i < v.size(); i++) {
        data_.push_back(v.at(i));
    }
}

void DataFrame::ReadCSV(std::string filename) {
    std::ifstream ifs {filename};
    std::string line;
    bool firstLine = true;
    size_t num_cols = -1;
    if (!ifs.good()) {
        throw std::runtime_error("Invalid file name");
    }
    while(std::getline(ifs, line, '\n')) {
        std::stringstream line_stream(line);
        std::string column;
        std::vector<std::string> data;
        int col_count = 1;
        while(std::getline(line_stream, column, ',')) {
            // First row of CSV contains column names. Rest of CSV contains actual data
            if (firstLine) {
                // Can't have columns with no name
                if (column == "" || column == "\r") {
                    col_names_.push_back("Unnamed: " + std::to_string(col_count));
                } else {
                    col_names_.push_back(column);
                }
            } else {
                data.push_back(column);
            }
            col_count++;
        }
        if (firstLine) {
            firstLine = false;
            num_cols = col_names_.size();
        } else {
            // There is a comma in the player's name
            if (data.size() != num_cols) {
                data.at(0) += "," + data.at(1);
                data.erase(data.begin() + 1);
            }
            data_.push_back(data);
        }
    }
}

void DataFrame::ConvertEmptyToInt(std::string column) {
    int index = ColIndexOf(col_names_, column);
    if (index == -1) {
        throw std::runtime_error("Invalid column");
    }

    // Go over every row in column. If row has empty data make it 0, otherwise make it 1
    for (size_t row = 0; row < data_.size(); row++) {
        if (data_.at(row).at(index) == "") {
            data_.at(row).at(index) = "0";
        } else {
            data_.at(row).at(index) = "1";
        }
    }
}

void DataFrame::FillEmpty(std::string column, std::string value) {
    int index = ColIndexOf(col_names_, column);
    if (index == -1) {
        throw std::runtime_error("Invalid column");
    }

    // Go over every row in column. If row has empty data set it equal to the value
    for (size_t row = 0; row < data_.size(); row++) {
        if (data_.at(row).at(index) == "") {
            data_.at(row).at(index) = value;
        }
    }
}

void DataFrame::DropColumns(std::vector<std::string> cols) {
    // Drops all columns in the vector passed from the database
    std::vector<int> indices;
    for (size_t i = 0; i < cols.size(); i++) {
        int index = ColIndexOf(col_names_, cols.at(i));
        if (index == -1) {
            throw std::runtime_error("Invalid column");
        }
        indices.push_back(index);
    }

    // Sort indices in descending order so deleting them dfrom the dataframe 
    // doesn't alter any other indices
    std::sort(indices.begin(), indices.end(), std::greater<>());
    for (size_t i = 0; i < indices.size(); i++) {
        col_names_.erase(col_names_.begin() + indices.at(i));
    }

    for (size_t row = 0; row < data_.size(); row++) {
        for (size_t i = 0; i < indices.size(); i++) {
            data_.at(row).erase(data_.at(row).begin() + indices.at(i));
        }
    }
}

void DataFrame::DropRowsWithEmptyData() {
    // Go over every row. If it has an empty column drop the row
    for (size_t row = data_.size() - 1; row >= 0 ; row--) {
        for (size_t col = 0; col < data_.at(row).size(); col++) {
            if (data_.at(row).at(col) == "") {
                data_.erase(data_.begin() + row);
                break;
            }
        }
        if (row == 0) {
            break;
        }
    }
}

DataFrame DataFrame::GetColumn(std::string column) {
    // Returns relevant column as dataframe
    int index = ColIndexOf(col_names_, column);
    if (index == -1) {
        throw std::runtime_error("Invalid column");
    }

    std::vector<std::string> new_col_names = {column};
    std::vector<std::vector<std::string>> new_data;
    new_data.push_back(new_col_names);
    for (size_t row = 0; row < data_.size(); row++) {
        std::vector<std::string> v = {data_.at(row).at(index)};
        new_data.push_back(v);
    }
    DataFrame new_df = DataFrame(new_data);
    return new_df;
}

void DataFrame::ConvertToNumber() {
    for (size_t i = 0; i < data_.size(); i++) {
        std::vector<double> v;
        for (size_t j = 0; j < data_.at(i).size(); j++) {
            v.push_back(std::stod(data_.at(i).at(j)));
        }
        inputs_.push_back(v);
    }
}

void DataFrame::Normalize() {
    // For every column find the min and max value
    // Then for every row in the column, normalize it by
    // subtracting the min and dividing by the max - min
    std::vector<double> maxes;
    std::vector<double> mins;
    for (size_t col = 0; col < col_names_.size(); col++) {
        double max = inputs_.at(0).at(col);
        double min = inputs_.at(0).at(col);
        for (size_t row = 1; row < inputs_.size(); row++) {
            if (inputs_.at(row).at(col) > max) {
                max = inputs_.at(row).at(col);
            }
            if (inputs_.at(row).at(col) < min) {
                min = inputs_.at(row).at(col);
            }
        }
        maxes.push_back(max);
        mins.push_back(min);
    }

    for (size_t col = 0; col < col_names_.size(); col++) {
        for (size_t row = 0; row < inputs_.size(); row++) {
            inputs_.at(row).at(col) = (inputs_.at(row).at(col) - mins.at(col)) / (maxes.at(col) - mins.at(col));
        }
    }
}

int DataFrame::ColIndexOf(std::vector<std::string> col_names, std::string col) {
    // Returns index of passed column name
    // -1 if not present
    for (size_t i = 0; i < col_names.size(); i++) {
        if (col == col_names.at(i)) {
            return i;
        }
    }
    return -1;
}