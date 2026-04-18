#pragma once

#include <string>
#include <vector>

std::vector<double> load_csv_matrix(const std::string& path, int expected_rows, int expected_cols);
void write_csv_matrix(const std::string& path, const std::vector<double>& values, int rows, int cols);
void write_text_file(const std::string& path, const std::string& text);
std::string join_path(const std::string& left, const std::string& right);
