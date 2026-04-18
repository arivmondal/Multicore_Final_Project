#include "io.hpp"

#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

namespace {

bool is_separator(char ch) {
    return ch == '/' || ch == '\\';
}

int make_dir(const std::string& path) {
#ifdef _WIN32
    return _mkdir(path.c_str());
#else
    return mkdir(path.c_str(), 0755);
#endif
}

std::string parent_path(const std::string& path) {
    const std::size_t pos = path.find_last_of("/\\");
    return pos == std::string::npos ? std::string() : path.substr(0, pos);
}

void ensure_directory(const std::string& path) {
    if (path.empty()) {
        return;
    }

    std::string current;
    std::size_t start = 0;

    if (path.size() >= 2 && path[1] == ':') {
        current = path.substr(0, 2);
        start = 2;
    } else if (is_separator(path[0])) {
        current.assign(1, path[0]);
        start = 1;
    }

    for (std::size_t i = start; i <= path.size(); ++i) {
        if (i != path.size() && !is_separator(path[i])) {
            continue;
        }

        const std::string piece = path.substr(start, i - start);
        if (!piece.empty()) {
            if (!current.empty() && !is_separator(current.back())) {
                current.push_back('/');
            }
            current += piece;
            if (make_dir(current) != 0 && errno != EEXIST) {
                throw std::runtime_error("Failed to create directory: " + current);
            }
        }

        while (i + 1 < path.size() && is_separator(path[i + 1])) {
            ++i;
        }
        start = i + 1;
    }
}

void ensure_parent_directory(const std::string& path) {
    ensure_directory(parent_path(path));
}

std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream stream(line);
    std::string item;
    while (std::getline(stream, item, ',')) {
        fields.push_back(item);
    }
    if (!line.empty() && line.back() == ',') {
        fields.push_back(std::string());
    }
    return fields;
}

}  // namespace

std::vector<double> load_csv_matrix(const std::string& path, int expected_rows, int expected_cols) {
    std::ifstream stream(path.c_str());
    if (!stream) {
        throw std::runtime_error("Failed to open CSV: " + path);
    }

    std::vector<double> values;
    std::string line;
    int rows = 0;
    int cols = -1;

    while (std::getline(stream, line)) {
        if (line.empty()) {
            continue;
        }
        const std::vector<std::string> fields = split_csv_line(line);
        if (cols < 0) {
            cols = static_cast<int>(fields.size());
        }
        if (static_cast<int>(fields.size()) != cols) {
            throw std::runtime_error("Inconsistent column count in CSV: " + path);
        }
        for (std::size_t i = 0; i < fields.size(); ++i) {
            values.push_back(std::stod(fields[i]));
        }
        ++rows;
    }

    if (rows != expected_rows || cols != expected_cols) {
        std::ostringstream oss;
        oss << "Expected " << expected_rows << "x" << expected_cols << " kernel but found "
            << rows << "x" << cols << " in " << path;
        throw std::runtime_error(oss.str());
    }

    return values;
}

void write_csv_matrix(const std::string& path, const std::vector<double>& values, int rows, int cols) {
    if (static_cast<int>(values.size()) != rows * cols) {
        throw std::runtime_error("CSV shape does not match value count for " + path);
    }

    ensure_parent_directory(path);

    std::ofstream stream(path.c_str());
    if (!stream) {
        throw std::runtime_error("Failed to open file for writing: " + path);
    }

    stream << std::setprecision(17);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            if (col > 0) {
                stream << ',';
            }
            stream << values[static_cast<std::size_t>(row) * cols + col];
        }
        stream << '\n';
    }
}

void write_text_file(const std::string& path, const std::string& text) {
    ensure_parent_directory(path);

    std::ofstream stream(path.c_str());
    if (!stream) {
        throw std::runtime_error("Failed to open file for writing: " + path);
    }

    stream << text;
}

std::string join_path(const std::string& left, const std::string& right) {
    if (left.empty()) {
        return right;
    }
    if (right.empty()) {
        return left;
    }
    if (is_separator(left.back())) {
        return left + right;
    }
    return left + "/" + right;
}
