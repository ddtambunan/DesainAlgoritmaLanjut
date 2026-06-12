#include "dataset.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream> // Tambahkan ini untuk std::cout

DataSet CSVReader::readCSV(const std::string& filename, int targetCol, bool hasHeader) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Gagal membuka file: " + filename);
    }

    DataSet data;
    std::string line;

    if (hasHeader) {
        std::getline(file, line);
    }

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string cell;
        std::vector<double> row;
        int col = 0;
        double yValue = 0.0;

        while (std::getline(ss, cell, ',')) {
            double value = std::stod(cell);

            if (col == targetCol) {
                yValue = value;
            } else {
                row.push_back(value);
            }
            col++;
        }

        data.X.push_back(row);
        data.y.push_back(yValue);
    }

    // === TAMBAHKAN KODE INI SEBELUM RETURN ===
    if (!data.X.empty()) {
        std::cout << "\n=========================================\n";
        std::cout << "Berhasil memuat file: " << filename << "\n";
        std::cout << "Dimensi Matriks Fitur (X) : " << data.X.size() << " baris x " << data.X[0].size() << " kolom\n";
        std::cout << "Dimensi Vektor Target (y) : " << data.y.size() << " elemen\n";
        std::cout << "=========================================\n\n";
    } else {
        std::cout << "Peringatan: Dataset kosong atau tidak memiliki baris data.\n";
    }
    // ========================================

    return data;
}

void DataSplitter::trainTestSplit(const DataSet& data, double rasioTrain, DataSet& train, DataSet& test) {
    int n = data.X.size();
    int batas = static_cast<int>(n * rasioTrain);

    for (int i = 0; i < n; i++) {
        if (i < batas) {
            train.X.push_back(data.X[i]);
            train.y.push_back(data.y[i]);
        } else {
            test.X.push_back(data.X[i]);
            test.y.push_back(data.y[i]);
        }
    }
}