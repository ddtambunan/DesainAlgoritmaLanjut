#ifndef DATASET_H
#define DATASET_H

#include <vector>
#include <string>

struct DataSet {
    std::vector<std::vector<double>> X;
    std::vector<double> y;
};

class CSVReader {
public:
    static DataSet readCSV(const std::string& filename, int targetCol, bool hasHeader = true);
};

class DataSplitter {
public:
    static void trainTestSplit(const DataSet& data, double rasioTrain, DataSet& train, DataSet& test);
};

#endif