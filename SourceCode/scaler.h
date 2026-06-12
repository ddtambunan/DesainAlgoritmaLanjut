#ifndef SCALER_H
#define SCALER_H

#include <vector>
using Matrix = std::vector<std::vector<double>>;

class StandardScaler {
private:
    std::vector<double> mean;
    std::vector<double> stddev;

public:
    void fit(const Matrix& X);
    Matrix transform(const Matrix& X) const;
    Matrix fitTransform(const Matrix& X);
};

#endif