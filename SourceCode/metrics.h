#ifndef METRICS_H
#define METRICS_H

#include <vector>
using Vector = std::vector<double>;

class Metrics {
public:
    static double mse(const Vector& yTrue, const Vector& yPred);
    static double rmse(const Vector& yTrue, const Vector& yPred);
    static double mae(const Vector& yTrue, const Vector& yPred);
    static double r2(const Vector& yTrue, const Vector& yPred);
};

#endif