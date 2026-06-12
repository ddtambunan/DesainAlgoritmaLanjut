#include "metrics.h"
#include <cmath>

double Metrics::mse(const Vector& yTrue, const Vector& yPred) {
    double jumlah = 0.0;
    int n = yTrue.size();
    for (int i = 0; i < n; i++) {
        double e = yTrue[i] - yPred[i];
        jumlah += e * e;
    }
    return jumlah / n;
}

double Metrics::rmse(const Vector& yTrue, const Vector& yPred) {
    return std::sqrt(mse(yTrue, yPred));
}

double Metrics::mae(const Vector& yTrue, const Vector& yPred) {
    double jumlah = 0.0;
    int n = yTrue.size();
    for (int i = 0; i < n; i++) {
        jumlah += std::fabs(yTrue[i] - yPred[i]);
    }
    return jumlah / n;
}

double Metrics::r2(const Vector& yTrue, const Vector& yPred) {
    double mean = 0.0;
    for (double v : yTrue) mean += v;
    mean /= yTrue.size();

    double ssRes = 0.0, ssTot = 0.0;
    for (int i = 0; i < (int)yTrue.size(); i++) {
        ssRes += (yTrue[i] - yPred[i]) * (yTrue[i] - yPred[i]);
        ssTot += (yTrue[i] - mean) * (yTrue[i] - mean);
    }
    return 1.0 - (ssRes / ssTot);
}