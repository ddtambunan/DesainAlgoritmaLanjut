#include "linear_model.h"

LinearModel::LinearModel() : bias(0.0) {}

Vector LinearModel::predict(const Matrix& X) const {
    Vector hasil(X.size(), 0.0);

    for (int i = 0; i < (int)X.size(); i++) {
        hasil[i] = bias;
        for (int j = 0; j < (int)X[i].size(); j++) {
            hasil[i] += X[i][j] * bobot[j];
        }
    }

    return hasil;
}

Vector LinearModel::getBobot() const {
    return bobot;
}

double LinearModel::getBias() const {
    return bias;
}

LinearModel::~LinearModel() {}