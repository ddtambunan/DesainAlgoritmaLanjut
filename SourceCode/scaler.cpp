#include "scaler.h"
#include <cmath>

void StandardScaler::fit(const Matrix& X) {
    int n = X.size();
    int m = X[0].size();

    mean.assign(m, 0.0);
    stddev.assign(m, 0.0);

    for (int j = 0; j < m; j++) {
        for (int i = 0; i < n; i++) {
            mean[j] += X[i][j];
        }
        mean[j] /= n;
    }

    for (int j = 0; j < m; j++) {
        for (int i = 0; i < n; i++) {
            stddev[j] += (X[i][j] - mean[j]) * (X[i][j] - mean[j]);
        }
        stddev[j] = std::sqrt(stddev[j] / n);
        if (stddev[j] == 0.0) stddev[j] = 1.0;
    }
}

Matrix StandardScaler::transform(const Matrix& X) const {
    Matrix hasil = X;
    for (int i = 0; i < (int)X.size(); i++) {
        for (int j = 0; j < (int)X[0].size(); j++) {
            hasil[i][j] = (X[i][j] - mean[j]) / stddev[j];
        }
    }
    return hasil;
}

Matrix StandardScaler::fitTransform(const Matrix& X) {
    fit(X);
    return transform(X);
}