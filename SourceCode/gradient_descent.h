#ifndef GRADIENT_DESCENT_H
#define GRADIENT_DESCENT_H

#include "linear_model.h"
#include <vector>

class GradientDescentLR : public LinearModel {
private:
    double learningRate;
    int maxIter;
    double tolerance;

    std::vector<double> lossHistory;
    int iterasiAktual;
    bool earlyStopped;

public:
    GradientDescentLR(double lr = 0.01, int nIter = 1000, double tol = 1e-6);

    void fit(const Matrix& X, const Vector& y) override;

    std::vector<double> getLossHistory() const;
    int getEpochTerpakai() const;
    bool isEarlyStopped() const;
    double getLearningRate() const;
    int getMaxIter() const;
    double getTolerance() const;
};

#endif