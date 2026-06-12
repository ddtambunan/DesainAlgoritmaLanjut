#ifndef GD_TUNER_H
#define GD_TUNER_H

#include <vector>
#include <limits>
#include "gradient_descent.h"

struct HasilTuningGD {
    double learningRateTerbaik;
    int maxIterTerbaik;
    double tolerance;
    int iterasiAktual;
    bool earlyStopped;
    double finalMSE;
    std::vector<double> bestCosts;
    GradientDescentLR modelTerbaik;

    HasilTuningGD()
        : learningRateTerbaik(0.0),
          maxIterTerbaik(0),
          tolerance(1e-6),
          iterasiAktual(0),
          earlyStopped(false),
          finalMSE(std::numeric_limits<double>::max()),
          modelTerbaik(0.01, 1000, 1e-6) {}
};

class GDTuner {
public:
    static HasilTuningGD cariTerbaik(
        const Matrix& Xtrain,
        const Vector& ytrain,
        const std::vector<double>& lrCandidates,
        const std::vector<int>& nIterCandidates,
        double tol = 1e-6,
        bool verbose = true
    );
};

#endif