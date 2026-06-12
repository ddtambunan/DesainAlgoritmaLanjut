#include "gradient_descent.h"
#include <stdexcept>
#include <cmath>

GradientDescentLR::GradientDescentLR(double lr, int nIter, double tol)
    : learningRate(lr), maxIter(nIter), tolerance(tol),
      iterasiAktual(0), earlyStopped(false) {}

void GradientDescentLR::fit(const Matrix& X, const Vector& y) {
    int nSamples = static_cast<int>(X.size());
    if (nSamples == 0) {
        throw std::invalid_argument("X kosong.");
    }

    int nFeatures = static_cast<int>(X[0].size());
    if (static_cast<int>(y.size()) != nSamples) {
        throw std::invalid_argument("Ukuran X dan y tidak cocok.");
    }

    bobot.assign(nFeatures, 0.0);
    bias = 0.0;

    lossHistory.clear();
    iterasiAktual = 0;
    earlyStopped = false;

    for (int iter = 0; iter < maxIter && earlyStopped!=true; ++iter)  {
        
        Vector yPred(nSamples, 0.0);

        for (int i = 0; i < nSamples; ++i) {
            double pred = bias;
            for (int j = 0; j < nFeatures; ++j) {
                pred += X[i][j] * bobot[j];
            }
            yPred[i] = pred;
        }

        Vector gradW(nFeatures, 0.0);
        double gradB = 0.0;
        double mse = 0.0;

        for (int i = 0; i < nSamples; ++i) {
            double err = yPred[i] - y[i];
            mse += err * err;
            gradB += err;
            for (int j = 0; j < nFeatures; ++j) {
                gradW[j] += err * X[i][j];
            }
        }

        mse /= static_cast<double>(nSamples);
        lossHistory.push_back(mse);

        for (int j = 0; j < nFeatures; ++j) {
            gradW[j] /= static_cast<double>(nSamples);
            bobot[j] -= learningRate * gradW[j];
        }

        gradB /= static_cast<double>(nSamples);
        bias -= learningRate * gradB;

        iterasiAktual = iter + 1;

        if (iter > 0) {
            double delta = std::abs(lossHistory[iter] - lossHistory[iter - 1]);
            if (delta < tolerance) {
                earlyStopped = true;
                break;
            }
        }
    }
}

std::vector<double> GradientDescentLR::getLossHistory() const {
    return lossHistory;
}

int GradientDescentLR::getEpochTerpakai() const {
    return iterasiAktual;
}

bool GradientDescentLR::isEarlyStopped() const {
    return earlyStopped;
}

double GradientDescentLR::getLearningRate() const {
    return learningRate;
}

int GradientDescentLR::getMaxIter() const {
    return maxIter;
}

double GradientDescentLR::getTolerance() const {
    return tolerance;
}