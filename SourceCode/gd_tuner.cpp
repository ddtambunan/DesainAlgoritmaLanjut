#include "gd_tuner.h"
#include <iostream>
#include <iomanip>

HasilTuningGD GDTuner::cariTerbaik(
    const Matrix& Xtrain,
    const Vector& ytrain,
    const std::vector<double>& lrCandidates,
    const std::vector<int>& nIterCandidates,
    double tol,
    bool verbose
) {
    HasilTuningGD hasil;
    hasil.tolerance = tol;
    hasil.earlyStopped = false;

    for (double lr : lrCandidates) {

        for (int nIter : nIterCandidates) {

            GradientDescentLR model(lr, nIter, tol);
            model.fit(Xtrain, ytrain);

            std::vector<double> costs = model.getLossHistory();
            if (costs.empty()) {
                continue;
            }

            double finalMSE = costs.back();

            if (verbose) {
                std::cout << std::fixed << std::setprecision(6)
                          << "lr=" << lr
                          << ", n_iter=" << nIter
                          << " -> final MSE=" << finalMSE
                          << ", iterasi_aktual=" << model.getEpochTerpakai()
                          << ", early_stop=" << (model.isEarlyStopped() ? "Ya" : "Tidak")
                          << "\n";
            }

            if (finalMSE < hasil.finalMSE) {
                hasil.learningRateTerbaik = lr;
                hasil.maxIterTerbaik = nIter;
                hasil.iterasiAktual = model.getEpochTerpakai();
                hasil.earlyStopped = model.isEarlyStopped();
                hasil.finalMSE = finalMSE;
                hasil.bestCosts = costs;
                hasil.modelTerbaik = model;
            }
        }
    }

    if (verbose) {
        std::cout << "\n=== Best Hyperparameter GD ===\n";
        std::cout << "Learning rate  : " << hasil.learningRateTerbaik << "\n";
        std::cout << "Max iter       : " << hasil.maxIterTerbaik << "\n";
        std::cout << "Tolerance      : " << hasil.tolerance << "\n";
        std::cout << "Iterasi aktual : " << hasil.iterasiAktual << "\n";
        std::cout << "Early stopped  : " << (hasil.earlyStopped ? "Ya" : "Tidak") << "\n";
        std::cout << "Best final MSE : " << hasil.finalMSE << "\n";
    }

    return hasil;
}