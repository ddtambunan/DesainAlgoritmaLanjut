#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <fstream>

#include "normal_equation.h"
#include "gradient_descent.h"
#include "gd_tuner.h"
#include "metrics.h"
#include "scaler.h"
#include "dataset.h"


#include <windows.h>
#include <psapi.h>

// Fungsi untuk cek memory
double getMemoryUsageMB() {
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        // Mengubah bytes ke Megabytes (MB)
        return static_cast<double>(pmc.WorkingSetSize) / (1024.0 * 1024.0);
    }
    return 0.0;
}

void evaluasiModel(
    const std::string& namaModel,
    LinearModel& model,
    const Matrix& Xtrain,
    const Vector& ytrain,
    const Matrix& Xtest,
    const Vector& ytest,
    double waktuTraining
) {
    Vector predTrain = model.predict(Xtrain);
    Vector predTest = model.predict(Xtest);

    double mseTrain  = Metrics::mse(ytrain, predTrain);
    double rmseTrain = Metrics::rmse(ytrain, predTrain);
    double maeTrain  = Metrics::mae(ytrain, predTrain);
    double r2Train   = Metrics::r2(ytrain, predTrain);

    double mseTest   = Metrics::mse(ytest, predTest);
    double rmseTest  = Metrics::rmse(ytest, predTest);
    double maeTest   = Metrics::mae(ytest, predTest);
    double r2Test    = Metrics::r2(ytest, predTest);

    std::cout << "\n=== " << namaModel << " ===\n";
    std::cout << "Train MSE  : " << mseTrain << "\n";
    std::cout << "Train RMSE : " << rmseTrain << "\n";
    std::cout << "Train MAE  : " << maeTrain << "\n";
    std::cout << "Train R2   : " << r2Train << "\n";
    std::cout << "Test MSE   : " << mseTest << "\n";
    std::cout << "Test RMSE  : " << rmseTest << "\n";
    std::cout << "Test MAE   : " << maeTest << "\n";
    std::cout << "Test R2    : " << r2Test << "\n";
    std::cout << "Waktu train: " << waktuTraining << " s\n";
}

int main() {
    
    // 1. LOAD DATASET
    DataSet data = CSVReader::readCSV("2_data_banyak_features.csv", 12, true);
    // DataSet data = CSVReader::readCSV("1_data_multiple.csv", 2, true);
    // Tambahkan kode cetak dimensi di sini
    if (!data.X.empty()) {
        std::cout << "Dataset berhasil dimuat.\n";
        std::cout << "Jumlah Sampel (Baris) : " << data.X.size() << "\n";
        std::cout << "Jumlah Fitur (Kolom X): " << data.X[0].size() << "\n";
    }

    // 2. SPLIT TRAIN-TEST
    DataSet trainData, testData;
    DataSplitter::trainTestSplit(data, 0.8, trainData, testData);

    // 3. SCALING (FIT HANYA DI TRAIN)
    StandardScaler scaler;
    Matrix XtrainScaled = scaler.fitTransform(trainData.X);
    Matrix XtestScaled = scaler.transform(testData.X);

    Vector ytrain = trainData.y;
    Vector ytest = testData.y;

    // A. NORMAL EQUATION /////////////////////////////////

    
    NormalEquationLR modelNE;
    double waktuNE;
    double memNE = 0.0;
    double memVanillaGD = 0.0;
    double memTunedGD = 0.0;

    try {
        double memSebelumNE = getMemoryUsageMB();
        auto startNE = std::chrono::high_resolution_clock::now();
        modelNE.fit(XtrainScaled, ytrain);
        auto endNE = std::chrono::high_resolution_clock::now();
        double memSesudahNE = getMemoryUsageMB();
        waktuNE = std::chrono::duration<double>(endNE - startNE).count();
        memNE = memSesudahNE - memSebelumNE;
        if (memNE < 0) memNE = 0.0;

        evaluasiModel(
            "Normal Equation",
            modelNE,
            XtrainScaled,
            ytrain,
            XtestScaled,
            ytest,
            waktuNE
        );        

    } catch (const std::exception& e) {std::cout << "\n---------------------------------------------------------------------------\n";
        std::cout << "[ERROR] Normal Equation Gagal pada Data Berfitur Banyak!\n";
        std::cout << "Alasan: " << e.what() << "\n";
        std::cout << "Analisis: Fitur Lag & Rolling Mean menyebabkan Multikolinearitas Sempurna.\n";
        std::cout << "---------------------------------------------------------------------------\n";
    }

    // B. VANILLA GRADIENT DESCENT //////////////////////////
    double memSebelumGD = getMemoryUsageMB();
    auto startGD = std::chrono::high_resolution_clock::now();
    GradientDescentLR modelGD(0.01, 10000, 1e-2);  // baseline default
    modelGD.fit(XtrainScaled, ytrain);
    auto endGD = std::chrono::high_resolution_clock::now();
    double memSesudahGD = getMemoryUsageMB();
    double waktuVanillaGD = std::chrono::duration<double>(endGD - startGD).count();
    memVanillaGD = memSesudahGD - memSebelumGD;
    if (memVanillaGD < 0) memVanillaGD = 0.0;

    evaluasiModel(
        "Vanilla Gradient Descent",
        modelGD,
        XtrainScaled,
        ytrain,
        XtestScaled,
        ytest,
        waktuVanillaGD
    );

    std::cout << "Iterasi aktual (Vanilla GD): " << modelGD.getEpochTerpakai() << "\n";
    std::cout << "Early stopped (Vanilla GD) : " << (modelGD.isEarlyStopped() ? "Ya" : "Tidak") << "\n";

    // C. TUNED GRADIENT DESCENT  ///////////////////////////////
    std::vector<double> kandidatLearningRate = {0.05, 0.01,0.1,};
    std::vector<int> kandidatIter = {5000, 10000,20000};

    double memSebelumTGD = getMemoryUsageMB();
    auto startTGD = std::chrono::high_resolution_clock::now();
    HasilTuningGD hasilGD = GDTuner::cariTerbaik(
        XtrainScaled,
        ytrain,
        kandidatLearningRate,
        kandidatIter,
        1e-2,
        true
    );
    auto endTGD = std::chrono::high_resolution_clock::now();
    double memSesudahTGD = getMemoryUsageMB();
    double waktuTunedGD = std::chrono::duration<double>(endTGD - startTGD).count();
    memTunedGD = memSesudahTGD - memSebelumTGD; 
    if (memTunedGD < 0) memTunedGD = 0.0;

    GradientDescentLR bestGD = hasilGD.modelTerbaik;

    evaluasiModel(
        "Tuned Gradient Descent",
        bestGD,
        XtrainScaled,
        ytrain,
        XtestScaled,
        ytest,
        waktuTunedGD
    );

    std::cout << "\n=== Ringkasan Best Tuned GD ===\n";
    std::cout << "Learning rate terbaik : " << hasilGD.learningRateTerbaik << "\n";
    std::cout << "Max iter terbaik      : " << hasilGD.maxIterTerbaik << "\n";
    std::cout << "Iterasi aktual        : " << hasilGD.iterasiAktual << "\n";
    std::cout << "Early stopped         : " << (hasilGD.earlyStopped ? "Ya" : "Tidak") << "\n";
    std::cout << "Final MSE terbaik     : " << hasilGD.finalMSE << "\n";

    // D. RINGKASAN AKHIR - METRIK EVALUASI /////////////////////////
    double neMSE   = Metrics::mse(ytest, modelNE.predict(XtestScaled));
    double neRMSE  = Metrics::rmse(ytest, modelNE.predict(XtestScaled));
    double neMAE   = Metrics::mae(ytest, modelNE.predict(XtestScaled));
    double neR2    = Metrics::r2(ytest, modelNE.predict(XtestScaled));

    double gdMSE   = Metrics::mse(ytest, modelGD.predict(XtestScaled));
    double gdRMSE  = Metrics::rmse(ytest, modelGD.predict(XtestScaled));
    double gdMAE   = Metrics::mae(ytest, modelGD.predict(XtestScaled));
    double gdR2    = Metrics::r2(ytest, modelGD.predict(XtestScaled));

    double tunedMSE   = Metrics::mse(ytest, bestGD.predict(XtestScaled));
    double tunedRMSE  = Metrics::rmse(ytest, bestGD.predict(XtestScaled));
    double tunedMAE   = Metrics::mae(ytest, bestGD.predict(XtestScaled));
    double tunedR2    = Metrics::r2(ytest, bestGD.predict(XtestScaled));

    // CETAK KE LAYAR KONSOL
    std::cout << "\n=====================================================================================================================\n";
    std::cout << "Ringkasan Perbandingan Algoritma\n";
    std::cout << "=====================================================================================================================\n";
    std::cout << std::left
              << std::setw(22) << "Model"
              << std::setw(12) << "Time(s)"
              << std::setw(14) << "Memory(MB)"  // Tambahan kolom memori di konsol
              << std::setw(10) << "Iter"
              << std::setw(15) << "MSE"
              << std::setw(15) << "RMSE"
              << std::setw(15) << "MAE"
              << std::setw(12) << "R2"
              << "\n";
    std::cout << "---------------------------------------------------------------------------------------------------------------------\n";

    std::cout << std::left
              << std::setw(22) << "Normal Equation"
              << std::setw(12) << waktuNE
              << std::setw(14) << memNE          // Isi data memori NE
              << std::setw(10) << "-"
              << std::setw(15) << neMSE
              << std::setw(15) << neRMSE
              << std::setw(15) << neMAE
              << std::setw(12) << neR2
              << "\n";

    std::cout << std::left
              << std::setw(22) << "Vanilla GD"
              << std::setw(12) << waktuVanillaGD
              << std::setw(14) << memVanillaGD    // Isi data memori Vanilla GD
              << std::setw(10) << modelGD.getEpochTerpakai()
              << std::setw(15) << gdMSE
              << std::setw(15) << gdRMSE
              << std::setw(15) << gdMAE
              << std::setw(12) << gdR2
              << "\n";

    std::cout << std::left
              << std::setw(22) << "Tuned GD"
              << std::setw(12) << waktuTunedGD
              << std::setw(14) << memTunedGD      // Isi data memori Tuned GD
              << std::setw(10) << hasilGD.iterasiAktual
              << std::setw(15) << tunedMSE
              << std::setw(15) << tunedRMSE
              << std::setw(15) << tunedMAE
              << std::setw(12) << tunedR2
              << "\n";

    std::cout << "=====================================================================================================================\n";

    // PROSES EXPORT DATA KE CSV (HANYA HEADER DAN DATA BERSIH)
    std::ofstream fileCSV("hasil_pengujian.csv", std::ios::app);
    if (fileCSV.is_open()) {
        fileCSV.seekp(0, std::ios::end);
        if (fileCSV.tellp() == 0) {
            // Header kolom yang telah diperbarui dengan Memory(MB)
            fileCSV << "Model,Time(s),Memory(MB),Iter,MSE,RMSE,MAE,R2\n";
        }
        
        // Memasukkan baris data secara bersih ke file CSV
        fileCSV << "Normal Equation," << waktuNE << "," << memNE << ",-," << neMSE << "," << neRMSE << "," << neMAE << "," << neR2 << "\n";
        fileCSV << "Vanilla GD," << waktuVanillaGD << "," << memVanillaGD << "," << modelGD.getEpochTerpakai() << "," << gdMSE << "," << gdRMSE << "," << gdMAE << "," << gdR2 << "\n";
        fileCSV << "Tuned GD," << waktuTunedGD << "," << memTunedGD << "," << hasilGD.iterasiAktual << "," << tunedMSE << "," << tunedRMSE << "," << tunedMAE << "," << tunedR2 << "\n";
        
        fileCSV.close();
    }
    return 0;

}