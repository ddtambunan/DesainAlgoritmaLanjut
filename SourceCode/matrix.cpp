#include "matrix.h"

Matrix MatrixOps::transpose(const Matrix& A) {
    int baris = A.size();
    int kolom = A[0].size();
    Matrix hasil(kolom, Vector(baris));

    for (int i = 0; i < baris; i++) {
        for (int j = 0; j < kolom; j++) {
            hasil[j][i] = A[i][j];
        }
    }
    return hasil;
}

Matrix MatrixOps::multiply(const Matrix& A, const Matrix& B) {
    int barisA = A.size();
    int kolomA = A[0].size();
    int barisB = B.size();
    int kolomB = B[0].size();

    if (kolomA != barisB) {
        throw std::runtime_error("Ukuran matriks tidak cocok untuk perkalian.");
    }

    Matrix hasil(barisA, Vector(kolomB, 0.0));

    for (int i = 0; i < barisA; i++) {
        for (int j = 0; j < kolomB; j++) {
            for (int k = 0; k < kolomA; k++) {
                hasil[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return hasil;
}

Vector MatrixOps::multiply(const Matrix& A, const Vector& x) {
    int baris = A.size();
    int kolom = A[0].size();

    if (kolom != (int)x.size()) {
        throw std::runtime_error("Ukuran matriks dan vektor tidak cocok.");
    }

    Vector hasil(baris, 0.0);

    for (int i = 0; i < baris; i++) {
        for (int j = 0; j < kolom; j++) {
            hasil[i] += A[i][j] * x[j];
        }
    }

    return hasil;
}

Matrix MatrixOps::identity(int n) {
    Matrix I(n, Vector(n, 0.0));
    for (int i = 0; i < n; i++) {
        I[i][i] = 1.0;
    }
    return I;
}

Matrix MatrixOps::inverse(Matrix A) {
    int n = A.size();
    Matrix I = identity(n);

    for (int i = 0; i < n; i++) {
        double pivot = A[i][i];

        if (std::fabs(pivot) < 1e-12) {
            int swapRow = -1;
            for (int r = i + 1; r < n; r++) {
                if (std::fabs(A[r][i]) > 1e-12) {
                    swapRow = r;
                    break;
                }
            }
            if (swapRow == -1) {
                throw std::runtime_error("Matriks singular, tidak dapat diinvers.");
            }
            std::swap(A[i], A[swapRow]);
            std::swap(I[i], I[swapRow]);
            pivot = A[i][i];
        }

        for (int j = 0; j < n; j++) {
            A[i][j] /= pivot;
            I[i][j] /= pivot;
        }

        for (int r = 0; r < n; r++) {
            if (r != i) {
                double faktor = A[r][i];
                for (int c = 0; c < n; c++) {
                    A[r][c] -= faktor * A[i][c];
                    I[r][c] -= faktor * I[i][c];
                }
            }
        }
    }

    return I;
}

Matrix MatrixOps::addBiasColumn(const Matrix& X) {
    Matrix hasil = X;
    for (auto& row : hasil) {
        row.insert(row.begin(), 1.0);
    }
    return hasil;
}