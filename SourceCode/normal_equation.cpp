#include "normal_equation.h"
#include "matrix.h"

void NormalEquationLR::fit(const Matrix& X, const Vector& y) {
    Matrix Xb = MatrixOps::addBiasColumn(X);
    Matrix Xt = MatrixOps::transpose(Xb);
    Matrix XtX = MatrixOps::multiply(Xt, Xb);

    // Penambahan Regularisasi Ridge (Ridge Regression / L2)
    // alpha adalah konstanta kecil (misal 1e-4) untuk memaksa matriks menjadi invertible
    double alpha = 1e-4; 
    int dim = XtX.size();
    for (int i = 0; i < dim; ++i) {
        XtX[i][i] += alpha; // penambahan nilai kecil di garis diagonal matriks
    }
    
    Matrix XtX_inv = MatrixOps::inverse(XtX);
    Vector Xty = MatrixOps::multiply(Xt, y);
    Vector theta = MatrixOps::multiply(XtX_inv, Xty);

    bias = theta[0];
    bobot.assign(theta.begin() + 1, theta.end());
}