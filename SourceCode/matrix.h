#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <stdexcept>
#include <cmath>

using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

class MatrixOps {
public:
    static Matrix transpose(const Matrix& A);
    static Matrix multiply(const Matrix& A, const Matrix& B);
    static Vector multiply(const Matrix& A, const Vector& x);
    static Matrix inverse(Matrix A);
    static Matrix identity(int n);
    static Matrix addBiasColumn(const Matrix& X);
};

#endif