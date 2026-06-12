#ifndef LINEAR_MODEL_H
#define LINEAR_MODEL_H

#include <vector>
using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

class LinearModel {
protected:
    Vector bobot;
    double bias;

public:
    LinearModel();
    virtual void fit(const Matrix& X, const Vector& y) = 0;
    virtual Vector predict(const Matrix& X) const;
    Vector getBobot() const;
    double getBias() const;
    virtual ~LinearModel();
};

#endif