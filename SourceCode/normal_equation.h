#ifndef NORMAL_EQUATION_H
#define NORMAL_EQUATION_H

#include "linear_model.h"

class NormalEquationLR : public LinearModel {
public:
    void fit(const Matrix& X, const Vector& y) override;
};

#endif