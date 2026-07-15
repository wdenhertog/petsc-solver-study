#pragma once
#include <petscksp.h>
#include <petscsnes.h>
#include <string>

enum class ProblemKind
{
    Linear,
    Nonlinear,
    VariationalInequality
};

class Problem
{
  public:
    virtual ~Problem() = default;
    virtual std::string name() const = 0;
    virtual ProblemKind kind() const = 0;

    // Only the relevant one gets called by the dispatcher, based on kind()
    virtual void assemble_linear(Mat& A, Vec& b, Vec& x) {}
    virtual void assemble_nonlinear(SNES snes, Vec& x) {}
    virtual void assemble_vi(SNES snes, Vec& x, Vec& lower, Vec& upper) {}
};
