#pragma once
#include <petscksp.h>
#include <petscsnes.h>
#include <string>

enum class ProblemKind
{
    Linear,
    Nonlinear,
  Transient
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
    virtual void assemble_transient(Mat& M, Mat& K, Vec& x) {}
    virtual PetscReal transient_l2_error(Vec& x, PetscReal time)
    {
      return -1.0;
    }
};
