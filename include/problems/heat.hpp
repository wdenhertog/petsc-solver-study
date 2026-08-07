#pragma once

#include "problem.hpp"

#include <petscdmda.h>

class HeatProblem : public Problem
{
  public:
    ~HeatProblem() override;

    std::string name() const override
    {
        return "heat";
    }

    ProblemKind kind() const override
    {
        return ProblemKind::Transient;
    }

    void assemble_transient(Mat& M, Mat& K, Vec& x) override;
    PetscReal transient_l2_error(Vec& x, PetscReal time) override;

  private:
    PetscScalar exact_value(PetscReal x, PetscReal y, PetscReal time) const;

    DM dm_ = nullptr;
    PetscReal alpha_ = 1.0;
    PetscInt n_ = 64;
};