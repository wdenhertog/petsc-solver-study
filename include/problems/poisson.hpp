#pragma once

#include "problem.hpp"

#include <petscdmda.h>

class PoissonProblem : public Problem
{
  public:
    ~PoissonProblem() override;

    std::string name() const override
    {
        return "poisson";
    }
    ProblemKind kind() const override
    {
        return ProblemKind::Linear;
    }

    void assemble_linear(Mat& A, Vec& b, Vec& x) override;

    // Exposed so main.cpp can record it in BenchmarkResult
    PetscInt dofs() const
    {
        return dofs_;
    }

  private:
    DM dm_ = nullptr;
    PetscInt dofs_ = 0;
};
