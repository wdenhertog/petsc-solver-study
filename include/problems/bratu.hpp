#pragma once

#include "problem.hpp"
#include <petscdmda.h>

class BratuProblem : public Problem
{
  public:
    ~BratuProblem() override;

    std::string name() const override
    {
        return "bratu";
    }
    ProblemKind kind() const override
    {
        return ProblemKind::Nonlinear;
    }

    void assemble_nonlinear(SNES snes, Vec& x) override;

  private:
    static PetscErrorCode FormFunctionLocal(DMDALocalInfo* info, PetscScalar** u, PetscScalar** f,
                                            void* ctx);
    static PetscErrorCode FormJacobianLocal(DMDALocalInfo* info, PetscScalar** u, Mat jac,
                                            Mat jacpre, void* ctx);

    DM dm_ = nullptr;
    PetscReal lambda_ = 6.0; // classic Bratu parameter; critical/turning value is ~6.808
};
