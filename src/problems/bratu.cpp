#include "problems/bratu.hpp"

BratuProblem::~BratuProblem()
{
    if (dm_)
        DMDestroy(&dm_);
}

void BratuProblem::assemble_nonlinear(SNES snes, Vec& x)
{
    PetscInt n = 64;
    PetscOptionsGetInt(nullptr, nullptr, "-n", &n, nullptr);
    PetscOptionsGetReal(nullptr, nullptr, "-lambda", &lambda_, nullptr);

    DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, n, n,
                 PETSC_DECIDE, PETSC_DECIDE, 1, 1, nullptr, nullptr, &dm_);
    DMSetFromOptions(dm_);
    DMSetUp(dm_);

    SNESSetDM(snes, dm_);

    // These attach the local-array-style callbacks directly to the DM;
    // SNES queries the DM for F/J automatically — no SNESSetFunction/SNESSetJacobian needed.
    DMDASNESSetFunctionLocal(dm_, INSERT_VALUES,
                             reinterpret_cast<DMDASNESFunctionFn*>(FormFunctionLocal), this);
    DMDASNESSetJacobianLocal(dm_, reinterpret_cast<DMDASNESJacobianFn*>(FormJacobianLocal), this);

    DMCreateGlobalVector(dm_, &x);
    VecSet(x, 0.0); // zero initial guess; Newton moves via the forcing from lambda*exp(u)
}

PetscErrorCode BratuProblem::FormFunctionLocal(DMDALocalInfo* info, PetscScalar** u,
                                               PetscScalar** f, void* ctx)
{
    auto* self = static_cast<BratuProblem*>(ctx);
    PetscReal lambda = self->lambda_;
    PetscReal h = 1.0 / (info->mx - 1);
    PetscReal h2 = h * h;

    for (PetscInt j = info->ys; j < info->ys + info->ym; ++j)
    {
        for (PetscInt i = info->xs; i < info->xs + info->xm; ++i)
        {
            if (i == 0 || j == 0 || i == info->mx - 1 || j == info->my - 1)
            {
                f[j][i] = u[j][i]; // Dirichlet: u = 0 on boundary
            }
            else
            {
                PetscScalar lap =
                    (4.0 * u[j][i] - u[j][i - 1] - u[j][i + 1] - u[j - 1][i] - u[j + 1][i]) / h2;
                f[j][i] = lap - lambda * PetscExpScalar(u[j][i]);
            }
        }
    }
    return 0;
}

PetscErrorCode BratuProblem::FormJacobianLocal(DMDALocalInfo* info, PetscScalar** u, Mat jac,
                                               Mat jacpre, void* ctx)
{
    auto* self = static_cast<BratuProblem*>(ctx);
    PetscReal lambda = self->lambda_;
    PetscReal h = 1.0 / (info->mx - 1);
    PetscReal h2 = h * h;

    for (PetscInt j = info->ys; j < info->ys + info->ym; ++j)
    {
        for (PetscInt i = info->xs; i < info->xs + info->xm; ++i)
        {
            MatStencil row{};
            row.i = i;
            row.j = j;

            if (i == 0 || j == 0 || i == info->mx - 1 || j == info->my - 1)
            {
                PetscScalar one = 1.0;
                MatSetValuesStencil(jacpre, 1, &row, 1, &row, &one, INSERT_VALUES);
            }
            else
            {
                MatStencil col[5];
                PetscScalar val[5];
                col[0] = {0, j, i, 0};
                val[0] = 4.0 / h2 - lambda * PetscExpScalar(u[j][i]);
                col[1] = {0, j, i - 1, 0};
                val[1] = -1.0 / h2;
                col[2] = {0, j, i + 1, 0};
                val[2] = -1.0 / h2;
                col[3] = {0, j - 1, i, 0};
                val[3] = -1.0 / h2;
                col[4] = {0, j + 1, i, 0};
                val[4] = -1.0 / h2;
                MatSetValuesStencil(jacpre, 1, &row, 5, col, val, INSERT_VALUES);
            }
        }
    }

    MatAssemblyBegin(jacpre, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(jacpre, MAT_FINAL_ASSEMBLY);
    if (jac != jacpre)
    {
        MatAssemblyBegin(jac, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(jac, MAT_FINAL_ASSEMBLY);
    }
    return 0;
}
