#include "problems/heat.hpp"

#include <cmath>

HeatProblem::~HeatProblem()
{
    if (dm_)
        DMDestroy(&dm_);
}

void HeatProblem::assemble_transient(Mat& M, Mat& K, Vec& x)
{
    PetscOptionsGetInt(nullptr, nullptr, "-n", &n_, nullptr);
    PetscOptionsGetReal(nullptr, nullptr, "-alpha", &alpha_, nullptr);

    DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, n_, n_,
                 PETSC_DECIDE, PETSC_DECIDE, 1, 1, nullptr, nullptr, &dm_);
    DMSetFromOptions(dm_);
    DMSetUp(dm_);

    DMCreateMatrix(dm_, &M);
    DMCreateMatrix(dm_, &K);
    DMCreateGlobalVector(dm_, &x);
    VecSet(x, 0.0);

    PetscInt xs, ys, xm, ym;
    DMDAGetCorners(dm_, &xs, &ys, nullptr, &xm, &ym, nullptr);

    const PetscReal h = 1.0 / (n_ - 1);
    PetscScalar** xarray;
    DMDAVecGetArray(dm_, x, &xarray);

    for (PetscInt j = ys; j < ys + ym; ++j)
    {
        for (PetscInt i = xs; i < xs + xm; ++i)
        {
            MatStencil row{};
            row.i = i;
            row.j = j;

            const bool boundary = (i == 0 || i == n_ - 1 || j == 0 || j == n_ - 1);
            const PetscScalar one = 1.0;
            MatSetValuesStencil(M, 1, &row, 1, &row, &one, INSERT_VALUES);

            if (boundary)
            {
                MatSetValuesStencil(K, 1, &row, 1, &row, &one, INSERT_VALUES);
                xarray[j][i] = 0.0;
                continue;
            }

            MatStencil col[5];
            PetscScalar val[5];

            col[0] = {0, j, i, 0};
            val[0] = 4.0 * alpha_ / (h * h);
            col[1] = {0, j, i - 1, 0};
            val[1] = -alpha_ / (h * h);
            col[2] = {0, j, i + 1, 0};
            val[2] = -alpha_ / (h * h);
            col[3] = {0, j - 1, i, 0};
            val[3] = -alpha_ / (h * h);
            col[4] = {0, j + 1, i, 0};
            val[4] = -alpha_ / (h * h);

            MatSetValuesStencil(K, 1, &row, 5, col, val, INSERT_VALUES);
            xarray[j][i] = exact_value(i * h, j * h, 0.0);
        }
    }

    DMDAVecRestoreArray(dm_, x, &xarray);

    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(K, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(K, MAT_FINAL_ASSEMBLY);
}

PetscReal HeatProblem::transient_l2_error(Vec& x, PetscReal time)
{
    const PetscReal h = 1.0 / (n_ - 1);
    PetscInt xs, ys, xm, ym;
    DMDAGetCorners(dm_, &xs, &ys, nullptr, &xm, &ym, nullptr);

    const PetscScalar** xarray;
    DMDAVecGetArrayRead(dm_, x, &xarray);

    PetscReal local_sq = 0.0;
    for (PetscInt j = ys; j < ys + ym; ++j)
    {
        for (PetscInt i = xs; i < xs + xm; ++i)
        {
            const PetscReal diff = PetscRealPart(xarray[j][i] - exact_value(i * h, j * h, time));
            local_sq += h * h * diff * diff;
        }
    }

    DMDAVecRestoreArrayRead(dm_, x, &xarray);

    PetscReal global_sq = 0.0;
    MPI_Allreduce(&local_sq, &global_sq, 1, MPIU_REAL, MPIU_SUM, PETSC_COMM_WORLD);
    return PetscSqrtReal(global_sq);
}

PetscScalar HeatProblem::exact_value(PetscReal x, PetscReal y, PetscReal time) const
{
    return std::sin(PETSC_PI * x) * std::sin(PETSC_PI * y)
           * std::exp(-2.0 * alpha_ * PETSC_PI * PETSC_PI * time);
}