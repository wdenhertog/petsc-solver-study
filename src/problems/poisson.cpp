#include "problems/poisson.hpp"

#include <cmath>

namespace
{
double forcing(double x, double y)
{
    double dx = x - 0.3, dy = y - 0.7;
    return std::exp(-500.0 * (dx * dx + dy * dy)); // localized Gaussian bump
}
} // namespace

PoissonProblem::~PoissonProblem()
{
    if (dm_)
        DMDestroy(&dm_);
}

void PoissonProblem::assemble_linear(Mat& A, Vec& b, Vec& x)
{
    PetscInt n = 64;
    PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL);

    DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, n,
                 n,                                // global grid points per dimension
                 PETSC_DECIDE, PETSC_DECIDE, 1, 1, // dof per node, stencil width
                 NULL, NULL, &dm_);
    DMSetFromOptions(dm_);
    DMSetUp(dm_);

    // DMCreateMatrix gives correct nonzero preallocation for the stencil
    // automatically — this is the main win over manual MatCreate here.
    DMCreateMatrix(dm_, &A);
    DMCreateGlobalVector(dm_, &x);
    VecDuplicate(x, &b);
    VecSet(x, 0.0);

    PetscInt xs, ys, xm, ym;
    DMDAGetCorners(dm_, &xs, &ys, NULL, &xm, &ym, NULL);

    double h = 1.0 / (n - 1);

    PetscScalar** barray;
    DMDAVecGetArray(dm_, b, &barray);

    for (PetscInt j = ys; j < ys + ym; ++j)
    {
        for (PetscInt i = xs; i < xs + xm; ++i)
        {
            MatStencil row{};
            row.i = i;
            row.j = j;

            bool boundary = (i == 0 || i == n - 1 || j == 0 || j == n - 1);

            if (boundary)
            {
                PetscScalar one = 1.0;
                MatSetValuesStencil(A, 1, &row, 1, &row, &one, INSERT_VALUES);
                barray[j][i] = 0.0;
                continue;
            }

            MatStencil col[5];
            PetscScalar val[5];

            col[0] = {0, j, i, 0};
            val[0] = 4.0 / (h * h);
            col[1] = {0, j, i - 1, 0};
            val[1] = -1.0 / (h * h);
            col[2] = {0, j, i + 1, 0};
            val[2] = -1.0 / (h * h);
            col[3] = {0, j - 1, i, 0};
            val[3] = -1.0 / (h * h);
            col[4] = {0, j + 1, i, 0};
            val[4] = -1.0 / (h * h);

            MatSetValuesStencil(A, 1, &row, 5, col, val, INSERT_VALUES);

            barray[j][i] = forcing(i * h, j * h);
        }
    }
    DMDAVecRestoreArray(dm_, b, &barray);

    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

    VecGetSize(x, &dofs_);
}
