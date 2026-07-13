#include "linear_system.hpp"

#include <petscksp.h>

void solve_linear_system()
{
    Mat A;
    Vec x, b;
    KSP ksp;
    PC pc;

    // Create n x n matrix
    PetscInt n = 1000;

    // Check whether n is set in options, otherwise use n = 1000
    PetscOptionsGetInt(nullptr, nullptr, "-n", &n, nullptr);

    MatCreate(PETSC_COMM_WORLD, &A);
    MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n);
    MatSetFromOptions(A);
    MatSetUp(A);

    // A =
    // [  2 -1  0  0 0 0 ... 0 ]
    // [ -1  2 -1  0 0 0 ... 0 ]
    // [  0 -1  2 -1 0 0 ... 0 ]
    // [ ..................... ]
    // [ 0 ...  0  0 0 0  -1 2 ]
    for (PetscInt i = 0; i < n; ++i) {
        if (i > 0) {
            MatSetValue(A, i, i-1, -1.0, INSERT_VALUES);
        }
        MatSetValue(A, i, i, 2.0, INSERT_VALUES);

        if (i < n-1) {
            MatSetValue(A, i, i+1, -1.0, INSERT_VALUES);
        }
    }

    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

    // Create vectors
    VecCreate(PETSC_COMM_WORLD, &b);
    VecSetSizes(b, PETSC_DECIDE, n);
    VecSetFromOptions(b);

    VecDuplicate(b, &x);

    // b = [ 1, 0]^T
    VecSet(b, 1.0);

    VecAssemblyBegin(b);
    VecAssemblyEnd(b);

    // Create solver
    KSPCreate(PETSC_COMM_WORLD, &ksp);

    // Associate matrix with solver
    KSPSetOperators(ksp, A, A);

    // Let PETSc read runtime options
    KSPSetFromOptions(ksp);

    // Get preconditioner and print default type
    KSPGetPC(ksp, &pc);

    PetscLogDouble start_time;
    PetscLogDouble end_time;

    PetscTime(&start_time);

    // Solve Ax = b
    KSPSolve(ksp, b, x);

    PetscTime(&end_time);

    PetscInt iterations;
    KSPGetIterationNumber(ksp, &iterations);

    PetscReal residual_norm;
    KSPGetResidualNorm(ksp, &residual_norm);

    PetscPrintf(PETSC_COMM_WORLD, "Solve time: %.6f s\n", end_time - start_time);
    PetscPrintf(PETSC_COMM_WORLD, "Number of iterations: %d\n", static_cast<int>(iterations));
    PetscPrintf(PETSC_COMM_WORLD, "Final residual: %e\n", residual_norm);

    // PetscPrintf(PETSC_COMM_WORLD, "\nSolution vector:\n");

    // VecView(x, PETSC_VIEWER_STDOUT_WORLD);

    KSPDestroy(&ksp);
    VecDestroy(&x);
    VecDestroy(&b);
    MatDestroy(&A);
}
