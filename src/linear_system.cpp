#include "linear_system.hpp"

#include <petscksp.h>

BenchmarkResult solve_linear_system()
{
    BenchmarkResult result;

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
    for (PetscInt i = 0; i < n; ++i)
    {
        if (i > 0)
        {
            MatSetValue(A, i, i - 1, -1.0, INSERT_VALUES);
        }
        MatSetValue(A, i, i, 2.0, INSERT_VALUES);

        if (i < n - 1)
        {
            MatSetValue(A, i, i + 1, -1.0, INSERT_VALUES);
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
    KSPViewFromOptions(ksp, nullptr, "-ksp_view");

    // Get preconditioner
    KSPGetPC(ksp, &pc);

    PetscLogDouble start_time;
    PetscLogDouble end_time;

    PetscTime(&start_time);

    // Solve Ax = b
    KSPSolve(ksp, b, x);

    PetscTime(&end_time);

    KSPType ksp_type;
    KSPGetType(ksp, &ksp_type);

    PCType pc_type;
    PCGetType(pc, &pc_type);

    KSPConvergedReason converged_reason;
    KSPGetConvergedReason(ksp, &converged_reason);

    const char* reason_str;
    KSPGetConvergedReasonString(ksp, &reason_str);

    PetscInt iterations;
    KSPGetIterationNumber(ksp, &iterations);

    PetscReal residual_norm;
    KSPGetResidualNorm(ksp, &residual_norm);

    result.problem = "1D Poisson Matrix";
    result.ksp = ksp_type;
    result.pc = pc_type;
    result.n = n;
    result.iterations = iterations;
    result.residual = residual_norm;
    result.solve_time = end_time - start_time;
    result.converged_reason = converged_reason;
    result.converged_reason_string = reason_str;
    result.error = std::nullopt;
    result.success = result.converged_reason > 0;

    KSPDestroy(&ksp);
    VecDestroy(&x);
    VecDestroy(&b);
    MatDestroy(&A);
    return result;
}
