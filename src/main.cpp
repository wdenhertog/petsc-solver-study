#include <iostream>
#include <petscsys.h>

#include "linear_system.hpp"
#include "problem.hpp"
#include "problem_registry.hpp"

int main(int argc, char** argv)
{
    PetscInitialize(&argc, &argv, nullptr, nullptr);

    char problem_name[256];
    PetscOptionsGetString(nullptr, nullptr, "-problem", problem_name, sizeof(problem_name),
                          nullptr);

    std::unique_ptr<Problem> problem;
    try
    {
        problem = make_problem(problem_name);
    }
    catch (const std::out_of_range& e)
    {
        PetscPrintf(PETSC_COMM_WORLD, "%s\n", e.what());
        PetscFinalize();
        return 1;
    }

    BenchmarkResult result;
    result.problem = problem->name();

    switch (problem->kind())
    {
    case ProblemKind::Linear:
    {
        Mat A;
        Vec b, x;
        problem->assemble_linear(A, b, x);
        KSP ksp;
        KSPCreate(PETSC_COMM_WORLD, &ksp);
        KSPSetOperators(ksp, A, A);
        KSPSetFromOptions(ksp); // -ksp_type, -pc_type all come from CLI, free
        PetscLogDouble t0, t1, t2;
        PetscTime(&t0);
        KSPSetUp(ksp);
        PetscTime(&t1);
        KSPSolve(ksp, b, x);
        PetscTime(&t2);
        result.setup_time = t1 - t0;
        result.solve_time = t2 - t1;
        KSPGetIterationNumber(ksp, &result.iterations);
        KSPGetResidualNorm(ksp, &result.residual_norm);

        KSPConvergedReason reason;
        KSPGetConvergedReason(ksp, &reason);
        result.converged_reason = reason;
        result.converged_reason_string = KSPConvergedReasons[reason];
        result.success = (reason > 0);

        KSPType ksp_type;
        KSPGetType(ksp, &ksp_type);
        result.ksp_type = ksp_type;

        PC pc;
        KSPGetPC(ksp, &pc);
        PCType pc_type;
        PCGetType(pc, &pc_type);
        result.pc_type = pc_type;

        VecGetSize(x, &result.dofs);
        break;
    }
    case ProblemKind::Nonlinear:
    { /* SNESSolve, analogous */
    }
    case ProblemKind::VariationalInequality:
    { /* SNESVISetVariableBounds + SNESSolve */
    }
    }

    std::cout << to_json(result) << std::endl;
    PetscFinalize();
}
