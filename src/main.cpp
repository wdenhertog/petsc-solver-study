#include <iostream>
#include <petscsys.h>

#include "benchmark_result.hpp"
#include "git_version.h"
#include "problem.hpp"
#include "problem_registry.hpp"

int main(int argc, char** argv)
{
    PetscInitialize(&argc, &argv, nullptr, nullptr);
    PETSC_STDOUT = stderr;
    PetscMemorySetGetMaximumUsage();

    char problem_name[256] = {};
    PetscBool found;
    PetscOptionsGetString(nullptr, nullptr, "-problem", problem_name, sizeof(problem_name), &found);
    if (!found)
    {
        PetscPrintf(PETSC_COMM_WORLD, "Missing required -problem flag\n");
        PetscFinalize();
        return 1;
    }

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
    result.git_sha = GIT_SHA;
    result.git_dirty = GIT_DIRTY;
    char petsc_ver[64];
    PetscGetVersion(petsc_ver, sizeof(petsc_ver));
    result.petsc_version = petsc_ver;

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
        KSPSetFromOptions(ksp);
        PetscLogDouble t0, t1, t2;
        PetscTime(&t0);
        KSPSetUp(ksp);
        PetscTime(&t1);
        KSPSolve(ksp, b, x);
        PetscTime(&t2);
        result.setup_time = t1 - t0;
        result.solve_time = t2 - t1;
        fill_solve_results(ksp, result);
        VecGetSize(x, &result.dofs);
        break;
    }
    case ProblemKind::Nonlinear:
    {
        Vec x;
        SNES snes;
        SNESCreate(PETSC_COMM_WORLD, &snes);
        problem->assemble_nonlinear(snes, x);
        SNESSetFromOptions(snes);
        PetscLogDouble t0, t1, t2;
        PetscTime(&t0);
        SNESSetUp(snes);
        PetscTime(&t1);
        SNESSolve(snes, nullptr, x);
        PetscTime(&t2);
        result.setup_time = t1 - t0;
        result.solve_time = t2 - t1;
        fill_solve_results(snes, result);
        VecGetSize(x, &result.dofs);
        SNESDestroy(&snes);
        break;
    }
    case ProblemKind::VariationalInequality:
    {
        break;
    }
    }

    PetscMPIInt rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
    {
        std::cout << to_json(result) << std::endl;
    }
    PetscFinalize();
}
