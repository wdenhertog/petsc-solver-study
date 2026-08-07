#include <iostream>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <petscsys.h>
#include <petscts.h>

#include "benchmark_result.hpp"
#include "git_version.h"
#include "problem.hpp"
#include "problem_registry.hpp"

namespace
{
struct TransientTsCtx
{
    Mat M = nullptr;
    Mat K = nullptr;
    Vec work = nullptr;
    PetscReal imex_alpha = 1.0;
};

PetscErrorCode ifunction(TS, PetscReal, Vec u, Vec udot, Vec f, void* ctx)
{
    auto* c = static_cast<TransientTsCtx*>(ctx);
    MatMult(c->M, udot, f);
    MatMult(c->K, u, c->work);
    VecAXPY(f, c->imex_alpha, c->work);
    return PETSC_SUCCESS;
}

PetscErrorCode ijacobian(TS, PetscReal, Vec, Vec, PetscReal shift, Mat A, Mat B, void* ctx)
{
    auto* c = static_cast<TransientTsCtx*>(ctx);
    MatCopy(c->K, B, SAME_NONZERO_PATTERN);
    MatScale(B, c->imex_alpha);
    MatAXPY(B, shift, c->M, SUBSET_NONZERO_PATTERN);
    if (A != B)
    {
        MatCopy(B, A, SAME_NONZERO_PATTERN);
    }
    return PETSC_SUCCESS;
}

PetscErrorCode rhsfunction(TS, PetscReal, Vec u, Vec g, void* ctx)
{
    auto* c = static_cast<TransientTsCtx*>(ctx);
    MatMult(c->K, u, g);
    VecScale(g, -(1.0 - c->imex_alpha));
    return PETSC_SUCCESS;
}

PetscErrorCode rhsjacobian(TS, PetscReal, Vec, Mat A, Mat B, void* ctx)
{
    auto* c = static_cast<TransientTsCtx*>(ctx);
    MatCopy(c->K, B, SAME_NONZERO_PATTERN);
    MatScale(B, -(1.0 - c->imex_alpha));
    if (A != B)
    {
        MatCopy(B, A, SAME_NONZERO_PATTERN);
    }
    return PETSC_SUCCESS;
}

void solve_transient_ts(Problem& problem, BenchmarkResult& result)
{
    Mat M, K, J;
    Vec x, f;
    problem.assemble_transient(M, K, x);

    TS ts;
    TSCreate(PETSC_COMM_WORLD, &ts);
    TSSetProblemType(ts, TS_LINEAR);
    TSSetSolution(ts, x);

    PetscReal dt = 1.0e-3, max_time = 0.1;
    PetscInt max_steps = 100000;
    PetscBool has_time_step = PETSC_FALSE;
    PetscOptionsGetReal(nullptr, nullptr, "-ts_time_step", &dt, &has_time_step);
    if (!has_time_step)
    {
        PetscOptionsGetReal(nullptr, nullptr, "-ts_dt", &dt, nullptr);
    }
    PetscOptionsGetReal(nullptr, nullptr, "-ts_max_time", &max_time, nullptr);
    PetscOptionsGetInt(nullptr, nullptr, "-ts_max_steps", &max_steps, nullptr);

    TSSetTime(ts, 0.0);
    TSSetTimeStep(ts, dt);
    TSSetMaxTime(ts, max_time);
    TSSetMaxSteps(ts, max_steps);

    TransientTsCtx ctx;
    ctx.M = M;
    ctx.K = K;
    ctx.imex_alpha = 1.0;
    PetscOptionsGetReal(nullptr, nullptr, "-ts_imex_alpha", &ctx.imex_alpha, nullptr);
    ctx.imex_alpha = std::clamp(ctx.imex_alpha, 0.0, 1.0);

    VecDuplicate(x, &ctx.work);
    VecDuplicate(x, &f);
    MatDuplicate(K, MAT_COPY_VALUES, &J);

    TSSetIFunction(ts, f, ifunction, &ctx);
    TSSetIJacobian(ts, J, J, ijacobian, &ctx);

    Vec g = nullptr;
    Mat R = nullptr;
    if (ctx.imex_alpha < 1.0 - 1.0e-14)
    {
        VecDuplicate(x, &g);
        MatDuplicate(K, MAT_COPY_VALUES, &R);
        TSSetRHSFunction(ts, g, rhsfunction, &ctx);
        TSSetRHSJacobian(ts, R, R, rhsjacobian, &ctx);
    }

    TSSetFromOptions(ts);

    PetscLogDouble t0, t1, t2;
    PetscTime(&t0);
    TSSetUp(ts);
    PetscTime(&t1);
    TSSolve(ts, x);
    PetscTime(&t2);

    result.setup_time = t1 - t0;
    result.solve_time = t2 - t1;

    PetscInt step = 0;
    PetscInt ksp_total = 0;
    PetscInt snes_total = 0;
    PetscReal time = 0.0;
    TSConvergedReason reason;

    TSGetStepNumber(ts, &step);
    TSGetTime(ts, &time);
    TSGetConvergedReason(ts, &reason);
    TSGetKSPIterations(ts, &ksp_total);
    TSGetSNESIterations(ts, &snes_total);

    result.iterations = ksp_total;
    result.n_ksp_iterations_total = ksp_total;
    result.n_snes_iterations_total = snes_total;
    result.n_timesteps = step;
    result.final_time = time;
    result.converged_reason = reason;
    result.converged_reason_string = TSConvergedReasons[reason];
    result.success = reason > 0 ? PETSC_TRUE : PETSC_FALSE;
    VecNorm(x, NORM_2, &result.residual_norm);
    result.l2_error_vs_exact = problem.transient_l2_error(x, time);
    VecGetSize(x, &result.dofs);

    if (R)
        MatDestroy(&R);
    if (g)
        VecDestroy(&g);
    MatDestroy(&J);
    VecDestroy(&f);
    VecDestroy(&ctx.work);
    TSDestroy(&ts);
    MatDestroy(&K);
    MatDestroy(&M);
    VecDestroy(&x);
}
} // namespace

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

    try
    {
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
        case ProblemKind::Transient:
        {
            solve_transient_ts(*problem, result);
            break;
        }
        }
    }
    catch (const std::invalid_argument& e)
    {
        PetscPrintf(PETSC_COMM_WORLD, "%s\n", e.what());
        PetscFinalize();
        return 1;
    }

    fill_memory_usage(result);
    PetscMPIInt rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
    {
        std::cout << to_json(result) << std::endl;
    }
    problem.reset();
    PetscFinalize();
}
