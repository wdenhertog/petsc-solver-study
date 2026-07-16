#pragma once

#include <petscksp.h>
#include <petscsys.h>

#include <string>

struct BenchmarkResult
{
    std::string problem;
    std::string ksp_type, pc_type;
    PetscInt gmres_restart;
    std::string jacobi_type;
    PetscInt ilu_level;
    std::string gamg_type;
    int mesh_size = 0;
    PetscInt dofs = 0;
    PetscInt iterations = 0;
    double solve_time = 0.0, setup_time = 0.0;
    KSPConvergedReason converged_reason;
    std::string converged_reason_string;
    bool success;
    double residual_norm = 0.0;
    int mpi_ranks = 1;
};

std::string to_json(const BenchmarkResult& result);
void fill_solver_config(KSP ksp, BenchmarkResult& result);
void fill_solve_results(KSP ksp, BenchmarkResult& result);
