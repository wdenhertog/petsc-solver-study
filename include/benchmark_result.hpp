#pragma once

#include <petscksp.h>
#include <petscsnes.h>
#include <petscsys.h>

#include <string>

struct BenchmarkResult
{
    std::string problem;
    std::string ksp_type;
    PetscInt mesh_size = 0;
    PetscInt dofs = 0;
    PetscInt iterations = 0;
    PetscReal solve_time = 0.0, setup_time = 0.0;
    PetscInt converged_reason;
    std::string converged_reason_string;
    PetscBool success;
    PetscReal residual_norm = 0.0;
    PetscLogDouble peak_memory_bytes = 0.0;
    PetscInt mpi_ranks = 1;
    PetscInt outer_iterations = 0;
};

std::string to_json(const BenchmarkResult& result);
void fill_solve_results(KSP ksp, BenchmarkResult& result);
void fill_solve_results(SNES snes, BenchmarkResult& result);
