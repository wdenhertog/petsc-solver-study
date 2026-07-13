#pragma once

#include <petscksp.h>
#include <petscsys.h>

#include <optional>
#include <string>

struct BenchmarkResult
{
    std::string problem;

    std::string ksp;
    std::string pc;

    PetscInt n{};

    PetscInt iterations{};
    PetscReal residual{};

    double solve_time{};

    KSPConvergedReason converged_reason{};
    std::string converged_reason_string;

    std::optional<std::string> error;
};

std::string to_json(const BenchmarkResult& result);
