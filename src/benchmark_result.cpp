#include "benchmark_result.hpp"

#include <petscksp.h>
#include <sstream>

std::string to_json(const BenchmarkResult& result)
{
    std::ostringstream oss;
    oss << "{\n";
    oss << "    \"problem\": \"" << result.problem << "\",\n";
    oss << "    \"dofs\": " << result.dofs << ",\n";
    oss << "    \"iterations\": " << result.iterations << ",\n";
    oss << "    \"residual\": " << result.residual_norm << ",\n";
    oss << "    \"setup_time\": " << result.setup_time << ",\n";
    oss << "    \"solve_time\": " << result.solve_time << ",\n";
    oss << "    \"peak_memory_bytes\": " << result.peak_memory_bytes << ",\n";
    oss << "    \"success\": " << result.success << ",\n";
    oss << "    \"converged_reason\": " << result.converged_reason << ",\n";
    oss << "    \"converged_reason_string\": \"" << result.converged_reason_string << "\",\n";
    oss << "    \"outer_iterations\": " << result.outer_iterations << "\n";
    oss << "}";
    return oss.str();
}

void fill_solve_results(KSP ksp, BenchmarkResult& result)
{
    KSPGetIterationNumber(ksp, &result.iterations);
    KSPGetResidualNorm(ksp, &result.residual_norm);
    PetscMemoryGetMaximumUsage(&result.peak_memory_bytes);

    KSPConvergedReason reason;
    KSPGetConvergedReason(ksp, &reason);
    result.converged_reason = reason;
    result.converged_reason_string = KSPConvergedReasons[reason];
    result.success = reason > 0 ? PETSC_TRUE : PETSC_FALSE;
}

void fill_solve_results(SNES snes, BenchmarkResult& result)
{
    SNESConvergedReason reason;
    SNESGetConvergedReason(snes, &reason);
    result.converged_reason = static_cast<int>(reason);
    result.converged_reason_string = SNESConvergedReasons[reason];
    result.success = reason > 0 ? PETSC_TRUE : PETSC_FALSE;

    PetscInt total_linear_its;
    SNESGetLinearSolveIterations(snes, &total_linear_its);
    result.iterations = total_linear_its;
    SNESGetIterationNumber(snes, &result.outer_iterations);

    Vec F;
    SNESGetFunction(snes, &F, nullptr, nullptr);
    VecNorm(F, NORM_2, &result.residual_norm);
    PetscMemoryGetMaximumUsage(&result.peak_memory_bytes);
}
