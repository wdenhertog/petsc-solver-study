#include "benchmark_result.hpp"

#include <petscksp.h>
#include <sstream>

std::string to_json(const BenchmarkResult& result)
{
    std::ostringstream oss;
    oss << "{\n";
    oss << "    \"problem\": \"" << result.problem << "\",\n";
    oss << "    \"ksp\": \"" << result.ksp_type << "\",\n";
    oss << "    \"gmres_restart\": " << result.gmres_restart << ",\n";
    oss << "    \"jacobi_type\": \"" << result.jacobi_type << "\",\n";
    oss << "    \"ilu_level\": " << result.ilu_level << ",\n";
    oss << "    \"gamg_type\": \"" << result.gamg_type << "\",\n";
    oss << "    \"pc\": \"" << result.pc_type << "\",\n";
    oss << "    \"dofs\": " << result.dofs << ",\n";
    oss << "    \"iterations\": " << result.iterations << ",\n";
    oss << "    \"residual\": " << result.residual_norm << ",\n";
    oss << "    \"setup_time\": " << result.setup_time << ",\n";
    oss << "    \"solve_time\": " << result.solve_time << ",\n";
    oss << "    \"success\": " << result.success << ",\n";
    oss << "    \"converged_reason\": " << result.converged_reason << ",\n";
    oss << "    \"converged_reason_string\": \"" << result.converged_reason_string << "\"\n";
    oss << "}";
    return oss.str();
}

void fill_solve_results(KSP ksp, BenchmarkResult& result)
{
    KSPGetIterationNumber(ksp, &result.iterations);
    KSPGetResidualNorm(ksp, &result.residual_norm);

    KSPConvergedReason reason;
    KSPGetConvergedReason(ksp, &reason);
    result.converged_reason = reason;
    result.converged_reason_string = KSPConvergedReasons[reason];
    result.success = (reason > 0);
}

void fill_solver_config(KSP ksp, BenchmarkResult& result)
{
    KSPType ksp_type;
    KSPGetType(ksp, &ksp_type);

    PC pc;
    KSPGetPC(ksp, &pc);
    PCType pc_type;
    PCGetType(pc, &pc_type);

    result.ksp_type = ksp_type;
    result.pc_type = pc_type;
    result.gmres_restart = -1;
    result.jacobi_type = "";
    result.ilu_level = -1;
    result.gamg_type = "";

    if (std::string(ksp_type) == "gmres")
    {
        KSPGMRESGetRestart(ksp, &result.gmres_restart);
    }
    if (std::string(pc_type) == "jacobi")
    {
        PCJacobiType jacobi_type;
        PCJacobiGetType(pc, &jacobi_type);
        result.jacobi_type = PCJacobiTypes[jacobi_type];
    }
    else if (std::string(pc_type) == "ilu")
    {
        PCFactorGetLevels(pc, &result.ilu_level);
    }
    else if (std::string(pc_type) == "gamg")
    {
        PCGAMGType gamg_type;
        PCGAMGGetType(pc, &gamg_type);
        result.gamg_type = gamg_type;
    }
}
