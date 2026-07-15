#include "benchmark_result.hpp"

#include <sstream>

std::string to_json(const BenchmarkResult& result)
{
    std::ostringstream oss;
    oss << "{\n";
    oss << "    \"problem\": \"" << result.problem << "\",\n";
    oss << "    \"ksp\": \"" << result.ksp_type << "\",\n";
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
