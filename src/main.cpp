#include <petscsys.h>

#include "linear_system.hpp"

int main(int argc, char** argv)
{
    PetscInitialize(&argc, &argv, nullptr, nullptr);
    auto result = solve_linear_system();

    PetscPrintf(PETSC_COMM_WORLD, "Solve time: %.6f s\n", result.solve_time);

    PetscPrintf(PETSC_COMM_WORLD, "Iterations: %d\n", static_cast<int>(result.iterations));

    PetscPrintf(PETSC_COMM_WORLD, "Residual: %e\n", result.residual);

    PetscPrintf(PETSC_COMM_WORLD, "Converged reason: %d\n", result.converged_reason);
    PetscPrintf(PETSC_COMM_WORLD, "Converged reason string: %s\n",
                result.converged_reason_string.c_str());

    PetscFinalize();
    return 0;
}
