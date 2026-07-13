#include <petscsys.h>

#include "linear_system.hpp"

int main(int argc, char** argv)
{
    PetscInitialize(&argc, &argv, nullptr, nullptr);
    solve_linear_system();

    PetscFinalize();
    return 0;
}