#include <iostream>
#include <petscsys.h>

#include "linear_system.hpp"

int main(int argc, char** argv)
{
    PetscInitialize(&argc, &argv, nullptr, nullptr);
    auto result = solve_linear_system();

    std::cout << to_json(result) << std::endl;

    PetscFinalize();
    return 0;
}
