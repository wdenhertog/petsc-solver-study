#include <petscsys.h>

int main(int argc, char** argv)
{
    PetscInitialize(&argc, &argv, nullptr, nullptr);

    PetscPrintf(PETSC_COMM_WORLD,
                "Hello from PETSc!\n");

    PetscFinalize();

    return 0;
}