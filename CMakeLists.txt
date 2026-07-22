cmake_minimum_required(VERSION 3.20)

project(
        PetscSolverStudy
        VERSION 0.1.0
        DESCRIPTION "Learning PETSc"
        LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(MPI REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(PETSC REQUIRED IMPORTED_TARGET "PETSc>=3.21")

add_executable(
        benchmark
        src/main.cpp
        src/benchmark_result.cpp
        src/problem_registry.cpp
        src/problems/poisson.cpp
        src/problems/bratu.cpp
)

set_target_properties(
        benchmark
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin
)

target_include_directories(
        benchmark
        PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/include/problems
)

target_link_libraries(
        benchmark
        PRIVATE
        MPI::MPI_CXX
        PkgConfig::PETSC
)

find_package(Git QUIET)
if (NOT GIT_FOUND)
    message(WARNING "Git not found. Build will use unknown git hashes.")
    set(GIT_EXECUTABLE "echo") # Fallback to prevent script crash
endif ()

# Run the script on every build
add_custom_target(
        UpdateGitVersion
        COMMAND ${CMAKE_COMMAND}
        -DGIT_EXECUTABLE=${GIT_EXECUTABLE}
        -DSOURCE_DIR=${CMAKE_SOURCE_DIR}
        -DGEN_HEADER=${CMAKE_CURRENT_BINARY_DIR}/git_version.h
        -P ${CMAKE_SOURCE_DIR}/git_version.cmake
        BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/git_version.h
)

add_dependencies(benchmark UpdateGitVersion)
