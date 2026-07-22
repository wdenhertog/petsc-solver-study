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
pkg_check_modules(PETSC REQUIRED IMPORTED_TARGET PETSc)

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
if (GIT_FOUND)
    execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
            OUTPUT_VARIABLE GIT_SHA OUTPUT_STRIP_TRAILING_WHITESPACE
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    execute_process(COMMAND ${GIT_EXECUTABLE} status --porcelain
            OUTPUT_VARIABLE GIT_DIRTY_OUT OUTPUT_STRIP_TRAILING_WHITESPACE
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    if (GIT_DIRTY_OUT STREQUAL "")
        set(GIT_DIRTY "0")
    else ()
        set(GIT_DIRTY "1")
    endif ()
else ()
    set(GIT_SHA "unknown")
    set(GIT_DIRTY "1")
endif ()

target_compile_definitions(benchmark PRIVATE
        GIT_SHA="${GIT_SHA}"
        GIT_DIRTY=${GIT_DIRTY}
)
