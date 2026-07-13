cmake_minimum_required(VERSION 3.20)

project(
        PetscLearningLab
        VERSION 0.1.0
        DESCRIPTION "Learning PETSc"
        LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(
        benchmark
        src/main.cpp
)

target_include_directories(
        benchmark
        PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)