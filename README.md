# petsc-solver-study

A personal project for learning modern C++, PETSc, MPI, and scientific computing from first principles.

The goal of this repository is to develop a deep understanding of PETSc by studying increasingly complex classes of linear and nonlinear systems arising from the discretization of partial differential equations.

The project follows a progression based on algebraic complexity rather than physical complexity:

SPD systems
→ Non-symmetric systems
→ Nonlinear systems
→ Saddle-point systems
→ Incompressible flow
→ Variational inequalities
→ Large-scale parallel computations

Rather than only using PETSc through FEniCSx, this project focuses on working directly with PETSc in C++ to understand the underlying algorithms and solver infrastructure.

---

## Objectives

This repository aims to answer questions such as:

- Why does CG require SPD matrices?
- When should GMRES be preferred over BiCGStab?
- How important is the choice of preconditioner?
- Why is AMG effective for elliptic problems?
- How do nonlinear solver strategies affect convergence?
- How does solver performance change with mesh size?
- How efficiently do PETSc solvers scale across multiple MPI processes?

---

## Benchmarking Philosophy

A central goal of this project is not only to understand individual PETSc components, but also to compare them quantitatively.

Whenever possible, solver configurations will be benchmarked against each other across multiple problem classes, mesh sizes, and MPI process counts.

The objective is to identify:

- Which solvers converge reliably
- Which preconditioners provide the best performance
- How performance changes with problem size
- How solver choices depend on matrix properties
- How scalability changes with increasing parallelism

All studies will be performed using reproducible benchmark suites and automated result collection.

---

## Planned Studies

### Linear Solvers (KSP)

Investigate:

- CG
- GMRES
- BiCGStab
- MINRES
- PREONLY

Benchmark Metrics:

- Iteration count
- Solve time
- Setup time
- Memory usage
- Convergence behavior
- Time-to-solution

---

### Preconditioners (PC)

Investigate:

- Jacobi
- SOR
- ILU
- LU
- GAMG
- Hypre BoomerAMG

Study:

- Robustness
- Scalability
- Sensitivity to problem type

---

### Nonlinear Solvers (SNES)

Classical nonlinear methods:

- Newton Line Search (`newtonls`)
- Newton Trust Region (`newtontr`)
- Quasi-Newton (`qn`)
- Nonlinear GMRES (`ngmres`)

Metrics:

- Newton iterations
- Linear iterations per Newton step
- Total solve time
- Robustness to poor initial guesses

---

### Variational Inequality Solvers (SNESVI)

Investigate:

- Reduced-Space Active Set (`vinewtonrsls`)
- Semi-Smooth Active Set (`vinewtonssls`)

Applications:

- Obstacle problems
- Contact mechanics
- Damage irreversibility
- Phase-field fracture
- Constrained optimization

Metrics:

- Active set evolution
- Nonlinear iterations
- Robustness
- Constraint satisfaction

---

## Problems

The study progresses through increasingly challenging algebraic structures.

### Poisson Equation

Properties:

- Linear
- Symmetric
- Positive Definite (SPD)

Purpose:

- PETSc fundamentals
- CG
- AMG
- Hypre

---

### Linear Elasticity

Properties:

- Linear
- SPD
- Vector-valued

Purpose:

- Block structures
- AMG robustness
- Larger systems

---

### Convection-Diffusion

Properties:

- Linear
- Non-symmetric

Purpose:

- GMRES
- BiCGStab
- Preconditioner sensitivity

---

### Nonlinear Poisson

Properties:

- Nonlinear

Purpose:

- SNES
- Newton methods
- Convergence studies

---

### Stokes Flow

Properties:

- Saddle-point system
- Indefinite matrix

Purpose:

- FieldSplit
- Schur complements
- Block preconditioners

---

### Navier-Stokes

Properties:

- Nonlinear
- Non-symmetric
- Saddle-point structure

Purpose:

- Realistic solver strategies
- Coupled nonlinear systems
- Advanced preconditioning

---

### Variational Inequality Problems

Examples:

- Obstacle problems
- Contact mechanics
- Phase-field fracture
- Damage irreversibility

Purpose:

- SNESVI
- Active-set methods
- Constraint handling

---

## Benchmark Studies

For each problem, benchmark campaigns will compare combinations of:

### Solvers

- CG
- GMRES
- BiCGStab
- MINRES
- PREONLY

### Preconditioners

- Jacobi
- SOR
- ILU
- LU
- GAMG
- Hypre BoomerAMG

### Nonlinear Methods

- newtonls
- newtontr
- qn
- ngmres
- vinewtonrsls
- vinewtonssls

### Study Dimensions

- Problem type
- Mesh size
- Degrees of freedom
- 2D versus 3D
- Number of MPI processes

Benchmark results will be exported to machine-readable formats (CSV or JSON) to enable automated post-processing, plotting, and report generation.

---

## Mesh Refinement Studies

Planned mesh sizes:

- 32 × 32
- 64 × 64
- 128 × 128
- 256 × 256
- 512 × 512

Metrics:

- Degrees of freedom
- Iteration count
- Runtime
- Memory consumption

Goal:

- Understand algorithmic scalability
- Investigate mesh-independent convergence

---

## Three-Dimensional Studies

Selected problems will eventually be extended to three dimensions.

Goals:

- Study memory consumption
- Compare direct and iterative methods
- Investigate scalability at larger problem sizes
- Prepare for realistic HPC workloads

Target scales:

- 10⁶ DOFs
- 10⁷ DOFs
- 10⁸ DOFs

---

## Parallel Performance Studies

### Strong Scaling

Fixed problem size.

Example:

- 1 process
- 2 processes
- 4 processes
- 8 processes
- 16 processes

Measure:

- Assembly time
- Solve time
- Speedup
- Parallel efficiency

---

### Weak Scaling

Problem size increases with the number of MPI processes.

Measure:

- Runtime growth
- Parallel efficiency
- Communication overhead

---

## Project Structure

```text
petsc-solver-study/
│
├── include/
├── src/
│
├── problems/
│   ├── poisson/
│   ├── elasticity/
│   ├── convection_diffusion/
│   ├── nonlinear_poisson/
│   ├── stokes/
│   ├── navier_stokes/
│   └── variational_inequalities/
│
├── docs/
├── results/
├── reports/
└── scripts/
```

---

## Learning Log

The repository will also serve as a study journal documenting:

- PETSc concepts
- Numerical linear algebra notes
- Solver observations
- Benchmark results
- Lessons learned

---

## Roadmap

### Phase 1 – Foundations

- [X] Configure CMake project
- [ ] Install and link PETSc
- [ ] Build first PETSc application
- [ ] Solve a small linear system
- [ ] Learn Mat, Vec, KSP and PC
- [ ] Learn PETSc options database
- [ ] Learn basic MPI workflow

### Phase 2 – SPD Systems

- [ ] Poisson problem
- [ ] CG
- [ ] GMRES
- [ ] Jacobi
- [ ] ILU

### Phase 3 – Multigrid

- [ ] GAMG
- [ ] Hypre BoomerAMG
- [ ] Mesh refinement studies

### Phase 4 – Non-Symmetric Systems

- [ ] Convection-Diffusion
- [ ] GMRES
- [ ] BiCGStab

### Phase 5 – Nonlinear Systems

- [ ] Nonlinear Poisson
- [ ] Newton methods
- [ ] SNES comparison

### Phase 6 – Saddle-Point Systems

- [ ] Stokes
- [ ] FieldSplit
- [ ] Schur complements

### Phase 7 – Incompressible Flow

- [ ] Navier-Stokes
- [ ] Nonlinear coupled systems
- [ ] Advanced preconditioning

### Phase 8 – Variational Inequalities

- [ ] vinewtonrsls
- [ ] vinewtonssls
- [ ] Obstacle problems
- [ ] Contact mechanics

### Phase 9 – HPC and Scalability

- [ ] MPI strong scaling
- [ ] MPI weak scaling
- [ ] 3D problems
- [ ] Automated benchmarking
- [ ] Solver performance report

---

## Future Extensions

Potential future directions include:

- FEniCSx implementations of selected benchmark problems
- Comparison between direct PETSc usage and PETSc through FEniCSx
- Matrix-free methods
- Advanced multigrid configurations
- Domain decomposition methods

---

## Motivation

The primary goal of this project is learning.

By building a benchmarking framework from scratch in modern C++, I aim to develop a deeper understanding of PETSc, scientific computing, numerical linear algebra, and high-performance computing.

A secondary goal is to create a reproducible solver study that systematically compares PETSc solver technologies across a wide range of representative problems.