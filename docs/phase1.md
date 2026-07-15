# Phase 1 – PETSc Fundamentals

## Introduction

The objective of this phase was to become familiar with the core PETSc abstractions and establish a minimal benchmarking
framework that can be extended throughout the rest of this project.

Rather than immediately solving finite element problems, the focus was placed on understanding PETSc's linear algebra
and solver infrastructure:

- Vec
- Mat
- KSP
- PC
- PETSc Options Database

The goal was to build a reusable benchmark workflow capable of:

- Comparing different Krylov methods.
- Comparing different preconditioners.
- Studying convergence behaviour.
- Measuring time-to-solution.
- Exporting results automatically for post-processing.

---

## Implementation

A small C++ application was developed using PETSc and CMake.

The benchmark constructs a sparse tridiagonal matrix corresponding to the one-dimensional Poisson operator

```text
A = tridiag(-1, 2, -1)
```

with a constant right-hand side vector.

The matrix size is configurable through PETSc's options database:

```bash
./benchmark -n 1000
```

Benchmark results are stored in a common `BenchmarkResult` structure and exported as JSON.

A Python automation script was developed to:

- Execute benchmark campaigns.
- Sweep over multiple matrix sizes.
- Compare different KSP and PC combinations.
- Store benchmark results.
- Generate plots automatically.

---

## Benchmark Setup

### Krylov Solvers

- CG
- GMRES

### Preconditioners

- Jacobi
- ILU
- GAMG

### Matrix Sizes

- 100
- 1,000
- 10,000
- 100,000
- 1,000,000

### Metrics

- Iteration count
- Time-to-solution
- Convergence status
- Residual norm

---

## Benchmark Results

### Iteration Counts

![](images/iterations_vs_size.png)

### Solve Times

![](images/solve_time_vs_size.png)

---

## Discussion

### CG + Jacobi

|      Size | Iterations | Status    |
|----------:|-----------:|-----------|
|       100 |         50 | Converged |
|     1,000 |        500 | Converged |
|    10,000 |      5,000 | Converged |
|   100,000 |     10,000 | Diverged  |
| 1,000,000 |     10,000 | Diverged  |

Observations:

- Iteration count grows approximately linearly with system size.
- Convergence deteriorates as the matrix becomes larger.
- The solver eventually reaches the maximum iteration limit.

This demonstrates that Jacobi is not a scalable preconditioner for this problem.

---

### CG + ILU

|      Size | Iterations |
|----------:|-----------:|
|       100 |          1 |
|     1,000 |          1 |
|    10,000 |          1 |
|   100,000 |          1 |
| 1,000,000 |          1 |

Observations:

- ILU reduces the iteration count to a single iteration for all tested matrix sizes.
- For this simple tridiagonal problem, ILU behaves almost like a direct solver.
- The solve time remains extremely small even at one million unknowns.

This result should be interpreted carefully because the one-dimensional Poisson operator is a particularly favourable
case for ILU.

---

### CG + GAMG

|      Size | Iterations |
|----------:|-----------:|
|       100 |          6 |
|     1,000 |          6 |
|    10,000 |          6 |
|   100,000 |          8 |
| 1,000,000 |         10 |

Observations:

- The iteration count remains nearly constant as the problem size increases.
- Only a small increase is observed between one hundred and one million unknowns.

This is the characteristic behaviour expected from algebraic multigrid methods and demonstrates mesh-independent
convergence.

---

### GMRES + Jacobi

|      Size | Iterations | Status    |
|----------:|-----------:|-----------|
|       100 |        795 | Converged |
|     1,000 |     10,000 | Diverged  |
|    10,000 |     10,000 | Diverged  |
|   100,000 |     10,000 | Diverged  |
| 1,000,000 |     10,000 | Diverged  |

Observations:

- GMRES performs considerably worse than CG on this symmetric positive definite problem.
- The method fails to converge for all but the smallest test case.

This demonstrates that a more general Krylov method is not necessarily a better choice.

---

### GMRES + ILU

Observations:

- Behaviour is nearly identical to CG + ILU.
- The quality of the preconditioner dominates solver performance.

---

### GMRES + GAMG

|      Size | Iterations |
|----------:|-----------:|
|       100 |          6 |
|     1,000 |          6 |
|    10,000 |          5 |
|   100,000 |          6 |
| 1,000,000 |          8 |

Observations:

- Iteration counts remain nearly constant.
- Similar behaviour is observed for CG + GAMG.

Again, the preconditioner appears to have a much larger impact than the Krylov method.

---

## Key Findings

### 1. Preconditioners matter more than Krylov methods

The largest performance gains were obtained by changing the preconditioner rather than the Krylov solver.

---

### 2. Jacobi does not scale

Jacobi preconditioning becomes increasingly ineffective as the system size grows and eventually fails to converge within
the iteration limit.

---

### 3. GAMG exhibits mesh-independent convergence

The number of iterations remained nearly constant from 100 up to 1,000,000 unknowns.

---

### 4. GMRES is not automatically better than CG

For this symmetric positive definite problem, CG significantly outperformed GMRES when both methods used the same
preconditioner.

---

### 5. Automated benchmarking is practical

Combining PETSc, JSON output and Python scripting provides a reproducible workflow for large benchmark campaigns.
