# petsc-solver-study

Config-driven PETSc benchmarking for PDE problems, with two authoring paths:

- C++ problems implemented directly against PETSc
- Python problems defined with UFL and solved with FEniCSx/PETSc

The goal is to benchmark solver and preconditioner choices in a reproducible way, on your own problems, with your own parameter sweeps.

## What This Project Is

This repository is evolving from a personal PETSc study project into a benchmarking framework.

You should be able to:

- Add a new PDE problem in C++ or Python
- Describe benchmark campaigns in YAML
- Sweep mesh/problem/solver settings
- Collect results as JSONL, CSV, and plots
- Explore runs in a Streamlit dashboard

## Current Status

- C++ benchmark pipeline: implemented
- YAML campaign configs: implemented
- Run-level config snapshot to results: implemented
- Dynamic dashboard instance controls from config/data: implemented
- Generic ranking by inferred instance columns: implemented
- Python UFL/FEniCSx problem backend: planned next major step

## Core Workflow

1. Define a benchmark campaign in YAML.
2. Run the campaign (single node or cluster array chunks).
3. Each run writes JSONL records and snapshots the effective config to the run folder.
4. Post-process with ranking and plots.
5. Explore interactively in the dashboard.

Planned automation for cluster runs:

- Generate a Slurm array submission file directly from the campaign YAML
- Generate a companion bash launcher/wrapper script from the same config
- Keep generated job settings (array size, nprocs targets, run id wiring) consistent with the benchmark plan

## Campaign Configs

Default campaign files live in:

- [configs/benchmarks/default.yaml](configs/benchmarks/default.yaml)
- [configs/benchmarks/smoke.yaml](configs/benchmarks/smoke.yaml)

Run with default config:

```bash
pixi run benchmark
```

Run with custom config:

```bash
pixi run benchmark --config configs/benchmarks/my_campaign.yaml
```

Each run folder under [results/json](results/json) includes:

- chunk JSONL files
- a copied [config.yaml](results/json) snapshot for reproducibility

The copied config contains metadata and stores source config as a repo-relative path for privacy.

## Config Shape

Today, campaigns are problem-centric and solver-sweep-centric:

```yaml
problems:
  poisson:
    kind: linear
    mesh_sweep:
      n: [64, 128, 256]
    param_sweep: {}
    labels:
      n: Mesh size (n)
    solver_sweep:
      ksp:
        - ksp_type: cg
          extra: {}
      pc:
        - pc_type: gamg
          extra:
            pc_gamg_type: agg
      direct: []
```

Key points:

- mesh_sweep and param_sweep define the problem instance dimensions
- solver_sweep defines how PETSc is configured
- labels lets the dashboard render human-friendly control names

## Problem Authoring: C++ and Python

### C++ Problems

C++ problems are compiled into the benchmark binary and selected by problem name from YAML.

Typical flow:

1. Add a problem implementation in [src/problems](src/problems).
2. Add headers in [include/problems](include/problems).
3. Register the problem in the registry in [src/problem_registry.cpp](src/problem_registry.cpp).
4. Add a config entry in YAML under problems.
5. Run a quick smoke campaign to validate wiring and solver options.
6. Run your desired benchmark campaign.

Example commands:

```bash
# quick smoke check
pixi run benchmark --config configs/benchmarks/smoke.yaml --dry-run

# full campaign
pixi run benchmark --config configs/benchmarks/default.yaml
```

### Python Problems (UFL/FEniCSx)

Planned direction:

- Define variational forms in UFL
- Assemble/solve through FEniCSx with PETSc backends
- Emit benchmark records in the same schema as C++ runs
- Reuse the same ranking, plotting, and dashboard stack

Design goal:

- one benchmark config format
- multiple problem backends
- shared output format and analysis tools

## Benchmark Outputs

Raw results:

- [results/json](results/json): line-delimited JSON records per run chunk

Derived outputs:

- [results/csv](results/csv): ranked tables
- [results/plots](results/plots): static PNG diagnostics

## Analysis and Visualization

Generate static plots:

```bash
pixi run plot
```

Generate rankings:

```bash
pixi run python scripts/rank_results.py
```

Launch interactive dashboard:

```bash
pixi run dashboard
```

The dashboard reads each run folder's copied config and builds problem-instance controls dynamically.

## HPC Execution (Current)

Current cluster workflow uses two handwritten scripts:

- `submit_benchmarks.sh`: creates one shared `RUN_ID` and submits one Slurm array per MPI size (`1, 2, 4, 8, 16`)
- `petsc_benchmark_array.slurm`: executes one array task and runs `run_benchmarks.py` with exported `TARGET_NPROCS` and `RUN_ID`

This is a good baseline and already matches the benchmark runner contract:

- `RUN_ID` keeps all chunks across MPI sizes grouped in one run folder
- `TARGET_NPROCS` selects which MPI-size slice each array submission executes
- `SLURM_ARRAY_TASK_ID` and `SLURM_ARRAY_TASK_COUNT` drive chunk partitioning

## HPC Execution (Planned Automation)

Future automation should generate both scripts from YAML campaign config while preserving your current behavior.

Planned generated artifacts:

- `generated/submit_<campaign>.sh`
- `generated/petsc_benchmark_array_<campaign>.slurm`

Planned config-to-Slurm mapping:

- `hpc.nprocs: [1, 2, 4, 8, 16]` -> one `sbatch` per entry, exporting `TARGET_NPROCS`
- `hpc.time_limit` -> `sbatch --time=...`
- `hpc.array.count` -> `#SBATCH --array=0-(count-1)`
- `hpc.cluster`, `hpc.account`, `hpc.mail_user`, `hpc.mail_type` -> corresponding `#SBATCH` directives
- `hpc.modules` -> generated `module load ...` lines
- `hpc.repo_dir` -> `cd` target for executing `run_benchmarks.py`

Design goal: keep generated scripts transparent and editable, not hidden runtime magic.

## Repository Structure

Main folders relevant to the benchmark workflow:

- [configs/benchmarks](configs/benchmarks): benchmark campaign YAML files
- [include](include): C++ interfaces and problem declarations
- [src](src): C++ implementations and benchmark entrypoint
- [scripts](scripts): benchmark orchestration, ranking, plotting, dashboard
- [results](results): raw and derived benchmark outputs
- [docs](docs): notes and experiment writeups

## Suggested Workflow for Custom Problems

1. Implement the problem backend (C++ today, Python UFL/FEniCSx next).
2. Add campaign entries in YAML, including mesh_sweep, param_sweep, and solver_sweep.
3. Run the smoke config first for fast validation.
4. Run the full campaign locally or on a cluster.
5. Inspect results with dashboard and plots (solve time, iterations, memory, robustness).

## Roadmap

Near-term priorities:

- Add first Python problem adapter using UFL/FEniCSx
- Normalize C++ and Python outputs to one shared record schema
- Add config schema validation and helpful errors
- Add mixed-language campaign examples
- Auto-generate Slurm array submission files from campaign config
- Auto-generate cluster launcher bash scripts from campaign config
- Preserve compatibility with existing `submit_benchmarks.sh` + `petsc_benchmark_array.slurm` behavior
- Add docs for cluster-scale runs and reproducibility best practices

## Why This Matters

The point is not just to solve one PDE once.

The point is to make solver decisions evidence-driven:

- reproducible campaigns
- comparable outputs
- easy custom problem integration
- analysis tools that scale with project complexity