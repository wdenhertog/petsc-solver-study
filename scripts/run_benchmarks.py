"""
run_benchmarks.py

CONFIG below is written as a plain nested dict deliberately shaped like the
YAML config this will eventually read from disk. Today it's hardcoded here;
later, `CONFIG = yaml.safe_load(open(path))` replaces this literal and
nothing downstream changes.
"""

import datetime
import itertools
import json
import subprocess
from pathlib import Path
import math
import os
import argparse

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_BIN = REPO_ROOT / "bin" / "benchmark"
RESULTS_DIR = REPO_ROOT / "results" / "json"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Config shape. This dict is the thing a YAML file will eventually deserialize
# into — keep every value here JSON/YAML-serializable (no functions, no
# Python-only objects) so the swap-over stays a one-liner.
# ---------------------------------------------------------------------------
CONFIG = {
    "problems": {
        "poisson": {
            "kind": "linear",
            "mesh_sweep": {"n": [32, 64, 128, 256, 512, 1024]},
            "param_sweep": {},  # no problem-specific params yet
            "solver_sweep": {
                "ksp": [
                    {"ksp_type": "cg", "extra": {}},
                    {"ksp_type": "gmres", "extra": {"ksp_gmres_restart": 20}},
                    {"ksp_type": "gmres", "extra": {"ksp_gmres_restart": 50}},
                    {"ksp_type": "gmres", "extra": {"ksp_gmres_restart": 100}},
                    {"ksp_type": "gmres", "extra": {"ksp_gmres_restart": 150}},
                    {"ksp_type": "gmres", "extra": {"ksp_gmres_restart": 200}},
                ],
                "pc": [
                    {"pc_type": "jacobi", "extra": {"pc_jacobi_type": "diagonal"}},
                    {"pc_type": "ilu", "extra": {"pc_factor_levels": 0}},
                    {"pc_type": "ilu", "extra": {"pc_factor_levels": 1}},
                    {"pc_type": "ilu", "extra": {"pc_factor_levels": 2}},
                    {"pc_type": "ilu", "extra": {"pc_factor_levels": 3}},
                    {"pc_type": "gamg", "extra": {"pc_gamg_type": "agg"}},
                    # {"pc_type": "gamg", "extra": {"pc_gamg_type": "classical"}},
                ],
                "direct": [
                    {"ksp_type": "preonly", "pc_type": "lu", "extra": {"pc_factor_mat_solver_type": "mumps"}},
                    {"ksp_type": "preonly", "pc_type": "cholesky", "extra": {"pc_factor_mat_solver_type": "mumps"}},
                ],
            },
        },
        "bratu": {
            "kind": "nonlinear",
            "mesh_sweep": {"n": [32, 64, 128, 256, 512, 1024]},
            "param_sweep": {"lambda": [1.0, 3.0, 5.0, 6.5, 6.8]},
            "solver_sweep": {
                "snes": [
                    {"snes_type": "newtonls", "extra": {"snes_linesearch_type": "basic"}},
                    {"snes_type": "newtonls", "extra": {"snes_linesearch_type": "bt"}},
                    {"snes_type": "newtontr", "extra": {}},
                ],
                "ksp": [
                    {"ksp_type": "gmres", "extra": {"ksp_gmres_restart": 30}},
                    {"ksp_type": "gmres", "extra": {"ksp_gmres_restart": 100}},
                    {"ksp_type": "bcgs", "extra": {}},
                    {"ksp_type": "fgmres", "extra": {"ksp_gmres_restart": 50}},
                ],
                "pc": [
                    {"pc_type": "gamg", "extra": {"pc_gamg_type": "agg"}},
                    {"pc_type": "gamg", "extra": {"pc_gamg_type": "agg", "pc_gamg_threshold": 0.05}},
                    {"pc_type": "asm", "extra": {"pc_asm_overlap": 1}},
                    {"pc_type": "asm", "extra": {"pc_asm_overlap": 2}},
                ],
            },
        },
    },
}

SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") == "1"

if SMOKE_TEST:
    CONFIG = {
        "problems": {
            "poisson": {
                "kind": "linear",
                "mesh_sweep": {"n": [64]},
                "param_sweep": {},
                "solver_sweep": {
                    "ksp": [{"ksp_type": "cg", "extra": {}}],
                    "pc": [{"pc_type": "jacobi", "extra": {"pc_jacobi_type": "diagonal"}}],
                    "direct": [],
                },
            },
            "bratu": {
                "kind": "nonlinear",
                "mesh_sweep": {"n": [64]},
                "param_sweep": {"lambda": [3.0]},
                "solver_sweep": {
                    "snes": [{"snes_type": "newtonls", "extra": {"snes_linesearch_type": "bt"}}],
                    "ksp": [{"ksp_type": "gmres", "extra": {"ksp_gmres_restart": 30}}],
                    "pc": [{"pc_type": "gamg", "extra": {"pc_gamg_type": "agg"}}],
                },
            },
        },
    }


def is_valid_combo(solver_flags: dict, nprocs: int) -> bool:
    """
    Filter out invalid combinations of solvers and MPI process counts.
    PETSc's native ILU, LU, and Cholesky preconditioners do not scale
    beyond 1 process unless an external parallel package like MUMPS is used.
    """
    pc = solver_flags.get("pc_type")
    mat_solver = solver_flags.get("pc_factor_mat_solver_type")

    if nprocs > 1 and pc in ["ilu", "lu", "cholesky"]:
        # Allow the run ONLY if a parallel external package is specified
        if mat_solver not in ["mumps", "superlu_dist", "pastix"]:
            return False

    return True


def product_dict(d: dict):
    """Cartesian product over a dict of lists -> list of dicts.
    {"a": [1,2], "b": [3,4]} -> [{"a":1,"b":3}, {"a":1,"b":4}, ...]
    """
    if not d:
        return [{}]
    keys = list(d.keys())
    return [dict(zip(keys, combo)) for combo in itertools.product(*[d[k] for k in keys])]


def flatten_extra(*configs) -> dict:
    """Merge a list of {"<type>_type": ..., "extra": {...}} dicts into one flat CLI-flag dict."""
    flags = {}
    for cfg in configs:
        for k, v in cfg.items():
            if k == "extra":
                flags.update(v)
            else:
                flags[k] = v
    return flags


def build_run_specs(problem_name: str, problem_cfg: dict) -> list[dict]:
    """Expand one problem's config into a flat list of {mesh, param, solver} combos."""
    mesh_combos = product_dict(problem_cfg["mesh_sweep"])
    param_combos = product_dict(problem_cfg["param_sweep"])

    if problem_cfg["kind"] == "linear":
        solver_combos = [
            flatten_extra(pc, ksp)
            for pc, ksp in itertools.product(problem_cfg["solver_sweep"]["pc"], problem_cfg["solver_sweep"]["ksp"])
        ] + [flatten_extra(d) for d in problem_cfg["solver_sweep"].get("direct", [])]
    elif problem_cfg["kind"] == "nonlinear":
        solver_combos = [
            flatten_extra(snes, ksp, pc)
            for snes, ksp, pc in itertools.product(
                problem_cfg["solver_sweep"]["snes"],
                problem_cfg["solver_sweep"]["ksp"],
                problem_cfg["solver_sweep"]["pc"],
            )
        ]
    else:
        raise ValueError(f"Unknown problem kind: {problem_cfg['kind']}")

    specs = []
    for mesh, param, solver in itertools.product(mesh_combos, param_combos, solver_combos):
        specs.append(
            {
                "problem": problem_name,
                "mesh": mesh,
                "param": param,
                "solver": solver,
            }
        )
    return specs


def run(spec: dict, nprocs: int, max_it: int = 3000, snes_max_it: int = 100, timeout_s=300) -> dict:
    cmd = ["mpiexec", "-n", str(nprocs), str(BENCHMARK_BIN), "-problem", spec["problem"]]
    for k, v in spec["mesh"].items():
        cmd += [f"-{k}", str(v)]
    for k, v in spec["param"].items():
        cmd += [f"-{k}", str(v)]
    for k, v in spec["solver"].items():
        cmd += [f"-{k}", str(v)]
    if spec["problem_kind"] == "linear":
        cmd += ["-ksp_max_it", str(max_it)]
    elif spec["problem_kind"] == "nonlinear":
        cmd += ["-snes_max_it", str(snes_max_it), "-ksp_max_it", str(max_it)]

    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return {"error": f"Command timed out after {timeout_s} seconds", "timed_out": True}

    try:
        return json.loads(out.stdout)
    except json.JSONDecodeError:
        return {"error": out.stderr.strip() or f"exit code {out.returncode}"}


def log(f, result: dict, **context):
    record = {**context, **result}
    f.write(json.dumps(record) + "\n")
    f.flush()
    ctx_str = " ".join(f"{k}={v}" for k, v in context.items())
    if "error" in result:
        print(f"{ctx_str}: ERROR - {result['error']}")
    else:
        print(
            f"{ctx_str}: {result.get('iterations')} its, "
            f"solve={result.get('solve_time', 0):.4f}s, success={result.get('success')}"
        )


def main():
    parser = argparse.ArgumentParser(description="Run PETSc Solver Benchmarks")
    parser.add_argument("--dry-run", action="store_true", help="Print execution plan without running")
    args = parser.parse_args()

    # Prefer a RUN_ID passed in from the submit script (keeps all nprocs/chunks
    # of one sweep together); fall back to generating one locally for ad-hoc runs.
    run_id = os.environ.get("RUN_ID")
    if run_id is None:
        sha = (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT
            ).stdout.strip()
            or "nogit"
        )
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        run_id = f"{ts}_{sha}"

    run_dir = RESULTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # 2. Build the baseline specs
    all_specs = []
    for name, cfg in CONFIG["problems"].items():
        specs = build_run_specs(name, cfg)
        for s in specs:
            s["problem_kind"] = cfg["kind"]
        all_specs.extend(specs)

    # 3. Slurm Variables
    array_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    array_count = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    target_nprocs = int(os.environ.get("TARGET_NPROCS", 1))

    # 4. Flatten the execution space ONLY for the requested nprocs
    flat_jobs = []
    for spec in all_specs:
        if is_valid_combo(spec["solver"], target_nprocs):
            flat_jobs.append((target_nprocs, spec))

    # 5. Calculate the chunk slice
    chunk_size = math.ceil(len(flat_jobs) / array_count)
    start_idx = array_id * chunk_size
    end_idx = min(start_idx + chunk_size, len(flat_jobs))
    my_jobs = flat_jobs[start_idx:end_idx]

    results_file = run_dir / f"p{target_nprocs}_chunk_{array_id:04d}.jsonl"

    # 6. Print the execution plan for verification
    print("=" * 55)
    print(f"TARGET_NPROCS:          {target_nprocs}")
    print(f"TOTAL VALID JOBS:       {len(flat_jobs)}")
    print(f"SLURM_ARRAY_TASK_COUNT: {array_count}")
    print(f"CHUNK SIZE:             {chunk_size}")
    print("=" * 55)
    print(f"Array Task {array_id}/{array_count-1}:")
    print(f"Assigned jobs {start_idx} to {end_idx-1} ({len(my_jobs)} runs) -> {results_file}")

    if not my_jobs:
        print("No jobs fall into this chunk. Exiting.")
        return

    # 7. Intercept execution if --dry-run is active
    if args.dry_run:
        print("\n[DRY RUN ENABLED] Previewing first 3 jobs in this chunk:")
        for nprocs, spec in my_jobs[:3]:
            # Print a clean summary of the solver parameters
            solver_summary = f"{spec['solver'].get('ksp_type', 'N/A')} + {spec['solver'].get('pc_type', 'N/A')}"
            print(f" -> {spec['problem']} | Mesh: {spec['mesh']} | Solver: {solver_summary}")
        print("...\nDry run complete. No simulations were executed.")
        return

    # 8. Execute the chunk (Only reached if --dry-run is omitted)
    with open(results_file, "w") as f:
        for nprocs, spec in my_jobs:
            result = run(spec, nprocs)
            log(
                f,
                result,
                problem=spec["problem"],
                nprocs=nprocs,
                **spec["mesh"],
                **spec["param"],
                **spec["solver"],
            )


if __name__ == "__main__":
    main()
