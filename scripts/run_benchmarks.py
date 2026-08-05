"""
run_benchmarks.py

Run benchmark sweeps from a YAML config file.

The active config is copied into the run output folder so every result set is
self-describing and reproducible.
"""

import argparse
import datetime
import itertools
import json
import math
import os
import subprocess
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_BIN = REPO_ROOT / "bin" / "benchmark"
RESULTS_DIR = REPO_ROOT / "results" / "json"
CONFIGS_DIR = REPO_ROOT / "configs" / "benchmarks"
DEFAULT_CONFIG_PATH = CONFIGS_DIR / "default.yaml"
SMOKE_CONFIG_PATH = CONFIGS_DIR / "smoke.yaml"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") == "1"
VALID_BACKENDS = {"cpp", "python"}
VALID_ASSEMBLY_MODES = {"manual", "ufl_highlevel"}


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    if not isinstance(cfg, dict):
        raise TypeError(f"Config must be a mapping at top level: {config_path}")
    if (
        "problems" not in cfg
        or not isinstance(cfg["problems"], dict)
        or not cfg["problems"]
    ):
        raise ValueError(
            f"Config must define a non-empty 'problems' mapping: {config_path}"
        )

    return cfg


def snapshot_config(run_dir: Path, config: dict, source_path: Path):
    snapshot_path = run_dir / "config.yaml"
    if snapshot_path.exists():
        return

    try:
        source_config = str(source_path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        # Keep metadata privacy-friendly when config is outside the repository.
        source_config = source_path.name

    snapshot_doc = {
        "meta": {
            "generated_by": "scripts/run_benchmarks.py",
            "source_config": source_config,
            "copied_at_utc": datetime.datetime.now(datetime.UTC).isoformat(
                timespec="seconds"
            ),
        },
        **config,
    }
    with open(snapshot_path, "w") as fh:
        yaml.safe_dump(snapshot_doc, fh, sort_keys=False)


def is_valid_combo(solver_flags: dict, nprocs: int) -> bool:
    """
    Filter out invalid combinations of solvers and MPI process counts.
    PETSc's native ILU, LU, and Cholesky preconditioners do not scale
    beyond 1 process unless an external parallel package like MUMPS is used.
    """
    pc = solver_flags.get("pc_type")
    mat_solver = solver_flags.get("pc_factor_mat_solver_type")

    return not (
        nprocs > 1
        and pc in ["ilu", "lu", "cholesky"]
        and mat_solver not in ["mumps", "superlu_dist", "pastix"]
    )


def product_dict(d: dict):
    """Cartesian product over a dict of lists -> list of dicts.
    {"a": [1,2], "b": [3,4]} -> [{"a":1,"b":3}, {"a":1,"b":4}, ...]
    """
    if not d:
        return [{}]
    keys = list(d.keys())
    return [
        dict(zip(keys, combo)) for combo in itertools.product(*[d[k] for k in keys])
    ]


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


def parse_string_or_list_field(
    problem_name: str, field_name: str, value, default: list[str]
) -> list[str]:
    if value is None:
        resolved = default
    elif isinstance(value, list):
        if not value:
            raise ValueError(
                f"Problem '{problem_name}' field '{field_name}' must not be an empty list."
            )
        resolved = value
    elif isinstance(value, str):
        # Support both YAML list syntax (`field: [a, b]`) and
        # shorthand comma-separated string (`field: a, b`).
        resolved = [part.strip() for part in value.split(",") if part.strip()]
        if not resolved:
            raise ValueError(
                f"Problem '{problem_name}' field '{field_name}' must define at least one value."
            )
    else:
        raise ValueError(
            f"Problem '{problem_name}' field '{field_name}' must be a string or list of strings."
        )

    normalized = []
    for entry in resolved:
        if not isinstance(entry, str):
            raise TypeError(
                f"Problem '{problem_name}' field '{field_name}' entries must be strings, got {type(entry).__name__}."
            )
        entry = entry.strip().lower()
        if entry and entry not in normalized:
            normalized.append(entry)

    if not normalized:
        raise ValueError(
            f"Problem '{problem_name}' field '{field_name}' must define at least one value."
        )

    return normalized


def resolve_backends(problem_name: str, problem_cfg: dict) -> list[str]:
    backend = problem_cfg.get("backend")

    normalized = parse_string_or_list_field(
        problem_name=problem_name,
        field_name="backend",
        value=backend,
        default=["cpp"],
    )
    for b in normalized:
        if b not in VALID_BACKENDS:
            allowed = ", ".join(sorted(VALID_BACKENDS))
            raise ValueError(
                f"Problem '{problem_name}' has unknown backend '{b}'. Allowed: {allowed}."
            )

    return normalized


def resolve_assembly_modes(problem_name: str, problem_cfg: dict) -> list[str]:
    assembly_mode = problem_cfg.get("assembly_mode")

    normalized = parse_string_or_list_field(
        problem_name=problem_name,
        field_name="assembly_mode",
        value=assembly_mode,
        default=["manual"],
    )
    for mode in normalized:
        if mode not in VALID_ASSEMBLY_MODES:
            allowed = ", ".join(sorted(VALID_ASSEMBLY_MODES))
            raise ValueError(
                f"Problem '{problem_name}' has unknown assembly_mode '{mode}'. Allowed: {allowed}."
            )

    return normalized


def build_run_specs(problem_name: str, problem_cfg: dict) -> list[dict]:
    """Expand one problem's config into a flat list of benchmark specs."""
    backends = resolve_backends(problem_name, problem_cfg)
    assembly_modes = resolve_assembly_modes(problem_name, problem_cfg)
    mesh_combos = product_dict(problem_cfg.get("mesh_sweep", {}))
    param_combos = product_dict(problem_cfg.get("param_sweep", {}))

    if any(mode != "manual" for mode in assembly_modes) and "python" not in backends:
        raise ValueError(
            f"Problem '{problem_name}' uses non-manual assembly_mode but has no python backend."
        )

    if problem_cfg["kind"] == "linear":
        solver_combos = [
            flatten_extra(pc, ksp)
            for pc, ksp in itertools.product(
                problem_cfg["solver_sweep"]["pc"], problem_cfg["solver_sweep"]["ksp"]
            )
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
    for backend in backends:
        backend_modes = ["manual"] if backend == "cpp" else assembly_modes
        for assembly_mode, mesh, param, solver in itertools.product(
            backend_modes, mesh_combos, param_combos, solver_combos
        ):
            specs.append(
                {
                    "problem": problem_name,
                    "backend": backend,
                    "assembly_mode": assembly_mode,
                    "mesh": mesh,
                    "param": param,
                    "solver": solver,
                }
            )
    return specs


def build_command(
    backend: str, problem: str, assembly_mode: str, nprocs: int
) -> list[str]:
    if backend == "cpp":
        return ["mpiexec", "-n", str(nprocs), str(BENCHMARK_BIN), "-problem", problem]
    elif backend == "python":
        return [
            "mpiexec",
            "-n",
            str(nprocs),
            "python3",
            "-m",
            "python_backend.runner",
            "-problem",
            problem,
            "-assembly_mode",
            assembly_mode,
        ]
    raise ValueError(f"Unknown backend: {backend}")


def run(
    spec: dict, nprocs: int, max_it: int = 3000, snes_max_it: int = 100, timeout_s=300
) -> dict:
    cmd = build_command(spec["backend"], spec["problem"], spec["assembly_mode"], nprocs)
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
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s, cwd=REPO_ROOT
        )
    except subprocess.TimeoutExpired:
        return {
            "error": f"Command timed out after {timeout_s} seconds",
            "timed_out": True,
        }

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
    parser.add_argument(
        "--dry-run", action="store_true", help="Print execution plan without running"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Path to YAML benchmark config. Defaults to configs/benchmarks/default.yaml "
            "(or configs/benchmarks/smoke.yaml when SMOKE_TEST=1)."
        ),
    )
    args = parser.parse_args()

    default_cfg = SMOKE_CONFIG_PATH if SMOKE_TEST else DEFAULT_CONFIG_PATH
    config_path = args.config or default_cfg
    config_path = (
        config_path if config_path.is_absolute() else (REPO_ROOT / config_path)
    )

    try:
        config = load_config(config_path)
    except (FileNotFoundError, ValueError, yaml.YAMLError) as exc:
        raise SystemExit(f"Failed to load config: {exc}")

    # Prefer a RUN_ID passed in from the submit script (keeps all nprocs/chunks
    # of one sweep together); fall back to generating one locally for ad-hoc runs.
    run_id = os.environ.get("RUN_ID")
    if run_id is None:
        sha = (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                cwd=REPO_ROOT,
            ).stdout.strip()
            or "nogit"
        )
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        run_id = f"{ts}_{sha}"

    run_dir = RESULTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot_config(run_dir, config, config_path)

    # 2. Build the baseline specs
    all_specs = []
    for name, cfg in config["problems"].items():
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
    print(f"CONFIG:                 {config_path}")
    print(f"TOTAL VALID JOBS:       {len(flat_jobs)}")
    print(f"SLURM_ARRAY_TASK_COUNT: {array_count}")
    print(f"CHUNK SIZE:             {chunk_size}")
    print("=" * 55)
    print(f"Array Task {array_id}/{array_count - 1}:")
    print(
        f"Assigned jobs {start_idx} to {end_idx - 1} ({len(my_jobs)} runs) -> {results_file}"
    )

    if not my_jobs:
        print("No jobs fall into this chunk. Exiting.")
        return

    # 7. Intercept execution if --dry-run is active
    if args.dry_run:
        print("\n[DRY RUN ENABLED] Previewing first 3 jobs in this chunk:")
        for nprocs, spec in my_jobs[:3]:
            # Print a clean summary of the solver parameters
            solver_summary = f"{spec['solver'].get('ksp_type', 'N/A')} + {spec['solver'].get('pc_type', 'N/A')}"
            print(
                f" -> {spec['problem']} ({spec['backend']}, {spec['assembly_mode']}) | Mesh: {spec['mesh']} | Solver: {solver_summary}"
            )
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
                backend=spec["backend"],
                assembly_mode=spec["assembly_mode"],
                nprocs=nprocs,
                **spec["mesh"],
                **spec["param"],
                **spec["solver"],
            )


if __name__ == "__main__":
    main()
