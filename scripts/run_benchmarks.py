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
    "mpi": {"nprocs": [1, 4]},

    "problems": {
        "poisson": {
            "kind": "linear",
            "mesh_sweep": {"n": [32, 64, 128, 256]},
            "param_sweep": {},  # no problem-specific params yet
            "solver_sweep": {
                "ksp": [
                    {"ksp_type": "cg", "extra": {}},
                    {"ksp_type": "gmres", "extra": {"ksp_gmres_restart": 30}},
                    {"ksp_type": "gmres", "extra": {"ksp_gmres_restart": 50}},
                ],
                "pc": [
                    {"pc_type": "jacobi", "extra": {"pc_jacobi_type": "diagonal"}},
                    {"pc_type": "ilu", "extra": {"pc_factor_levels": 0}},
                    {"pc_type": "ilu", "extra": {"pc_factor_levels": 1}},
                    {"pc_type": "gamg", "extra": {"pc_gamg_type": "agg"}},
                ],
                "direct": [
                    {"ksp_type": "preonly", "pc_type": "lu", "extra": {}},
                    {"ksp_type": "preonly", "pc_type": "cholesky", "extra": {}},
                ],
            },
        },
        "bratu": {
            "kind": "nonlinear",
            "mesh_sweep": {"n": [32, 64, 128]},
            "param_sweep": {"lambda": [1.0, 3.0, 5.0, 6.5]},
            "solver_sweep": {
                "snes": [
                    {"snes_type": "newtonls", "extra": {}},
                    {"snes_type": "newtonls", "extra": {"snes_linesearch_type": "bt"}},
                    {"snes_type": "newtontr", "extra": {}},
                ],
                "ksp": [
                    {"ksp_type": "gmres", "extra": {}},
                ],
                "pc": [
                    {"pc_type": "gamg", "extra": {"pc_gamg_type": "agg"}},
                    {"pc_type": "ilu", "extra": {"pc_factor_levels": 0}},
                ],
            },
        },
    },
}


def is_valid_combo(solver_flags: dict, nprocs: int) -> bool:
    if nprocs > 1 and solver_flags.get("pc_type") == "ilu":
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
    for mesh, param, solver in itertools.product(mesh_combos, param_combos, solver_combos):
        specs.append({
            "problem": problem_name,
            "mesh": mesh,
            "param": param,
            "solver": solver,
        })
    return specs


def run(spec: dict, nprocs: int, max_it: int = 3000) -> dict:
    cmd = ["mpiexec", "-n", str(nprocs), str(BENCHMARK_BIN),
           "-problem", spec["problem"]]
    for k, v in spec["mesh"].items():
        cmd += [f"-{k}", str(v)]
    for k, v in spec["param"].items():
        cmd += [f"-{k}", str(v)]
    for k, v in spec["solver"].items():
        cmd += [f"-{k}", str(v)]
    if spec["problem_kind"] == "linear":
        cmd += ["-ksp_max_it", str(max_it)]

    out = subprocess.run(cmd, capture_output=True, text=True)
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
        print(f"{ctx_str}: {result.get('iterations')} its, "
              f"solve={result.get('solve_time', 0):.4f}s, success={result.get('success')}")


def main():
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip() or "nogit"
    ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    results_file = RESULTS_DIR / f"{sha}_{ts}.jsonl"

    all_specs = []
    for name, cfg in CONFIG["problems"].items():
        specs = build_run_specs(name, cfg)
        for s in specs:
            s["problem_kind"] = cfg["kind"]
        all_specs.extend(specs)

    total_runs = len(all_specs) * len(CONFIG["mpi"]["nprocs"])
    print(f"Total runs: {total_runs} -> {results_file}")

    with open(results_file, "w") as f:
        for nprocs in CONFIG["mpi"]["nprocs"]:
            for spec in all_specs:
                if not is_valid_combo(spec["solver"], nprocs):
                    continue
                result = run(spec, nprocs)
                log(f, result,
                    problem=spec["problem"], nprocs=nprocs,
                    **spec["mesh"], **spec["param"], **spec["solver"])


if __name__ == "__main__":
    main()
