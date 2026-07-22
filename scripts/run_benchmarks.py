import itertools
import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent  # scripts/ -> repo root
BENCHMARK_BIN = REPO_ROOT / "bin" / "benchmark"
RESULTS_FILE = REPO_ROOT / "results" / "json" / "benchmarks.jsonl"

RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

mesh_sizes = [32, 64]
gmres_restarts = [30, 50]
jacobi_types = ["diagonal"]
ilu_levels = [0, 1]
gamg_types = ["agg"]


def build_ksp_configs():
    configs = [{"ksp": "cg", "extra": {}}]
    for restart in gmres_restarts:
        configs.append({"ksp": "gmres", "extra": {"ksp_gmres_restart": restart}})
    return configs


def build_pc_configs():
    configs = []
    for jtype in jacobi_types:
        configs.append({"pc": "jacobi", "extra": {"pc_jacobi_type": jtype}})
    for levels in ilu_levels:
        configs.append({"pc": "ilu", "extra": {"pc_factor_levels": levels}})
    for gamg_type in gamg_types:
        configs.append({"pc": "gamg", "extra": {"pc_gamg_type": gamg_type}})
    return configs


def build_direct_configs():
    return [
        {"ksp": "preonly", "pc": "lu", "extra": {}},
        {"ksp": "preonly", "pc": "cholesky", "extra": {}},
    ]


def run(problem, n, pc_config, ksp_config, max_it=3000):
    cmd = [str(BENCHMARK_BIN), "-problem", problem,
           "-n", str(n),
           "-ksp_type", ksp_config["ksp"], "-pc_type", pc_config["pc"],
           "-ksp_max_it", str(max_it)]
    for flag, value in {**ksp_config["extra"], **pc_config["extra"]}.items():
        cmd += [f"-{flag}", str(value)]

    out = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return json.loads(out.stdout)
    except json.JSONDecodeError:
        return {"error": out.stderr.strip() or f"exit code {out.returncode}"}


def log(f, result, **context):
    record = {**context, **result}
    f.write(json.dumps(record) + "\n")
    f.flush()
    # ctx_str = " ".join(f"{k}={v}" for k, v in context.items())
    # if "error" in result:
    #     print(f"{ctx_str}: ERROR - {result['error']}")
    # else:
    #     print(f"{ctx_str}: {result['iterations']} its, "
    #           f"setup={result.get('setup_time', 0):.4f}s, solve={result['solve_time']:.4f}s, "
    #           f"success={result['success']}")


ksp_configs = build_ksp_configs()
pc_configs = build_pc_configs()
direct_configs = build_direct_configs()

iterative_runs = len(mesh_sizes) * len(ksp_configs) * len(pc_configs)
direct_runs = len(mesh_sizes) * len(direct_configs)
print(f"Total runs: {iterative_runs + direct_runs}")

with open(RESULTS_FILE, "w") as f:
    for problem, n, pc_config, ksp_config in itertools.product(["poisson"], mesh_sizes, pc_configs, ksp_configs):
        result = run(problem, n, pc_config, ksp_config)
        log(f, result,
            problem=problem, n=n,
            pc=pc_config["pc"], **pc_config["extra"],
            ksp=ksp_config["ksp"], **ksp_config["extra"])

    for problem, n, direct in itertools.product(["poisson"], mesh_sizes, direct_configs):
        pc_config = {"pc": direct["pc"], "extra": direct["extra"]}
        ksp_config = {"ksp": direct["ksp"], "extra": {}}
        result = run(problem, n, pc_config, ksp_config)
        log(f, result, problem=problem, n=n, pc=direct["pc"], ksp=direct["ksp"])
