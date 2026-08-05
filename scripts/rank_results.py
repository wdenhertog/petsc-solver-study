"""
rank_results.py

Aggregates a completed benchmark run's JSONL output into ranked
solver-configuration tables and writes them to results/csv/.

Usage:
    python3 scripts/rank_results.py results/json/20260722T222536_b3ccfea
    python3 scripts/rank_results.py results/json/20260722T222536_b3ccfea --top-n 5
    python3 scripts/rank_results.py results/json/20260722T222536_b3ccfea --metric iterations
    python3 scripts/rank_results.py                     # auto-picks the most recently
                                                          # modified folder under results/json/
    python3 scripts/rank_results.py <run_dir> --full     # also dump every valid row,
                                                          # not just the top N per group

Output:
    results/csv/<run_dir_name>.csv           top-N ranked rows per group
    results/csv/<run_dir_name>_full.csv       (only with --full) every valid row, unranked

A "group" is one (problem, nprocs, plus any problem-instance sweep
parameters from the run's config.yaml and dataset columns) combination.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_JSON_DIR = REPO_ROOT / "results" / "json"
RESULTS_CSV_DIR = REPO_ROOT / "results" / "csv"

SOLVER_FLAG_COLS = [
    "snes_type",
    "snes_linesearch_type",
    "ksp_type",
    "ksp_gmres_restart",
    "pc_type",
    "pc_jacobi_type",
    "pc_factor_levels",
    "pc_gamg_type",
    "pc_gamg_threshold",
    "pc_asm_overlap",
    "pc_factor_mat_solver_type",
]

NON_INSTANCE_COLS = {
    "problem",
    "success",
    "error",
    "timed_out",
    "converged_reason",
    "converged_reason_string",
    "iterations",
    "outer_iterations",
    "residual",
    "residual_norm",
    "setup_time",
    "solve_time",
    "dofs",
    "peak_memory_bytes",
    "total_memory_bytes",
    "git_sha",
    "git_dirty",
    "petsc_version",
    "backend",
    "assembly_mode",
}

# Lower is better for both currently-supported ranking metrics.
METRIC_CHOICES = ["solve_time", "iterations"]


def find_latest_run_dir() -> Path:
    candidates = [d for d in RESULTS_JSON_DIR.iterdir() if d.is_dir()]
    if not candidates:
        sys.exit(f"No run folders found under {RESULTS_JSON_DIR}")
    return max(candidates, key=lambda d: d.stat().st_mtime)


def load_run(run_dir: Path) -> pd.DataFrame:
    files = sorted(run_dir.glob("*.jsonl"))
    if not files:
        sys.exit(f"No .jsonl files found in {run_dir}")

    rows = []
    for f in files:
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    print(
                        f"WARNING: skipping malformed line in {f.name}", file=sys.stderr
                    )

    print(f"Loaded {len(rows)} rows from {len(files)} files in {run_dir}")
    return pd.DataFrame(rows)


def load_run_config(run_dir: Path) -> dict | None:
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        return None

    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)
    return cfg if isinstance(cfg, dict) else None


def infer_group_cols(valid: pd.DataFrame, run_config: dict | None) -> list[str]:
    group_cols = ["problem"]

    # Prefer instance-key definitions from the copied run config.
    config_keys = []
    if isinstance(run_config, dict):
        problems = run_config.get("problems")
        if isinstance(problems, dict):
            for problem_cfg in problems.values():
                if not isinstance(problem_cfg, dict):
                    continue
                mesh = problem_cfg.get("mesh_sweep", {})
                params = problem_cfg.get("param_sweep", {})
                if isinstance(mesh, dict):
                    config_keys.extend(mesh.keys())
                if isinstance(params, dict):
                    config_keys.extend(params.keys())

    for col in ["nprocs", "n"]:
        if col in valid.columns and col not in group_cols:
            group_cols.append(col)

    for col in config_keys:
        if col in valid.columns and col not in group_cols:
            group_cols.append(col)

    # Fallback: if config is missing or incomplete, include other varying
    # non-solver, non-metric columns as instance dimensions.
    excluded = set(SOLVER_FLAG_COLS) | NON_INSTANCE_COLS
    for col in valid.columns:
        if col in excluded or col in group_cols:
            continue
        if valid[col].dropna().nunique() > 1:
            group_cols.append(col)

    return group_cols


def summarize(df: pd.DataFrame) -> None:
    n_total = len(df)
    n_error = int(df["error"].notna().sum()) if "error" in df else 0
    n_timed_out = (
        int((df["timed_out"] == True).sum()) if "timed_out" in df.columns else 0
    )
    n_success = int((df.get("success") == True).sum())

    print("--- Run summary ---")
    print(f"total rows:      {n_total}")
    print(f"success=True:    {n_success}")
    print(f"error rows:      {n_error}  (of which timed_out: {n_timed_out})")

    for col in ["git_sha", "git_dirty", "petsc_version"]:
        if col not in df:
            continue
        distinct = df[col].dropna().unique()
        if len(distinct) > 1:
            print(
                f"WARNING: multiple distinct values for '{col}' in this run folder: {list(distinct)}"
            )
            print("         Results may not be directly comparable across all rows.")
        elif len(distinct) == 1:
            print(f"{col}: {distinct[0]}")
    print("-------------------")


def rank(
    df: pd.DataFrame, metric: str, top_n: int, group_cols: list[str]
) -> pd.DataFrame:
    valid = df[df.get("success") == True].copy()
    if valid.empty:
        sys.exit("No rows with success=True to rank.")
    if metric not in valid.columns:
        sys.exit(f"Metric '{metric}' not present in this dataset's columns.")

    valid = valid.sort_values(metric, ascending=True)
    valid["rank"] = valid.groupby(group_cols, dropna=False).cumcount() + 1
    ranked = valid[valid["rank"] <= top_n].copy()

    # Put identifying/ranking columns first, keep everything else after.
    front = ["rank"] + group_cols + [metric]
    rest = [c for c in ranked.columns if c not in front]
    ranked = ranked[front + rest]

    return ranked.sort_values(group_cols + ["rank"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(
        description="Rank benchmark results from a run folder."
    )
    parser.add_argument(
        "run_dir",
        nargs="?",
        type=Path,
        help="Path to results/json/<run_id>/. Defaults to the most "
        "recently modified folder under results/json/.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top configs to keep per problem-instance group (problem, nprocs, and sweep params).",
    )
    parser.add_argument(
        "--metric",
        choices=METRIC_CHOICES,
        default="solve_time",
        help="Ranking metric (lower is better for both options).",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Also write every valid (success=True) row, unranked, "
        "to <name>_full.csv for ad hoc analysis.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir or find_latest_run_dir()
    if not run_dir.is_dir():
        sys.exit(f"Not a directory: {run_dir}")

    df = load_run(run_dir)
    run_config = load_run_config(run_dir)
    summarize(df)

    valid = df[df.get("success") == True].copy()
    if valid.empty:
        sys.exit("No rows with success=True to rank.")
    group_cols = infer_group_cols(valid, run_config)
    print(f"Grouping columns: {group_cols}")

    ranked = rank(df, metric=args.metric, top_n=args.top_n, group_cols=group_cols)

    RESULTS_CSV_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_CSV_DIR / f"{run_dir.name}.csv"
    ranked.to_csv(out_path, index=False)
    print(f"\nWrote top-{args.top_n} rankings ({len(ranked)} rows) to {out_path}")

    if args.full:
        valid = df[df.get("success") == True]
        full_path = RESULTS_CSV_DIR / f"{run_dir.name}_full.csv"
        valid.to_csv(full_path, index=False)
        print(f"Wrote full valid dataset ({len(valid)} rows) to {full_path}")


if __name__ == "__main__":
    main()
