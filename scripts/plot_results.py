"""
plot_results.py

Generates diagnostic plots from a completed benchmark run's JSONL output.
Companion to rank_results.py — same loading/summary logic, different output.

Produces (by default, all of them; restrict via --plots):

  iterations_vs_dofs   Krylov iterations vs DOFs, one line per solver family.
                        Flat line = preconditioner is algorithmically optimal.
  success_rate         Bar chart of success rate (%) per solver family.
  pareto                solve_time vs memory scatter, Pareto frontier highlighted.
                        Axis defaults to peak_memory_bytes (per-rank max — "will it
                        fit"); pass --memory-metric total_memory_bytes for the
                        whole-job resource-cost view instead.
  setup_vs_solve        setup_time vs solve_time scatter — did the expensive
                        preconditioner setup (e.g. gamg coarse hierarchy) pay off?
  strong_scaling        solve_time vs nprocs at fixed problem size, one line per
                        solver family, with an ideal-scaling reference line.
  memory_scaling        peak_memory_bytes AND total_memory_bytes vs nprocs at fixed
                        problem size, one pair of lines per solver family. Peak
                        typically falls with more ranks (smaller local chunk per
                        rank); total often rises (per-rank overhead, redundant
                        coarse-grid/factorization structures) — this plot is what
                        makes that tradeoff visible, since neither pareto nor
                        strong_scaling shows it on its own.

Legend/series grouping
-----------------------
By default, series are grouped by the *algorithmic family* only
(snes_type + pc_type — the two choices that actually change the method),
not by every tuning knob (ksp_gmres_restart, pc_asm_overlap,
snes_linesearch_type, ...). Within each family, at every x-position, the
best-performing sub-configuration is plotted — i.e. each line answers
"what's the best this family can do", not "here is every hyperparameter
variant as a separate noisy line". Use --legend-by to override (e.g.
--legend-by pc_type ksp_type snes_type for full raw detail).

Problem-instance filtering
---------------------------
`n`, `nprocs`, and `lambda` (where applicable) identify *which problem
instance* was solved, not how. Plots that compare solver behavior at a
fixed problem size explicitly fix all of these (mixing e.g. lambda=1.0 and
lambda=6.8 into one line is mixing two different problem difficulties, not
solver noise — this was the cause of the previous "chaotic" Bratu plots).
`--lambda` lets you override which value is used; problems without a
lambda dimension (e.g. poisson) are unaffected.

Usage:
    python3 scripts/plot_results.py results/json/20260722T222536_b3ccfea
    python3 scripts/plot_results.py results/json/<run_id> --problem bratu
    python3 scripts/plot_results.py results/json/<run_id> --lambda 6.8
    python3 scripts/plot_results.py results/json/<run_id> --legend-by pc_type ksp_type snes_type
    python3 scripts/plot_results.py results/json/<run_id> --plots pareto strong_scaling
    python3 scripts/plot_results.py                                # auto-picks latest run

Output:
    results/plots/<run_id>/<problem>/<plot_name>.png
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_JSON_DIR = REPO_ROOT / "results" / "json"
RESULTS_PLOTS_DIR = REPO_ROOT / "results" / "plots"

# All flags that together fully specify *how* a problem was solved.
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

# Default legend grouping: the algorithmic-family columns only. Any column
# not present / all-NaN for a given problem (e.g. snes_type for poisson)
# is silently dropped per-row, so this generalizes across problem kinds
# without a kind-specific branch.
DEFAULT_LEGEND_COLS = ["snes_type", "pc_type"]

# Columns that identify *which problem instance* is being solved.
INSTANCE_FILTER_COLS = ["n", "nprocs", "lambda"]

ALL_PLOTS = ["iterations_vs_dofs", "success_rate", "pareto", "setup_vs_solve", "strong_scaling", "memory_scaling"]

MEMORY_METRIC_CHOICES = ["peak_memory_bytes", "total_memory_bytes"]


# ---------------------------------------------------------------------------
# Loading (mirrors rank_results.py)
# ---------------------------------------------------------------------------


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
                    print(f"WARNING: skipping malformed line in {f.name}", file=sys.stderr)

    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} rows from {len(files)} files in {run_dir}")
    return df


def make_label_fn(cols: list[str]):
    def label(row: pd.Series) -> str:
        parts = [f"{c}={row[c]}" for c in cols if c in row and pd.notna(row[c])]
        return "|".join(parts) if parts else "(no solver flags)"

    return label


def prep(df: pd.DataFrame, problem: str, legend_cols: list[str]) -> pd.DataFrame:
    sub = df[df["problem"] == problem].copy()
    if sub.empty:
        sys.exit(f"No rows for problem='{problem}' in this run.")
    sub["config_label"] = sub.apply(make_label_fn(legend_cols), axis=1)
    return sub


def pick_default_lambda(df: pd.DataFrame):
    """Median of the unique lambda values present, or None if this problem has no lambda dimension."""
    if "lambda" not in df.columns:
        return None
    vals = sorted(df["lambda"].dropna().unique())
    if not vals:
        return None
    return vals[len(vals) // 2]


def apply_instance_filter(df: pd.DataFrame, n=None, nprocs=None, lam=None) -> pd.DataFrame:
    """Fix problem-instance columns that are provided (not None) and present in df."""
    sub = df
    if n is not None and "n" in sub.columns:
        sub = sub[sub["n"] == n]
    if nprocs is not None and "nprocs" in sub.columns:
        sub = sub[sub["nprocs"] == nprocs]
    if lam is not None and "lambda" in sub.columns and sub["lambda"].notna().any():
        sub = sub[sub["lambda"] == lam]
    return sub


def best_of_per_group(df: pd.DataFrame, group_cols: list[str], x_col: str, metric: str) -> pd.DataFrame:
    """
    Collapse nuisance-flag variation: within each (group_cols..., x_col) cell,
    keep only the row with the best (lowest) metric value. This is what turns
    "every raw hyperparameter combo as its own noisy line" into "the best this
    algorithmic family can do at each x", and is also what removes the
    same-x/multiple-y zigzag when several sub-configs share an x value.
    """
    idx = df.groupby(group_cols + [x_col])[metric].idxmin()
    return df.loc[idx]


# ---------------------------------------------------------------------------
# Plot 1: Krylov iterations vs DOFs
# ---------------------------------------------------------------------------


def plot_iterations_vs_dofs(df: pd.DataFrame, problem: str, nprocs: int, lam, metric: str):
    sub = df[df["success"] == True]  # noqa: E712
    sub = apply_instance_filter(sub, nprocs=nprocs, lam=lam)
    if sub.empty:
        print(f"  [iterations_vs_dofs] no successful rows at nprocs={nprocs}, lambda={lam}, skipping")
        return

    best = best_of_per_group(sub, ["config_label"], "dofs", metric)

    fig, ax = plt.subplots(figsize=(8, 6))
    for label, grp in best.groupby("config_label"):
        grp = grp.sort_values("dofs")
        if grp["dofs"].nunique() < 2:
            continue  # need at least 2 mesh sizes to show a trend
        ax.plot(grp["dofs"], grp[metric], marker="o", label=label, alpha=0.85)

    ax.set_xscale("log")
    ax.set_xlabel("DOFs (log scale)")
    ax.set_ylabel(metric)
    lam_str = f", lambda={lam}" if lam is not None else ""
    ax.set_title(
        f"{problem}: best {metric} vs DOFs (nprocs={nprocs}{lam_str})\n"
        f"Flat = algorithmically optimal preconditioner. Each line: best sub-config per family."
    )
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    return fig


# ---------------------------------------------------------------------------
# Plot 2: Success rate / divergence bar chart
# ---------------------------------------------------------------------------


def plot_success_rate(df: pd.DataFrame, problem: str):
    sub = df.copy()
    if sub.empty:
        print("  [success_rate] no rows, skipping")
        return

    rates = (
        sub.groupby("config_label")["success"]
        .apply(lambda s: 100.0 * s.fillna(False).astype(bool).mean())
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(rates))))
    colors = ["#d62728" if v < 100 else "#2ca02c" for v in rates.values]
    ax.barh(rates.index, rates.values, color=colors)
    ax.set_xlabel("Success rate (%)")
    ax.set_xlim(0, 100)
    ax.set_title(
        f"{problem}: success rate by solver family\n(across all mesh sizes / nprocs / params / sub-tunings in this run)"
    )
    return fig


# ---------------------------------------------------------------------------
# Plot 3: Pareto frontier - solve_time vs peak_memory
# ---------------------------------------------------------------------------


def pareto_front(points: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    """Points minimizing both x_col and y_col. Returns the non-dominated subset, sorted by x."""
    pts = points.sort_values(x_col)
    front_rows = []
    best_y = float("inf")
    for _, row in pts.iterrows():
        if row[y_col] < best_y:
            front_rows.append(row)
            best_y = row[y_col]
    return pd.DataFrame(front_rows)


def plot_pareto(df: pd.DataFrame, problem: str, n: int, nprocs: int, lam, memory_metric: str):
    sub = df[df["success"] == True]  # noqa: E712
    sub = apply_instance_filter(sub, n=n, nprocs=nprocs, lam=lam)
    if sub.empty:
        print(f"  [pareto] no successful rows at n={n}, nprocs={nprocs}, lambda={lam}, skipping")
        return

    front = pareto_front(sub, "solve_time", memory_metric)

    fig, ax = plt.subplots(figsize=(8, 6))
    for label, grp in sub.groupby("config_label"):
        ax.scatter(grp["solve_time"], grp[memory_metric], alpha=0.5, s=25, label=label)
    ax.plot(front["solve_time"], front[memory_metric], "k--", alpha=0.6, label="Pareto frontier")
    ax.set_xlabel("solve_time (s)")
    ax.set_ylabel(memory_metric)
    lam_str = f", lambda={lam}" if lam is not None else ""
    metric_note = (
        "per-rank max — will it fit" if memory_metric == "peak_memory_bytes" else "whole-job sum — resource cost"
    )
    ax.set_title(f"{problem}: solve_time vs {memory_metric} ({metric_note})\n(n={n}, nprocs={nprocs}{lam_str})")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    return fig


# ---------------------------------------------------------------------------
# Plot 4: Setup time vs solve time
# ---------------------------------------------------------------------------


def plot_setup_vs_solve(df: pd.DataFrame, problem: str, n: int, nprocs: int, lam):
    sub = df[df["success"] == True]  # noqa: E712
    sub = apply_instance_filter(sub, n=n, nprocs=nprocs, lam=lam)
    if sub.empty:
        print(f"  [setup_vs_solve] no successful rows at n={n}, nprocs={nprocs}, lambda={lam}, skipping")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    for label, grp in sub.groupby("config_label"):
        ax.scatter(grp["setup_time"], grp["solve_time"], label=label, alpha=0.6, s=25)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("setup_time (s, log scale)")
    ax.set_ylabel("solve_time (s, log scale)")
    lam_str = f", lambda={lam}" if lam is not None else ""
    ax.set_title(f"{problem}: setup vs solve time (n={n}, nprocs={nprocs}{lam_str})")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    return fig


# ---------------------------------------------------------------------------
# Plot 5: Strong scaling
# ---------------------------------------------------------------------------


def plot_strong_scaling(df: pd.DataFrame, problem: str, n: int, lam):
    sub = df[df["success"] == True]  # noqa: E712
    sub = apply_instance_filter(sub, n=n, lam=lam)
    if sub.empty:
        print(f"  [strong_scaling] no successful rows at n={n}, lambda={lam}, skipping")
        return

    best = best_of_per_group(sub, ["config_label"], "nprocs", "solve_time")

    fig, ax = plt.subplots(figsize=(8, 6))
    for label, grp in best.groupby("config_label"):
        grp = grp.sort_values("nprocs")
        if grp["nprocs"].nunique() < 2:
            continue  # need at least 2 rank counts to show a trend
        ax.plot(grp["nprocs"], grp["solve_time"], marker="o", label=label, alpha=0.85)

    # Ideal scaling reference: T(1) / p, anchored at the fastest single-rank time present.
    baseline = sub[sub["nprocs"] == sub["nprocs"].min()]
    if not baseline.empty:
        t1 = baseline["solve_time"].min()
        p1 = baseline["nprocs"].min()
        p_range = sorted(sub["nprocs"].unique())
        ideal = [t1 * p1 / p for p in p_range]
        ax.plot(p_range, ideal, "k--", label="ideal scaling", alpha=0.6)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("nprocs (log2 scale)")
    ax.set_ylabel("solve_time (s, log scale)")
    lam_str = f", lambda={lam}" if lam is not None else ""
    ax.set_title(
        f"{problem}: strong scaling (n={n}{lam_str}, fixed problem size)\nEach line: best sub-config per family, per nprocs"
    )
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    return fig


# ---------------------------------------------------------------------------
# Plot 6: Memory scaling — peak (per-rank max) vs total (whole-job sum)
# ---------------------------------------------------------------------------


def plot_memory_scaling(df: pd.DataFrame, problem: str, n: int, lam):
    sub = df[df["success"] == True]  # noqa: E712
    sub = apply_instance_filter(sub, n=n, lam=lam)
    if sub.empty:
        print(f"  [memory_scaling] no successful rows at n={n}, lambda={lam}, skipping")
        return

    labels = sorted(sub["config_label"].unique())
    cmap = plt.get_cmap("tab10")
    color_for = {label: cmap(i % 10) for i, label in enumerate(labels)}

    peak_best = best_of_per_group(sub, ["config_label"], "nprocs", "peak_memory_bytes")
    total_best = best_of_per_group(sub, ["config_label"], "nprocs", "total_memory_bytes")

    fig, ax = plt.subplots(figsize=(8, 6))
    for label, grp in peak_best.groupby("config_label"):
        grp = grp.sort_values("nprocs")
        if grp["nprocs"].nunique() < 2:
            continue
        ax.plot(
            grp["nprocs"],
            grp["peak_memory_bytes"],
            marker="o",
            linestyle="-",
            color=color_for[label],
            alpha=0.85,
            label=f"{label} (peak, per-rank max)",
        )
    for label, grp in total_best.groupby("config_label"):
        grp = grp.sort_values("nprocs")
        if grp["nprocs"].nunique() < 2:
            continue
        ax.plot(
            grp["nprocs"],
            grp["total_memory_bytes"],
            marker="s",
            linestyle="--",
            color=color_for[label],
            alpha=0.85,
            label=f"{label} (total, whole-job sum)",
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("nprocs (log2 scale)")
    ax.set_ylabel("memory (bytes, log scale)")
    lam_str = f", lambda={lam}" if lam is not None else ""
    ax.set_title(
        f"{problem}: memory scaling (n={n}{lam_str}, fixed problem size)\n"
        f"Solid=peak (per-rank max, falling is good) — Dashed=total (whole-job sum, rising is overhead cost)"
    )
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def pick_default_n(df: pd.DataFrame) -> int:
    """Largest mesh size present at every nprocs value, so strong_scaling has full coverage."""
    counts = df.groupby("n")["nprocs"].nunique()
    full_coverage = counts[counts == df["nprocs"].nunique()]
    if full_coverage.empty:
        return int(df["n"].median())
    return int(full_coverage.index.max())


def save_fig(fig, out_dir: Path, name: str, plot_label: str):
    if fig is None:
        print(f"  [{plot_label}] no figure produced (see message above), skipping save")
        return
    fig.savefig(out_dir / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate diagnostic plots from a benchmark run.")
    parser.add_argument(
        "run_dir",
        nargs="?",
        type=Path,
        help="Path to results/json/<run_id>/. Defaults to the most " "recently modified folder under results/json/.",
    )
    parser.add_argument(
        "--problem",
        action="append",
        dest="problems",
        help="Problem(s) to plot. Repeatable. Defaults to all problems present.",
    )
    parser.add_argument(
        "--plots", nargs="+", choices=ALL_PLOTS, default=ALL_PLOTS, help="Which plots to generate. Defaults to all."
    )
    parser.add_argument(
        "--iter-metric",
        choices=["iterations", "outer_iterations"],
        default="iterations",
        help="Metric for iterations_vs_dofs.",
    )
    parser.add_argument(
        "--nprocs",
        type=int,
        default=1,
        help="Fixed nprocs for per-DOF/single-rank plots (iterations_vs_dofs, pareto, setup_vs_solve).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Fixed mesh size for plots that need one (pareto, setup_vs_solve, strong_scaling). "
        "Defaults to the largest n present at every nprocs value.",
    )
    parser.add_argument(
        "--lambda",
        type=float,
        default=None,
        dest="lam",
        help="Fixed lambda (for problems with a lambda param sweep, e.g. bratu). "
        "Defaults to the median lambda present. Ignored for problems without lambda.",
    )
    parser.add_argument(
        "--legend-by",
        nargs="+",
        default=None,
        help="Columns to build the legend/series grouping from. Defaults to "
        f"{DEFAULT_LEGEND_COLS} (algorithmic family only). Pass e.g. "
        "--legend-by pc_type ksp_type snes_type for full raw-config detail.",
    )
    parser.add_argument(
        "--memory-metric",
        choices=MEMORY_METRIC_CHOICES,
        default="peak_memory_bytes",
        help="Memory axis for the pareto plot. peak_memory_bytes (default) is the "
        "per-rank max — 'will it fit on a node'. total_memory_bytes is the "
        "whole-job sum across ranks — cluster resource cost. Only affects "
        "pareto; memory_scaling always plots both.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir or find_latest_run_dir()
    if not run_dir.is_dir():
        sys.exit(f"Not a directory: {run_dir}")

    legend_cols = args.legend_by or DEFAULT_LEGEND_COLS

    df = load_run(run_dir)
    problems = args.problems or sorted(df["problem"].dropna().unique())

    for problem in problems:
        print(f"\n=== {problem} ===")
        sub = prep(df, problem, legend_cols)
        n = args.n if args.n is not None else pick_default_n(sub)
        lam = args.lam if args.lam is not None else pick_default_lambda(sub)

        out_dir = RESULTS_PLOTS_DIR / run_dir.name / problem
        out_dir.mkdir(parents=True, exist_ok=True)

        if "iterations_vs_dofs" in args.plots:
            fig = plot_iterations_vs_dofs(sub, problem, nprocs=args.nprocs, lam=lam, metric=args.iter_metric)
            save_fig(fig, out_dir, "iterations_vs_dofs", "iterations_vs_dofs")
        if "success_rate" in args.plots:
            fig = plot_success_rate(sub, problem)
            save_fig(fig, out_dir, "success_rate", "success_rate")
        if "pareto" in args.plots:
            fig = plot_pareto(sub, problem, n=n, nprocs=args.nprocs, lam=lam, memory_metric=args.memory_metric)
            save_fig(fig, out_dir, "pareto", "pareto")
        if "setup_vs_solve" in args.plots:
            fig = plot_setup_vs_solve(sub, problem, n=n, nprocs=args.nprocs, lam=lam)
            save_fig(fig, out_dir, "setup_vs_solve", "setup_vs_solve")
        if "strong_scaling" in args.plots:
            fig = plot_strong_scaling(sub, problem, n=n, lam=lam)
            save_fig(fig, out_dir, "strong_scaling", "strong_scaling")
        if "memory_scaling" in args.plots:
            fig = plot_memory_scaling(sub, problem, n=n, lam=lam)
            save_fig(fig, out_dir, "memory_scaling", "memory_scaling")

        print(
            f"  wrote plots to {out_dir} (legend grouped by {legend_cols}"
            f"{f', lambda={lam}' if lam is not None else ''})"
        )


if __name__ == "__main__":
    main()
