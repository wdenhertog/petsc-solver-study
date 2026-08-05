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
`n`, `nprocs`, and any problem-specific sweep parameters identify *which
problem instance* was solved, not how. If a run varies extra problem
parameters (for example a nonlinear continuation parameter), fix those with
`--instance-filter key=value` so plots compare solvers on the same instance.

Usage:
    python3 scripts/plot_results.py results/json/20260722T222536_b3ccfea
    python3 scripts/plot_results.py results/json/<run_id> --problem bratu
    python3 scripts/plot_results.py results/json/<run_id> --instance-filter continuation=6.8
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

ALL_PLOTS = [
    "iterations_vs_dofs",
    "success_rate",
    "pareto",
    "setup_vs_solve",
    "strong_scaling",
    "memory_scaling",
]

MEMORY_METRIC_CHOICES = ["peak_memory_bytes", "total_memory_bytes"]
BYTES_PER_GIB = float(1024**3)


def bytes_to_gib(series: pd.Series) -> pd.Series:
    return series / BYTES_PER_GIB


def memory_metric_label(metric: str) -> str:
    if metric == "peak_memory_bytes":
        return "peak memory (GiB)"
    if metric == "total_memory_bytes":
        return "total memory (GiB)"
    return f"{metric} (GiB)"


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
                    print(
                        f"WARNING: skipping malformed line in {f.name}", file=sys.stderr
                    )

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


def apply_instance_filter(
    df: pd.DataFrame, n=None, nprocs=None, instance_filters: dict | None = None
) -> pd.DataFrame:
    """Fix problem-instance columns that are provided (not None) and present in df."""
    sub = df
    if n is not None and "n" in sub.columns:
        sub = sub[sub["n"] == n]
    if nprocs is not None and "nprocs" in sub.columns:
        sub = sub[sub["nprocs"] == nprocs]
    if instance_filters:
        for col, value in instance_filters.items():
            if value is None:
                continue
            if col in {"n", "nprocs"}:
                continue
            if col in sub.columns:
                sub = sub[sub[col] == value]
    return sub


def format_instance_context(
    instance_filters: dict | None, exclude_keys: set[str] | None = None
) -> str:
    if not instance_filters:
        return ""
    exclude = exclude_keys or set()
    parts = []
    for key in sorted(instance_filters.keys()):
        if key in exclude:
            continue
        value = instance_filters[key]
        if value is None:
            continue
        parts.append(f"{key}={value}")
    return f", {', '.join(parts)}" if parts else ""


def best_of_per_group(
    df: pd.DataFrame, group_cols: list[str], x_col: str, metric: str
) -> pd.DataFrame:
    """
    Collapse nuisance-flag variation: within each (group_cols..., x_col) cell,
    keep only the row with the best (lowest) metric value. This is what turns
    "every raw hyperparameter combo as its own noisy line" into "the best this
    algorithmic family can do at each x", and is also what removes the
    same-x/multiple-y zigzag when several sub-configs share an x value.
    """
    idx = df.groupby(group_cols + [x_col])[metric].idxmin()
    return df.loc[idx]


def compare_backends(
    df: pd.DataFrame,
    problem: str,
    join_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compare cpp and python rows for one problem on matching solver/instance keys.
    Returns one row per matched pair with timing/iteration deltas.
    """
    if "backend" not in df.columns:
        return pd.DataFrame()

    problem_rows = df[df["problem"] == problem].copy()
    if problem_rows.empty:
        return pd.DataFrame()

    cpp = problem_rows[problem_rows["backend"] == "cpp"].copy()
    py = problem_rows[problem_rows["backend"] == "python"].copy()
    if cpp.empty or py.empty:
        return pd.DataFrame()

    join_keys = join_cols or (SOLVER_FLAG_COLS + ["n", "nprocs"])
    join_keys = [c for c in join_keys if c in cpp.columns and c in py.columns]

    # Keep mode visible in output even if not used as a join key.
    if "assembly_mode" in py.columns and "assembly_mode" not in join_keys:
        py = py.rename(columns={"assembly_mode": "assembly_mode_python"})
    if "assembly_mode" in cpp.columns and "assembly_mode" not in join_keys:
        cpp = cpp.rename(columns={"assembly_mode": "assembly_mode_cpp"})

    merged = cpp.merge(py, on=join_keys, how="inner", suffixes=("_cpp", "_python"))
    if merged.empty:
        return merged
    merged["problem"] = problem

    if "solve_time_cpp" in merged.columns and "solve_time_python" in merged.columns:
        merged["solve_time_diff"] = (
            merged["solve_time_python"] - merged["solve_time_cpp"]
        )
        merged["solve_time_ratio"] = merged["solve_time_python"] / merged[
            "solve_time_cpp"
        ].replace({0: pd.NA})
    if "iterations_cpp" in merged.columns and "iterations_python" in merged.columns:
        merged["iterations_diff"] = (
            merged["iterations_python"] - merged["iterations_cpp"]
        )
    if (
        "converged_reason_cpp" in merged.columns
        and "converged_reason_python" in merged.columns
    ):
        merged["converged_reason_match"] = (
            merged["converged_reason_python"] == merged["converged_reason_cpp"]
        )
    if "success_cpp" in merged.columns and "success_python" in merged.columns:
        merged["success_match"] = merged["success_python"] == merged["success_cpp"]

    front_cols = ["problem"]
    front_cols += [c for c in join_keys if c in merged.columns]
    for col in ["assembly_mode_cpp", "assembly_mode_python"]:
        if col in merged.columns and col not in front_cols:
            front_cols.append(col)
    rest = [c for c in merged.columns if c not in front_cols]
    return merged[front_cols + rest]


# ---------------------------------------------------------------------------
# Plot 1: Krylov iterations vs DOFs
# ---------------------------------------------------------------------------


def plot_iterations_vs_dofs(
    df: pd.DataFrame,
    problem: str,
    nprocs: int,
    metric: str,
    instance_filters: dict | None = None,
):
    sub = df[df["success"] == True]
    sub = apply_instance_filter(sub, nprocs=nprocs, instance_filters=instance_filters)
    if sub.empty:
        ctx = format_instance_context(instance_filters, exclude_keys={"n", "nprocs"})
        print(
            f"  [iterations_vs_dofs] no successful rows at nprocs={nprocs}{ctx}, skipping"
        )
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
    inst_str = format_instance_context(instance_filters, exclude_keys={"n", "nprocs"})
    ax.set_title(
        f"{problem}: best {metric} vs DOFs (nprocs={nprocs}{inst_str})\n"
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


def plot_pareto(
    df: pd.DataFrame,
    problem: str,
    n: int,
    nprocs: int,
    memory_metric: str,
    instance_filters: dict | None = None,
):
    sub = df[df["success"] == True]
    sub = apply_instance_filter(
        sub, n=n, nprocs=nprocs, instance_filters=instance_filters
    )
    if sub.empty:
        ctx = format_instance_context(instance_filters, exclude_keys={"n", "nprocs"})
        print(f"  [pareto] no successful rows at n={n}, nprocs={nprocs}{ctx}, skipping")
        return

    front = pareto_front(sub, "solve_time", memory_metric)

    fig, ax = plt.subplots(figsize=(8, 6))
    for label, grp in sub.groupby("config_label"):
        ax.scatter(
            grp["solve_time"],
            bytes_to_gib(grp[memory_metric]),
            alpha=0.5,
            s=25,
            label=label,
        )
    ax.plot(
        front["solve_time"],
        bytes_to_gib(front[memory_metric]),
        "k--",
        alpha=0.6,
        label="Pareto frontier",
    )
    ax.set_xlabel("solve_time (s)")
    ax.set_ylabel(memory_metric_label(memory_metric))
    inst_str = format_instance_context(instance_filters, exclude_keys={"n", "nprocs"})
    metric_note = (
        "per-rank max — will it fit"
        if memory_metric == "peak_memory_bytes"
        else "whole-job sum — resource cost"
    )
    ax.set_title(
        f"{problem}: solve_time vs {memory_metric_label(memory_metric)} ({metric_note})\n"
        f"(n={n}, nprocs={nprocs}{inst_str})"
    )
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    return fig


# ---------------------------------------------------------------------------
# Plot 4: Setup time vs solve time
# ---------------------------------------------------------------------------


def plot_setup_vs_solve(
    df: pd.DataFrame,
    problem: str,
    n: int,
    nprocs: int,
    instance_filters: dict | None = None,
):
    sub = df[df["success"] == True]
    sub = apply_instance_filter(
        sub, n=n, nprocs=nprocs, instance_filters=instance_filters
    )
    if sub.empty:
        ctx = format_instance_context(instance_filters, exclude_keys={"n", "nprocs"})
        print(
            f"  [setup_vs_solve] no successful rows at n={n}, nprocs={nprocs}{ctx}, skipping"
        )
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    for label, grp in sub.groupby("config_label"):
        ax.scatter(grp["setup_time"], grp["solve_time"], label=label, alpha=0.6, s=25)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("setup_time (s, log scale)")
    ax.set_ylabel("solve_time (s, log scale)")
    inst_str = format_instance_context(instance_filters, exclude_keys={"n", "nprocs"})
    ax.set_title(f"{problem}: setup vs solve time (n={n}, nprocs={nprocs}{inst_str})")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    return fig


# ---------------------------------------------------------------------------
# Plot 5: Strong scaling
# ---------------------------------------------------------------------------


def plot_strong_scaling(
    df: pd.DataFrame, problem: str, n: int, instance_filters: dict | None = None
):
    sub = df[df["success"] == True]
    sub = apply_instance_filter(sub, n=n, instance_filters=instance_filters)
    if sub.empty:
        ctx = format_instance_context(instance_filters, exclude_keys={"n", "nprocs"})
        print(f"  [strong_scaling] no successful rows at n={n}{ctx}, skipping")
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
    inst_str = format_instance_context(instance_filters, exclude_keys={"n", "nprocs"})
    ax.set_title(
        f"{problem}: strong scaling (n={n}{inst_str}, fixed problem size)\n"
        f"Each line: best sub-config per family, per nprocs"
    )
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    return fig


# ---------------------------------------------------------------------------
# Plot 6: Memory scaling — peak (per-rank max) vs total (whole-job sum)
# ---------------------------------------------------------------------------


def plot_memory_scaling(
    df: pd.DataFrame, problem: str, n: int, instance_filters: dict | None = None
):
    sub = df[df["success"] == True]
    sub = apply_instance_filter(sub, n=n, instance_filters=instance_filters)
    if sub.empty:
        ctx = format_instance_context(instance_filters, exclude_keys={"n", "nprocs"})
        print(f"  [memory_scaling] no successful rows at n={n}{ctx}, skipping")
        return

    labels = sorted(sub["config_label"].unique())
    cmap = plt.get_cmap("tab10")
    color_for = {label: cmap(i % 10) for i, label in enumerate(labels)}

    peak_best = best_of_per_group(sub, ["config_label"], "nprocs", "peak_memory_bytes")
    total_best = best_of_per_group(
        sub, ["config_label"], "nprocs", "total_memory_bytes"
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    for label, grp in peak_best.groupby("config_label"):
        grp = grp.sort_values("nprocs")
        if grp["nprocs"].nunique() < 2:
            continue
        ax.plot(
            grp["nprocs"],
            bytes_to_gib(grp["peak_memory_bytes"]),
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
            bytes_to_gib(grp["total_memory_bytes"]),
            marker="s",
            linestyle="--",
            color=color_for[label],
            alpha=0.85,
            label=f"{label} (total, whole-job sum)",
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("nprocs (log2 scale)")
    ax.set_ylabel("memory (GiB, log scale)")
    inst_str = format_instance_context(instance_filters, exclude_keys={"n", "nprocs"})
    ax.set_title(
        f"{problem}: memory scaling (n={n}{inst_str}, fixed problem size)\n"
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
    parser = argparse.ArgumentParser(
        description="Generate diagnostic plots from a benchmark run."
    )
    parser.add_argument(
        "run_dir",
        nargs="?",
        type=Path,
        help="Path to results/json/<run_id>/. Defaults to the most "
        "recently modified folder under results/json/.",
    )
    parser.add_argument(
        "--problem",
        action="append",
        dest="problems",
        help="Problem(s) to plot. Repeatable. Defaults to all problems present.",
    )
    parser.add_argument(
        "--plots",
        nargs="+",
        choices=ALL_PLOTS,
        default=ALL_PLOTS,
        help="Which plots to generate. Defaults to all.",
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
        "--instance-filter",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Repeatable instance filter for problem-specific sweep parameters, e.g. "
        "--instance-filter continuation=6.8 --instance-filter Re=1000.",
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
        "whole-job sum across ranks — cluster resource cost. Values are displayed "
        "as GiB in the plots. Only affects pareto; memory_scaling always plots both.",
    )
    args = parser.parse_args()

    raw_instance_filters = {}
    for item in args.instance_filter:
        if "=" not in item:
            parser.error(f"Invalid --instance-filter '{item}'. Expected KEY=VALUE.")
        key, raw = item.split("=", 1)
        key = key.strip()
        raw = raw.strip()
        if not key:
            parser.error(f"Invalid --instance-filter '{item}'. Empty key.")
        raw_instance_filters[key] = raw

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

        instance_filters = {}
        for key, raw in raw_instance_filters.items():
            if key in sub.columns:
                col = sub[key]
                if pd.api.types.is_integer_dtype(col):
                    instance_filters[key] = int(raw)
                elif pd.api.types.is_float_dtype(col):
                    instance_filters[key] = float(raw)
                elif pd.api.types.is_bool_dtype(col):
                    instance_filters[key] = raw.lower() in {"1", "true", "yes", "on"}
                else:
                    instance_filters[key] = raw
            else:
                instance_filters[key] = raw

        out_dir = RESULTS_PLOTS_DIR / run_dir.name / problem
        out_dir.mkdir(parents=True, exist_ok=True)

        if "iterations_vs_dofs" in args.plots:
            fig = plot_iterations_vs_dofs(
                sub,
                problem,
                nprocs=args.nprocs,
                metric=args.iter_metric,
                instance_filters=instance_filters,
            )
            save_fig(fig, out_dir, "iterations_vs_dofs", "iterations_vs_dofs")
        if "success_rate" in args.plots:
            fig = plot_success_rate(sub, problem)
            save_fig(fig, out_dir, "success_rate", "success_rate")
        if "pareto" in args.plots:
            fig = plot_pareto(
                sub,
                problem,
                n=n,
                nprocs=args.nprocs,
                memory_metric=args.memory_metric,
                instance_filters=instance_filters,
            )
            save_fig(fig, out_dir, "pareto", "pareto")
        if "setup_vs_solve" in args.plots:
            fig = plot_setup_vs_solve(
                sub, problem, n=n, nprocs=args.nprocs, instance_filters=instance_filters
            )
            save_fig(fig, out_dir, "setup_vs_solve", "setup_vs_solve")
        if "strong_scaling" in args.plots:
            fig = plot_strong_scaling(
                sub, problem, n=n, instance_filters=instance_filters
            )
            save_fig(fig, out_dir, "strong_scaling", "strong_scaling")
        if "memory_scaling" in args.plots:
            fig = plot_memory_scaling(
                sub, problem, n=n, instance_filters=instance_filters
            )
            save_fig(fig, out_dir, "memory_scaling", "memory_scaling")

        inst_str = format_instance_context(
            instance_filters, exclude_keys={"n", "nprocs"}
        )
        print(f"  wrote plots to {out_dir} (legend grouped by {legend_cols}{inst_str})")


if __name__ == "__main__":
    main()
