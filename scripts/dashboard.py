"""
dashboard.py

Interactive Streamlit dashboard for exploring a benchmark run — the
interactive complement to plot_results.py / rank_results.py, not a
replacement. Those two remain the tool for producing fixed, archived
PNG/CSV artifacts (reports, pipeline output); this dashboard is for
dragging sliders and seeing plots update on the fly.

All analysis logic (loading, filtering, best-of aggregation, the plot
functions themselves) is imported directly from plot_results.py and
rank_results.py — this file only adds widgets and calls them. There is
deliberately no second copy of e.g. best_of_per_group or config_label
logic here; if the underlying analysis changes, both the CLI scripts and
this dashboard pick it up automatically.

Run locally (e.g. on your WSL machine, after syncing results/json/ from
the cluster):

    streamlit run scripts/dashboard.py

Note: this starts a local web server (default http://localhost:8501).
Running it directly on a cluster login node works the same way but you'd
need to SSH-tunnel the port (`ssh -L 8501:localhost:8501 <host>`) to view
it from your laptop — usually simpler to just sync results/json/ locally
and run the dashboard there.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_results as pr
import rank_results as rr

st.set_page_config(page_title="petsc-solver-study dashboard", layout="wide")

RUN_CONFIG_NAME = "config.yaml"


@st.cache_data
def load_run_cached(run_dir_str: str, mtime: float) -> pd.DataFrame:
    """mtime is part of the cache key so editing/re-syncing a run folder invalidates the cache."""
    return pr.load_run(Path(run_dir_str))


def list_run_dirs() -> list[Path]:
    if not pr.RESULTS_JSON_DIR.exists():
        return []
    return sorted(
        [d for d in pr.RESULTS_JSON_DIR.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )


def find_run_config_path(run_dir: Path) -> Path | None:
    candidate = run_dir / RUN_CONFIG_NAME
    if candidate.exists() and candidate.is_file():
        return candidate
    return None


@st.cache_data
def load_yaml_cached(path_str: str, mtime: float) -> dict | None:
    with open(path_str) as fh:
        cfg = yaml.safe_load(fh)
    return cfg if isinstance(cfg, dict) else None


def sorted_unique(series: pd.Series) -> list:
    vals = [v for v in series.dropna().unique().tolist()]
    try:
        return sorted(vals)
    except TypeError:
        return sorted(vals, key=str)


def default_for_column(df: pd.DataFrame, col: str, options: list):
    if not options:
        return None
    if col == "n" and "n" in df.columns:
        return pr.pick_default_n(df)
    if pd.api.types.is_numeric_dtype(df[col]):
        return options[len(options) // 2]
    return options[0]


def get_problem_cfg(run_config: dict | None, problem: str) -> dict:
    if not run_config:
        return {}
    problems = run_config.get("problems")
    if not isinstance(problems, dict):
        return {}
    cfg = problems.get(problem)
    return cfg if isinstance(cfg, dict) else {}


def get_instance_keys(problem_cfg: dict, sub_df: pd.DataFrame) -> list[str]:
    keys = []
    mesh = problem_cfg.get("mesh_sweep", {}) if isinstance(problem_cfg, dict) else {}
    params = problem_cfg.get("param_sweep", {}) if isinstance(problem_cfg, dict) else {}

    if isinstance(mesh, dict):
        keys.extend([k for k in mesh.keys() if k in sub_df.columns])
    if isinstance(params, dict):
        keys.extend([k for k in params.keys() if k in sub_df.columns])

    if "nprocs" in sub_df.columns and "nprocs" not in keys:
        keys = ["nprocs"] + keys
    elif "nprocs" in sub_df.columns:
        keys = ["nprocs"] + [k for k in keys if k != "nprocs"]

    if not keys:
        fallback = [k for k in ["nprocs", "n"] if k in sub_df.columns]
        keys.extend(fallback)

    # If config metadata is missing/incomplete, auto-surface varying non-solver
    # columns as problem-instance controls.
    exclude_cols = {
        "problem",
        "config_label",
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
    exclude_cols.update(pr.SOLVER_FLAG_COLS)

    inferred = []
    for col in sub_df.columns:
        if col in exclude_cols or col in keys:
            continue
        if sub_df[col].dropna().nunique() > 1:
            inferred.append(col)
    keys.extend(sorted(inferred))

    return keys


def get_labels(run_config: dict | None, problem_cfg: dict) -> dict:
    labels = {}
    if isinstance(run_config, dict) and isinstance(run_config.get("labels"), dict):
        labels.update(run_config["labels"])
    if isinstance(problem_cfg, dict) and isinstance(problem_cfg.get("labels"), dict):
        labels.update(problem_cfg["labels"])
    return labels


# ---------------------------------------------------------------------------
# Sidebar: run + instance selection
# ---------------------------------------------------------------------------

st.sidebar.header("Run")

run_dirs = list_run_dirs()
if not run_dirs:
    st.error(f"No run folders found under {pr.RESULTS_JSON_DIR}")
    st.stop()

run_dir = st.sidebar.selectbox(
    "Run folder", run_dirs, format_func=lambda d: d.name, index=0
)

df_raw = load_run_cached(str(run_dir), run_dir.stat().st_mtime)
config_path = find_run_config_path(run_dir)
run_config = None
if config_path is not None:
    run_config = load_yaml_cached(str(config_path), config_path.stat().st_mtime)
    st.sidebar.caption(f"Config: {config_path.name}")
else:
    st.sidebar.caption(
        "Config: not found in run folder (using data-driven fallback controls)"
    )

problems = sorted(df_raw["problem"].dropna().unique())
problem = st.sidebar.selectbox("Problem", problems)

st.sidebar.header("Legend grouping")
legend_cols = st.sidebar.multiselect(
    "Group series by",
    options=pr.SOLVER_FLAG_COLS,
    default=pr.DEFAULT_LEGEND_COLS,
    help="Defaults to the algorithmic family only (snes_type + pc_type). "
    "Add more columns for finer-grained, noisier series.",
)
if not legend_cols:
    legend_cols = pr.DEFAULT_LEGEND_COLS

sub = pr.prep(df_raw, problem, legend_cols)
problem_cfg = get_problem_cfg(run_config, problem)
label_overrides = get_labels(run_config, problem_cfg)

st.sidebar.header("Problem instance")

instance_filters = {}
instance_keys = get_instance_keys(problem_cfg, sub)

for key in instance_keys:
    if key not in sub.columns:
        continue
    options = sorted_unique(sub[key])
    if not options:
        continue

    default_value = default_for_column(sub, key, options)
    if default_value not in options:
        default_value = options[0]

    label = label_overrides.get(key, key)
    if key == "nprocs" and len(options) > 1:
        value = st.sidebar.select_slider(
            f"{label} (for per-DOF / pareto / setup_vs_solve)",
            options=options,
            value=default_value,
        )
    elif len(options) > 1:
        value = st.sidebar.selectbox(
            label, options=options, index=options.index(default_value)
        )
    else:
        value = options[0]
        st.sidebar.caption(f"{label}: {value} (single value in this run)")

    instance_filters[key] = value

nprocs = int(instance_filters.get("nprocs", 1))
n = int(instance_filters.get("n", pr.pick_default_n(sub))) if "n" in sub.columns else 0

memory_metric = st.sidebar.radio(
    "Pareto memory axis",
    pr.MEMORY_METRIC_CHOICES,
    index=0,
    help="peak = per-rank max (will it fit). total = whole-job sum (resource cost). Display units are GiB.",
)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

st.title("petsc-solver-study — results dashboard")
st.caption(f"Run: `{run_dir.name}`  •  Problem: `{problem}`")

n_total = len(sub)
n_success = int((sub["success"] == True).sum())
n_error = int(sub["error"].notna().sum()) if "error" in sub else 0
n_timed_out = int(sub["timed_out"].fillna(False).sum()) if "timed_out" in sub else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total rows (this problem)", n_total)
c2.metric("Successful", n_success)
c3.metric("Errors", n_error)
c4.metric("Timed out", n_timed_out)

for col in ["git_sha", "git_dirty", "petsc_version"]:
    if col not in sub:
        continue
    distinct = sub[col].dropna().unique()
    if len(distinct) > 1:
        st.warning(
            f"Multiple distinct values for `{col}` in this run: {list(distinct)}. "
            f"Results may not be directly comparable across all rows."
        )

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

tab_names = [
    "iterations_vs_dofs",
    "success_rate",
    "pareto",
    "setup_vs_solve",
    "strong_scaling",
    "memory_scaling",
    "ranked table",
]
tabs = st.tabs(tab_names)

with tabs[0]:
    metric = st.radio(
        "Metric", ["iterations", "outer_iterations"], horizontal=True, key="iter_metric"
    )
    iter_filters = {k: v for k, v in instance_filters.items() if k != "n"}
    fig = pr.plot_iterations_vs_dofs(
        sub,
        problem,
        nprocs=nprocs,
        metric=metric,
        instance_filters=iter_filters,
    )
    if fig is not None:
        st.pyplot(fig)
    else:
        st.info("No successful rows for this selection.")

with tabs[1]:
    fig = pr.plot_success_rate(sub, problem)
    if fig is not None:
        st.pyplot(fig)
    else:
        st.info("No rows for this selection.")

with tabs[2]:
    fig = pr.plot_pareto(
        sub,
        problem,
        n=n,
        nprocs=nprocs,
        memory_metric=memory_metric,
        instance_filters=instance_filters,
    )
    if fig is not None:
        st.pyplot(fig)
    else:
        st.info("No successful rows for this selection.")

with tabs[3]:
    fig = pr.plot_setup_vs_solve(
        sub, problem, n=n, nprocs=nprocs, instance_filters=instance_filters
    )
    if fig is not None:
        st.pyplot(fig)
    else:
        st.info("No successful rows for this selection.")

with tabs[4]:
    fig = pr.plot_strong_scaling(sub, problem, n=n, instance_filters=instance_filters)
    if fig is not None:
        st.pyplot(fig)
    else:
        st.info("No successful rows for this selection.")

with tabs[5]:
    fig = pr.plot_memory_scaling(sub, problem, n=n, instance_filters=instance_filters)
    if fig is not None:
        st.pyplot(fig)
    else:
        st.info("No successful rows for this selection.")

with tabs[6]:
    top_n = st.slider("Top N per group", min_value=1, max_value=25, value=10)
    metric_choice = st.radio(
        "Rank by", rr.METRIC_CHOICES, horizontal=True, key="rank_metric"
    )
    rank_input = sub.drop(columns=["config_label"], errors="ignore")
    valid_for_grouping = rank_input[rank_input.get("success") == True].copy()
    if valid_for_grouping.empty:
        st.info("No successful rows for this selection.")
    else:
        group_cols = rr.infer_group_cols(valid_for_grouping, run_config)
        ranked = rr.rank(
            rank_input, metric=metric_choice, top_n=top_n, group_cols=group_cols
        )
        st.dataframe(ranked, width="stretch")
        st.download_button(
            "Download this table as CSV",
            ranked.to_csv(index=False),
            file_name=f"{run_dir.name}_{problem}_top{top_n}_{metric_choice}.csv",
            mime="text/csv",
        )
