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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_results as pr  # noqa: E402
import rank_results as rr  # noqa: E402

st.set_page_config(page_title="petsc-solver-study dashboard", layout="wide")


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


# ---------------------------------------------------------------------------
# Sidebar: run + instance selection
# ---------------------------------------------------------------------------

st.sidebar.header("Run")

run_dirs = list_run_dirs()
if not run_dirs:
    st.error(f"No run folders found under {pr.RESULTS_JSON_DIR}")
    st.stop()

run_dir = st.sidebar.selectbox("Run folder", run_dirs, format_func=lambda d: d.name, index=0)

df_raw = load_run_cached(str(run_dir), run_dir.stat().st_mtime)

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

st.sidebar.header("Problem instance")

nprocs_options = sorted(sub["nprocs"].dropna().unique().astype(int))
nprocs = st.sidebar.select_slider(
    "nprocs (for per-DOF / pareto / setup_vs_solve)", options=nprocs_options, value=nprocs_options[0]
)

n_options = sorted(sub["n"].dropna().unique().astype(int))
default_n = pr.pick_default_n(sub)
n = st.sidebar.select_slider(
    "n (mesh size, for pareto / setup_vs_solve / scaling plots)", options=n_options, value=default_n
)

lam = None
if "lambda" in sub.columns and sub["lambda"].notna().any():
    lam_options = sorted(sub["lambda"].dropna().unique())
    default_lam = pr.pick_default_lambda(sub)
    lam = st.sidebar.select_slider("lambda", options=lam_options, value=default_lam)

memory_metric = st.sidebar.radio(
    "Pareto memory axis",
    pr.MEMORY_METRIC_CHOICES,
    index=0,
    help="peak = per-rank max (will it fit). total = whole-job sum (resource cost).",
)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

st.title("petsc-solver-study — results dashboard")
st.caption(f"Run: `{run_dir.name}`  •  Problem: `{problem}`")

n_total = len(sub)
n_success = int((sub["success"] == True).sum())  # noqa: E712
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
    metric = st.radio("Metric", ["iterations", "outer_iterations"], horizontal=True, key="iter_metric")
    fig = pr.plot_iterations_vs_dofs(sub, problem, nprocs=nprocs, lam=lam, metric=metric)
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
    fig = pr.plot_pareto(sub, problem, n=n, nprocs=nprocs, lam=lam, memory_metric=memory_metric)
    if fig is not None:
        st.pyplot(fig)
    else:
        st.info("No successful rows for this selection.")

with tabs[3]:
    fig = pr.plot_setup_vs_solve(sub, problem, n=n, nprocs=nprocs, lam=lam)
    if fig is not None:
        st.pyplot(fig)
    else:
        st.info("No successful rows for this selection.")

with tabs[4]:
    fig = pr.plot_strong_scaling(sub, problem, n=n, lam=lam)
    if fig is not None:
        st.pyplot(fig)
    else:
        st.info("No successful rows for this selection.")

with tabs[5]:
    fig = pr.plot_memory_scaling(sub, problem, n=n, lam=lam)
    if fig is not None:
        st.pyplot(fig)
    else:
        st.info("No successful rows for this selection.")

with tabs[6]:
    top_n = st.slider("Top N per group", min_value=1, max_value=25, value=10)
    metric_choice = st.radio("Rank by", rr.METRIC_CHOICES, horizontal=True, key="rank_metric")
    ranked = rr.rank(sub, metric=metric_choice, top_n=top_n)
    st.dataframe(ranked, width="stretch")
    st.download_button(
        "Download this table as CSV",
        ranked.to_csv(index=False),
        file_name=f"{run_dir.name}_{problem}_top{top_n}_{metric_choice}.csv",
        mime="text/csv",
    )
