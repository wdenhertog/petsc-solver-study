#!/usr/bin/env python3
"""Compare burn-injury screening runs against the MUMPS reference.

Usage:
    python3 scripts/analyze_burn_screening.py results/json/<RUN_ID>/merged/all_results.jsonl
    python3 scripts/analyze_burn_screening.py a.jsonl b.jsonl --tol 5e-3 --csv summary.csv

For every (case, level, nprocs) group the reference is the direct MUMPS run
(ksp preonly, pc lu, mumps, no BLR) with Jacobian lag 1. If the group has no
reference, one with the same case/level and any rank count is used.

A run is "valid" when it succeeded AND every functional matches the reference
within --tol (max relative difference). With adaptive stepping the step
sequence depends on the solver, so differences of the order of the TS
tolerance are expected; compare the MUMPS lag-2 row (same solver, different
path) to see the noise floor.

Only the Python standard library is used.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

FUNCTIONALS = [
    "int_N", "int_M", "int_rho", "int_c",
    "l2sq_N", "l2sq_M", "l2sq_rho", "l2sq_c", "l2sq_v", "l2sq_eps",
]


def load(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def is_direct_mumps(r):
    return (
        r.get("ksp_type") == "preonly"
        and r.get("pc_type") == "lu"
        and r.get("pc_factor_mat_solver_type") == "mumps"
    )


def solver_label(r):
    pc = r.get("pc_type", "?")
    if is_direct_mumps(r):
        return "mumps-direct"
    if pc == "lu" and r.get("mat_mumps_icntl_35"):
        return f"gmres({r.get('ksp_rtol')})+mumps-blr"
    sub = r.get("sub_pc_type") or r.get("pc_hypre_type") or ""
    return f"gmres({r.get('ksp_rtol')})+{pc}" + (f"/{sub}" if sub else "")


def failure_reason(r):
    err = str(r.get("error") or r.get("converged_reason_string") or "")
    if r.get("timed_out") or "timed out" in err:
        return "timeout"
    if "Killed" in err or "signal 9" in err or "out of memory" in err.lower():
        return "oom"
    if "DIVERGED_NONLINEAR_SOLVE" in err:
        return "newton-diverged"
    if "DIVERGED" in err:
        return "diverged"
    if "error" in r:
        return "error"
    return "failed"


def events(r):
    return (r.get("details") or {}).get("petsc_events") or {}


def max_rel_diff(r, ref):
    fr = (r.get("details") or {}).get("functionals") or {}
    ff = (ref.get("details") or {}).get("functionals") or {}
    worst, worst_key = 0.0, ""
    for k in FUNCTIONALS:
        if k in fr and k in ff:
            d = abs(fr[k] - ff[k]) / max(abs(ff[k]), 1e-30)
            if d > worst:
                worst, worst_key = d, k
    return worst, worst_key


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--tol", type=float, default=5e-3, help="max relative functional difference (default 5e-3)")
    ap.add_argument("--csv", type=Path, default=None, help="write the full table as CSV")
    args = ap.parse_args()

    rows = load(args.files)
    if not rows:
        sys.exit("no rows found")

    groups = defaultdict(list)
    for r in rows:
        groups[(r.get("burn_case"), r.get("burn_level"), r.get("nprocs"))].append(r)

    def find_ref(case, level, nprocs):
        cands = [
            r for r in rows
            if r.get("burn_case") == case and r.get("burn_level") == level
            and is_direct_mumps(r) and r.get("success") and r.get("snes_lag_jacobian") == 1
            and (r.get("details") or {}).get("functionals")
        ]
        same = [r for r in cands if r.get("nprocs") == nprocs]
        return (same or cands or [None])[0]

    table = []
    for key in sorted(groups, key=lambda k: tuple(str(x) for x in k)):
        case, level, nprocs = key
        ref = find_ref(case, level, nprocs)
        ref_time = ref["solve_time"] if ref else None
        for r in sorted(groups[key], key=lambda r: (solver_label(r), r.get("snes_lag_jacobian") or 0)):
            ev = events(r)
            ok = bool(r.get("success"))
            d = r.get("details") or {}
            diff, diff_key = (max_rel_diff(r, ref) if ok and ref is not None else (None, ""))
            if not ok:
                verdict = failure_reason(r)
            elif ref is None:
                verdict = "no-ref"
            elif diff is not None and diff > args.tol:
                verdict = "inaccurate"
            else:
                verdict = "valid"
            t = r.get("solve_time") if ok else None
            nit = r.get("n_snes_iterations_total") or 0
            kit = r.get("n_ksp_iterations_total") or 0
            table.append({
                "case": case, "level": level, "nprocs": nprocs,
                "solver": solver_label(r), "lag": r.get("snes_lag_jacobian"),
                "verdict": verdict,
                "time_s": t,
                "speedup": (ref_time / t) if (t and ref_time and verdict == "valid") else None,
                "newton": nit if ok else None,
                "ksp_per_newton": (kit / nit) if (ok and nit) else None,
                "rejected": d.get("n_rejected_steps") if ok else None,
                "pcsetup_s": ev.get("PCSetUp") if ok else None,
                "kspsolve_s": ev.get("KSPSolve") if ok else None,
                "jac_assembly_s": d.get("t_jacobian_assembly") if ok else None,
                "max_rel_diff": diff,
                "worst_functional": diff_key,
                "peak_mem_gb": (r.get("total_memory_bytes") or 0) / 1e9 if ok else None,
            })

    def fmt(v, spec):
        return "-" if v is None else format(v, spec)

    cols = ["solver", "lag", "verdict", "time_s", "speedup", "newton", "ksp_per_newton",
            "rejected", "pcsetup_s", "kspsolve_s", "jac_assembly_s", "max_rel_diff", "peak_mem_gb"]
    specs = {"time_s": ".0f", "speedup": ".2f", "newton": "d", "ksp_per_newton": ".1f",
             "rejected": "d", "pcsetup_s": ".0f", "kspsolve_s": ".0f",
             "jac_assembly_s": ".0f", "max_rel_diff": ".1e", "peak_mem_gb": ".1f"}

    current = None
    for t in table:
        grp = (t["case"], t["level"], t["nprocs"])
        if grp != current:
            current = grp
            print(f"\n=== {t['case']}  L{t['level']}  {t['nprocs']} rank(s) ===")
            print("  ".join(f"{c:>14}" if c != "solver" else f"{c:<26}" for c in cols))
        cells = []
        for c in cols:
            v = t[c]
            if c == "solver":
                cells.append(f"{v:<26}")
            elif c in specs:
                cells.append(f"{fmt(v, specs[c]):>14}")
            else:
                cells.append(f"{str(v):>14}")
        print("  ".join(cells))

    # Robustness: valid runs per solver/lag across all levels and rank counts.
    print("\n=== Robustness (valid runs / total, per solver and lag) ===")
    agg = defaultdict(lambda: [0, 0, defaultdict(int)])
    for t in table:
        a = agg[(t["solver"], t["lag"])]
        a[1] += 1
        if t["verdict"] == "valid":
            a[0] += 1
        else:
            a[2][t["verdict"]] += 1
    for (solver, lag), (n_ok, n_all, bad) in sorted(agg.items(), key=lambda kv: (-kv[1][0] / kv[1][1], str(kv[0]))):
        why = ", ".join(f"{k}:{v}" for k, v in sorted(bad.items()))
        print(f"  {solver:<26} lag={lag}  {n_ok}/{n_all}  {why}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
            w.writeheader()
            w.writerows(table)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
