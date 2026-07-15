import itertools
import json
import subprocess
from pathlib import Path

EXECUTABLE = "../cmake-build-debug-wsl/benchmark"

KSPS = [
    "cg",
    "gmres",
]

PCS = [
    "jacobi",
    "ilu",
    "gamg"
]

SIZES = [
    100,
    1_000,
    10_000,
    100_000,
    1_000_000
]

results = []

for ksp, pc, n in itertools.product(
    KSPS,
    PCS,
    SIZES,
):
    print(
        f"Running: ksp={ksp}, pc={pc}, n={n}"
    )

    command = [
        EXECUTABLE,
        "-ksp_type", ksp,
        "-pc_type", pc,
        "-n", str(n),
    ]

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
    )

    benchmark_result = json.loads(
        completed.stdout
    )

    results.append(
        benchmark_result
    )

output_file = Path(
    "../results/json/benchmark_results.json"
)

output_file.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with output_file.open("w") as f:
    json.dump(
        results,
        f,
        indent=4,
    )

print(
    f"Saved {len(results)} results to "
    f"{output_file}"
)
