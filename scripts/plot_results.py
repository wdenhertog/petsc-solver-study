import json
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS_FILE = (
    Path("../results/json/benchmark_results.json")
)

with RESULTS_FILE.open() as f:
    data = json.load(f)

configs = {}

for result in data:
    label = (
        f"{result['ksp']} + "
        f"{result['pc']}"
    )

    configs.setdefault(
        label,
        {
            "n": [],
            "iterations": [],
        },
    )

    configs[label]["n"].append(
        result["n"]
    )

    configs[label]["iterations"].append(
        result["iterations"]
    )

plt.figure(figsize=(8, 5))

for label, values in configs.items():
    plt.plot(
        values["n"],
        values["iterations"],
        marker="o",
        label=label,
    )

plt.xscale("log")
plt.yscale("log")

plt.xlabel("Matrix size (n)")
plt.ylabel("Iterations")
plt.title("Iterations vs Matrix Size")

plt.grid(True)
plt.legend()

output = (
    Path("../results/plots/")
    / "iterations_vs_size.png"
)

output.parent.mkdir(
    exist_ok=True,
    parents=True,
)

plt.savefig(
    output,
    dpi=300,
    bbox_inches="tight",
)

plt.show()
