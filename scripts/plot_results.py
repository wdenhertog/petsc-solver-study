import json
import matplotlib.pyplot as plt
from pathlib import Path

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
            "solve_time": [],
        },
    )

    configs[label]["n"].append(
        result["n"]
    )

    configs[label]["iterations"].append(
        result["iterations"]
    )

    configs[label]["solve_time"].append(result["solve_time"])

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

plt.figure(figsize=(8, 5))

for label, values in configs.items():
    plt.plot(
        values["n"],
        values["solve_time"],
        marker="o",
        label=label,
    )

plt.xscale("log")
plt.yscale("log")

plt.xlabel("Matrix size (n)")
plt.ylabel("Solve time (s)")
plt.title("Solve time vs Matrix Size")

plt.grid(True)
plt.legend()

output = (
    Path("../results/plots/")
    / "solve_time_vs_size.png"
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
