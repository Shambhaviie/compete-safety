import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


ROOT = Path("mechinterp/results")
OUT = Path("figures")
OUT.mkdir(exist_ok=True)

with open(ROOT / "multi_random_control_results.json") as f:
    data = json.load(f)


# Same 153-prompt population for learned and random ablations
N = 153

layers = [9, 11]

fig, ax = plt.subplots(figsize=(9, 6))

rng = np.random.default_rng(42)

for i, layer in enumerate(layers):
    result = data[str(layer)]["ablation"]

    learned = result["learned_flips"] / N * 100
    random_flips = np.array(result["random_flips"]) / N * 100

    x_random = np.full(len(random_flips), i, dtype=float)
    x_random += rng.uniform(-0.08, 0.08, size=len(random_flips))

    # Random directions
    ax.scatter(
        x_random,
        random_flips,
        s=55,
        alpha=0.75,
        label="Random direction" if i == 0 else None,
        zorder=2,
    )

    # Random mean
    ax.hlines(
        random_flips.mean(),
        i - 0.14,
        i + 0.14,
        linewidth=2.5,
        label="Random mean" if i == 0 else None,
        zorder=3,
    )

    # Learned direction
    ax.scatter(
        i,
        learned,
        s=180,
        marker="D",
        edgecolor="black",
        linewidth=1.2,
        label="Learned direction" if i == 0 else None,
        zorder=5,
    )

    ax.text(
        i,
        learned + 3,
        f"{learned:.1f}%",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )

    ax.text(
        i,
        random_flips.mean() + 3,
        f"random mean: {random_flips.mean():.1f}%",
        ha="center",
        va="bottom",
        fontsize=10,
    )


ax.set_xticks(range(len(layers)))
ax.set_xticklabels(["Layer 9", "Layer 11"])

ax.set_ylabel("Refusal → compliance (%)")
ax.set_xlabel("Ablation layer")

ax.set_title(
    "Ablating the learned direction causes behavioural changes\n"
    "far beyond random-direction controls"
)

ax.set_ylim(0, 50)
ax.grid(axis="y", alpha=0.2)

ax.legend(frameon=False, loc="upper right")

plt.tight_layout()

outfile = OUT / "figure_ablation_learned_vs_random.png"
plt.savefig(outfile, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved: {outfile}")

# Print exact numbers for checking
print("\nRESULTS")
print("=" * 60)

for layer in layers:
    result = data[str(layer)]["ablation"]

    learned = result["learned_flips"]
    random_flips = np.array(result["random_flips"])

    print(f"\nLayer {layer}")
    print(f"  Learned: {learned}/153 = {learned/N*100:.2f}%")
    print(
        f"  Random:  mean={random_flips.mean():.2f}, "
        f"SD={random_flips.std(ddof=1):.2f}, "
        f"max={random_flips.max()}"
    )
    print(
        f"  Random %: mean={random_flips.mean()/N*100:.2f}%, "
        f"max={random_flips.max()/N*100:.2f}%"
    )
