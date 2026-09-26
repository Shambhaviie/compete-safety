import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

RESULTS = Path("mechinterp/results/multi_random_control_results.json")
OUT = Path("figures")
OUT.mkdir(exist_ok=True)

# Same evaluation population used for learned and random ablations
N_PROMPTS = 153


# ------------------------------------------------------------
# Load results
# ------------------------------------------------------------

with open(RESULTS) as f:
    data = json.load(f)


layers = [9, 11]

learned = []
random_values = []

for layer in layers:
    result = data[str(layer)]["ablation"]

    learned_flips = result["learned_flips"]
    random_flips = np.array(result["random_flips"], dtype=float)

    learned.append(100 * learned_flips / N_PROMPTS)
    random_values.append(100 * random_flips / N_PROMPTS)


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8.5, 5.8))

x = np.arange(len(layers))

rng = np.random.default_rng(42)

# Random-direction distributions
for i, values in enumerate(random_values):

    # Small horizontal jitter so individual random directions
    # can be seen without overlapping completely.
    jitter = rng.uniform(-0.075, 0.075, size=len(values))

    ax.scatter(
        np.full(len(values), x[i]) + jitter,
        values,
        s=45,
        alpha=0.65,
        linewidths=0,
        zorder=2,
    )

    # Mean of random directions
    mean_random = values.mean()

    ax.plot(
        [x[i] - 0.12, x[i] + 0.12],
        [mean_random, mean_random],
        linewidth=3,
        zorder=3,
    )

    # Label random mean
    ax.text(
        x[i] + 0.15,
        mean_random + 0.7,
        f"{mean_random:.2f}% mean",
        fontsize=9,
        va="bottom",
    )


# Learned-direction results
for i, value in enumerate(learned):

    ax.scatter(
        x[i],
        value,
        s=260,
        marker="D",
        edgecolor="black",
        linewidth=1.3,
        zorder=5,
    )

    ax.text(
        x[i],
        value + 2.0,
        f"{value:.1f}%",
        ha="center",
        va="bottom",
        fontsize=14,
        fontweight="bold",
    )


# ------------------------------------------------------------
# Labels
# ------------------------------------------------------------

ax.set_xticks(x)
ax.set_xticklabels(["Layer 9", "Layer 11"], fontsize=12)

ax.set_ylabel(
    "Prompts switching from refusal to compliance (%)",
    fontsize=12,
)

ax.set_xlabel(
    "Ablation layer",
    fontsize=12,
)

ax.set_title(
    "Ablating the learned direction selectively breaks refusal",
    fontsize=15,
    pad=14,
)

# Subtitle-like annotation
ax.text(
    0.5,
    1.01,
    "153 prompts · 20 independent random directions per layer",
    transform=ax.transAxes,
    ha="center",
    va="bottom",
    fontsize=10,
)


# ------------------------------------------------------------
# Visual cleanup
# ------------------------------------------------------------

ax.set_ylim(0, 48)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.grid(
    axis="y",
    alpha=0.18,
    linewidth=0.8,
)

ax.tick_params(axis="both", labelsize=11)

plt.tight_layout()


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

outfile = OUT / "figure_ablation_learned_vs_random_clean.png"

plt.savefig(
    outfile,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(f"Saved: {outfile}")


# ------------------------------------------------------------
# Print exact values
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("ABLATION RESULTS")
print("=" * 65)

for i, layer in enumerate(layers):

    random_pct = random_values[i]
    learned_pct = learned[i]

    print(f"\nLayer {layer}")
    print(f"  Learned direction: {learned_pct:.2f}%")
    print(f"  Random mean:       {random_pct.mean():.2f}%")
    print(f"  Random SD:         {random_pct.std(ddof=1):.2f}%")
    print(f"  Random minimum:    {random_pct.min():.2f}%")
    print(f"  Random maximum:    {random_pct.max():.2f}%")
