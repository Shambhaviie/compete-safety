import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

LAYER_RESULTS = Path(
    "mechinterp/results/layer_specific_ablation_results.json"
)

RANDOM_RESULTS = Path(
    "mechinterp/results/multi_random_control_results.json"
)

OUT = Path("figures")
OUT.mkdir(exist_ok=True)

N_PROMPTS = 153


# ============================================================
# Load learned-direction results
# ============================================================

with open(LAYER_RESULTS) as f:
    data = json.load(f)


# The layer-specific file stores results as a LIST.
# Each entry corresponds to a layer.
results = data["results"] if "results" in data else data


learned_layers = []
learned_flips = []
learned_rates = []


for layer_result in results:

    # --------------------------------------------------------
    # Find layer number
    # --------------------------------------------------------

    layer = (
        layer_result.get("layer")
        if isinstance(layer_result, dict)
        else None
    )

    if layer is None:
        continue

    layer = int(layer)


    # --------------------------------------------------------
    # Find prompt-level results
    # --------------------------------------------------------

    if "prompt_results" in layer_result:
        prompt_results = layer_result["prompt_results"]

    elif "results" in layer_result:
        prompt_results = layer_result["results"]

    elif "prompts" in layer_result:
        prompt_results = layer_result["prompts"]

    else:
        continue


    # --------------------------------------------------------
    # Count refusal -> compliance flips
    # --------------------------------------------------------

    flips = 0

    for r in prompt_results:

        if not isinstance(r, dict):
            continue

        # Explicit flip field
        if r.get("flip") is True:
            flips += 1
            continue

        # Try refusal labels
        baseline = r.get(
            "baseline_refusal",
            r.get(
                "baseline_label",
                r.get("natural_refusal")
            )
        )

        ablated = r.get(
            "ablated_refusal",
            r.get(
                "ablated_label",
                r.get("ablation_refusal")
            )
        )

        if isinstance(baseline, bool) and isinstance(ablated, bool):
            if baseline is True and ablated is False:
                flips += 1


    learned_layers.append(layer)
    learned_flips.append(flips)
    learned_rates.append(
        100 * flips / N_PROMPTS
    )


# ============================================================
# Sort by layer
# ============================================================

order = np.argsort(learned_layers)

learned_layers = np.array(learned_layers)[order]
learned_flips = np.array(learned_flips)[order]
learned_rates = np.array(learned_rates)[order]


# ============================================================
# Load random controls
# ============================================================

with open(RANDOM_RESULTS) as f:
    random_data = json.load(f)


random_values = {}

for layer in [9, 11]:

    result = random_data[str(layer)]["ablation"]

    flips = np.array(
        result["random_flips"],
        dtype=float
    )

    random_values[layer] = (
        100 * flips / N_PROMPTS
    )


# ============================================================
# Print extracted values
# ============================================================

print("\n" + "=" * 70)
print("LEARNED-DIRECTION ABLATION")
print("=" * 70)

for layer, flips, rate in zip(
    learned_layers,
    learned_flips,
    learned_rates
):
    print(
        f"L{layer:02d}: "
        f"{flips:3d}/{N_PROMPTS} "
        f"({rate:.1f}%)"
    )


print("\n" + "=" * 70)
print("RANDOM CONTROLS")
print("=" * 70)

for layer in [9, 11]:

    values = random_values[layer]

    print(
        f"L{layer}: "
        f"mean={values.mean():.2f}%, "
        f"max={values.max():.2f}%"
    )


# ============================================================
# Plot
# ============================================================

fig, ax = plt.subplots(
    figsize=(11, 5.8)
)

x = np.arange(len(learned_layers))


# ------------------------------------------------------------
# Learned-direction trajectory
# ------------------------------------------------------------

ax.plot(
    x,
    learned_rates,
    marker="o",
    markersize=7,
    linewidth=2.5,
    zorder=4,
    label="Learned direction",
)


# ------------------------------------------------------------
# Random controls at L9 and L11
# ------------------------------------------------------------

rng = np.random.default_rng(42)

for layer in [9, 11]:

    if layer not in learned_layers:
        continue

    idx = np.where(
        learned_layers == layer
    )[0][0]

    values = random_values[layer]

    jitter = rng.uniform(
        -0.13,
        0.13,
        size=len(values)
    )

    ax.scatter(
        np.full(len(values), idx) + jitter,
        values,
        s=38,
        alpha=0.55,
        zorder=2,
        label=(
            "Random directions"
            if layer == 9
            else None
        ),
    )

    # Random mean
    ax.plot(
        [
            idx - 0.14,
            idx + 0.14
        ],
        [
            values.mean(),
            values.mean()
        ],
        linewidth=3,
        zorder=3,
    )


# ------------------------------------------------------------
# Annotate larger effects
# ------------------------------------------------------------

for layer, rate in zip(
    learned_layers,
    learned_rates
):

    if rate >= 10:

        idx = np.where(
            learned_layers == layer
        )[0][0]

        ax.text(
            idx,
            rate + 1.5,
            f"{rate:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )


# ------------------------------------------------------------
# Axes
# ------------------------------------------------------------

ax.set_xticks(x)

ax.set_xticklabels(
    [f"L{layer}" for layer in learned_layers],
    fontsize=10,
)

ax.set_xlabel(
    "Transformer layer",
    fontsize=12,
)

ax.set_ylabel(
    "Refusal → compliance (%)",
    fontsize=12,
)

ax.set_title(
    "Ablating the learned direction breaks refusal at specific layers",
    fontsize=15,
    pad=15,
)

ax.text(
    0.5,
    1.01,
    "153 naturally refusing prompts · 20 random directions at L9 and L11",
    transform=ax.transAxes,
    ha="center",
    va="bottom",
    fontsize=10,
)


# ------------------------------------------------------------
# Style
# ------------------------------------------------------------

ax.set_ylim(
    0,
    max(
        45,
        learned_rates.max() + 7
    )
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.grid(
    axis="y",
    alpha=0.18,
    linewidth=0.8,
)

ax.tick_params(
    axis="both",
    labelsize=10,
)

ax.legend(
    frameon=False,
    loc="upper left",
)

plt.tight_layout()


# ============================================================
# Save
# ============================================================

outfile = (
    OUT /
    "figure_ablation_all_layers_clean.png"
)

plt.savefig(
    outfile,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print("\nSaved:")
print(outfile)
