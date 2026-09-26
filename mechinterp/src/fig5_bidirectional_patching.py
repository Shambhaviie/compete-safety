from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt


A2C_DIR = Path(
    "mechinterp/results/activation_patch_A_to_C_full_163"
)

BIDIR_DIR = Path(
    "mechinterp/results/activation_patch_bidirectional_analysis"
)

OUT = A2C_DIR / "figure_erik_05_bidirectional_causal_patching.png"


# ============================================================
# A -> C
# ============================================================

summary_path = (
    A2C_DIR /
    "full_A_to_C_20260907_summary.json"
)

with open(summary_path) as f:
    summary = json.load(f)


a2c = summary["results"]["A_to_C"]["by_layer"]

a2c_layers = sorted(
    int(layer)
    for layer in a2c.keys()
)

a2c_refusal = [
    a2c[str(layer)]["patched_refusal_rate"]
    for layer in a2c_layers
]


# ============================================================
# C -> A
# ============================================================

csv_path = (
    BIDIR_DIR /
    "prompt_level_bidirectional.csv"
)

df = pd.read_csv(csv_path)

print("Bidirectional CSV columns:")
print(list(df.columns))


c2a_layers = list(range(32))

c2a_refusal = []

for layer in c2a_layers:

    flip_col = f"C2A_L{layer}_flip"

    if flip_col not in df.columns:
        raise ValueError(
            f"Missing column: {flip_col}"
        )

    # A naturally refuses.
    #
    # C->A flip means:
    # A refusal -> compliance after inserting C activation.
    #
    # Therefore:
    #
    # refusal after patch = 1 - flip rate
    #
    flip_rate = df[flip_col].astype(bool).mean()

    refusal_rate = 1.0 - flip_rate

    c2a_refusal.append(refusal_rate)


# ============================================================
# Plot
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 4.8),
)


# ------------------------------------------------------------
# Panel A: A -> C
# ------------------------------------------------------------

ax = axes[0]

ax.plot(
    a2c_layers,
    a2c_refusal,
    marker="o",
    markersize=6,
    linewidth=2.7,
)

ax.set_xlabel(
    "Transformer layer",
    fontsize=12,
)

ax.set_ylabel(
    "Refusal rate after patch",
    fontsize=12,
)

ax.set_title(
    "A activation → C prompt",
    fontsize=13,
)

ax.set_ylim(
    0,
    1.0,
)

ax.set_xticks(
    a2c_layers
)

ax.grid(
    axis="y",
    alpha=0.2,
)


for x, y in zip(
    a2c_layers,
    a2c_refusal,
):
    ax.text(
        x,
        min(y + 0.045, 0.96),
        f"{100*y:.0f}%",
        ha="center",
        fontsize=9,
    )


# Highlight L16.
best_idx = a2c_refusal.index(
    max(a2c_refusal)
)

best_layer = a2c_layers[best_idx]
best_rate = a2c_refusal[best_idx]

ax.annotate(
    f"L{best_layer}: {100*best_rate:.1f}%",
    xy=(best_layer, best_rate),
    xytext=(8, -35),
    textcoords="offset points",
    arrowprops=dict(
        arrowstyle="->",
        linewidth=1,
    ),
    fontsize=10,
)


# ------------------------------------------------------------
# Panel B: C -> A
# ------------------------------------------------------------

ax = axes[1]

ax.plot(
    c2a_layers,
    c2a_refusal,
    marker="o",
    markersize=3,
    linewidth=2.5,
)

ax.set_xlabel(
    "Transformer layer",
    fontsize=12,
)

ax.set_ylabel(
    "Refusal rate after patch",
    fontsize=12,
)

ax.set_title(
    "C activation → A prompt",
    fontsize=13,
)

ax.set_ylim(
    0.65,
    1.02,
)

ax.set_xticks(
    range(0, 32, 4)
)

ax.grid(
    axis="y",
    alpha=0.2,
)


fig.suptitle(
    "Condition-specific activations causally transfer safety behaviour",
    fontsize=14,
)

fig.tight_layout()

OUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

fig.savefig(
    OUT,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print(f"\nSaved figure to:\n{OUT}")
