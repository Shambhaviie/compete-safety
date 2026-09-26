import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
INPUT = Path("mechinterp/results/A_vs_C_same_prompt/layer_summary.csv")
OUTPUT = Path("figures/figure_erik_01_same_prompt_cosine.png")

# ------------------------------------------------------------
# Load actual analysis results
# ------------------------------------------------------------
df = pd.read_csv(INPUT)

print("Columns found:")
for c in df.columns:
    print("  ", c)

# ------------------------------------------------------------
# Find columns robustly
# ------------------------------------------------------------
def find_col(candidates, required_words=None):
    candidates_lower = {c.lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in candidates_lower:
            return candidates_lower[candidate.lower()]

    if required_words:
        for c in df.columns:
            cl = c.lower()
            if all(word.lower() in cl for word in required_words):
                return c

    return None


layer_col = find_col(
    ["layer", "Layer", "layer_idx", "layer_index"],
    ["layer"]
)

flip_col = find_col(
    [
        "flip_cosine_mean",
        "flips_cosine_mean",
        "flip_mean_cosine",
        "mean_cosine_flip",
        "flip_cosine",
    ],
    ["flip", "cosine"]
)

stable_col = find_col(
    [
        "stable_cosine_mean",
        "stable_refusal_cosine_mean",
        "stable_mean_cosine",
        "mean_cosine_stable",
        "stable_cosine",
    ],
    ["stable", "cosine"]
)

# ------------------------------------------------------------
# Print what was selected
# ------------------------------------------------------------
print("\nSelected columns:")
print("  Layer :", layer_col)
print("  Flips :", flip_col)
print("  Stable:", stable_col)

if layer_col is None or flip_col is None or stable_col is None:
    raise RuntimeError(
        "\nCould not identify the required columns automatically.\n"
        "The actual column names are printed above. "
        "Please send that output before proceeding."
    )

# ------------------------------------------------------------
# Prepare data
# ------------------------------------------------------------
plot_df = df[[layer_col, flip_col, stable_col]].copy()
plot_df.columns = ["layer", "flips", "stable"]

plot_df = plot_df.sort_values("layer")

print("\nData to plot:")
print(plot_df.to_string(index=False))

if plot_df["flips"].isna().all():
    raise RuntimeError("Flip cosine column contains no usable values.")

if plot_df["stable"].isna().all():
    raise RuntimeError("Stable cosine column contains no usable values.")

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 6.5))

ax.plot(
    plot_df["layer"],
    plot_df["flips"],
    marker="o",
    linewidth=2.5,
    markersize=5,
    label="A→C flips (n=163)",
)

ax.plot(
    plot_df["layer"],
    plot_df["stable"],
    marker="o",
    linewidth=2.5,
    markersize=5,
    label="Stable refusals (n=512)",
)

# Middle-layer region
ax.axvspan(
    13,
    24,
    alpha=0.08,
    label="Middle layers"
)

ax.set_title(
    "Competitive framing increasingly changes the internal state of flipping prompts",
    fontsize=20,
    pad=14,
)

ax.set_xlabel(
    "Transformer layer",
    fontsize=16
)

ax.set_ylabel(
    "Cosine similarity between A and C",
    fontsize=16
)

ax.set_xlim(0, 31)
ax.set_ylim(
    min(plot_df[["flips", "stable"]].min()) - 0.03,
    1.01
)

ax.grid(
    axis="y",
    alpha=0.25
)

ax.legend(
    fontsize=13,
    loc="lower left",
    frameon=False
)

ax.tick_params(
    labelsize=12
)

# Small explanatory annotation
ax.text(
    18.5,
    ax.get_ylim()[0] + 0.04,
    "Greater A–C divergence\nfor prompts that flip",
    ha="center",
    va="bottom",
    fontsize=13,
)

plt.tight_layout()

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(
    OUTPUT,
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("\nSaved:")
print(OUTPUT)
