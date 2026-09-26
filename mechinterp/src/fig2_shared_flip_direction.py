from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


RESULTS = Path("mechinterp/results/A_vs_C_flips")
CSV = RESULTS / "layer_displacement_direction_consistency.csv"
OUT = RESULTS / "figure_erik_02_shared_direction.png"


df = pd.read_csv(CSV)

print("Columns:")
print(list(df.columns))


def find_layer(df):
    for c in df.columns:
        if c.lower() in {"layer", "layer_idx", "layer_index"}:
            return c

    for c in df.columns:
        if "layer" in c.lower():
            return c

    raise ValueError("Could not find layer column.")


def find_alignment(df):
    # Prefer a mean alignment column.
    for c in df.columns:
        cl = c.lower()
        if "alignment" in cl and "mean" in cl:
            return c

    for c in df.columns:
        cl = c.lower()
        if "alignment" in cl:
            return c

    raise ValueError(
        "Could not find alignment column.\n"
        f"Columns: {list(df.columns)}"
    )


layer_col = find_layer(df)
alignment_col = find_alignment(df)

df = df.sort_values(layer_col)

print(f"Layer column: {layer_col}")
print(f"Alignment column: {alignment_col}")


fig, ax = plt.subplots(figsize=(9, 5.3))

ax.plot(
    df[layer_col],
    df[alignment_col],
    marker="o",
    markersize=3.5,
    linewidth=2.7,
    label="163 A→C flipping prompts",
)

ax.axhline(
    0.8,
    linestyle="--",
    linewidth=1.1,
    alpha=0.55,
)

ax.axhline(
    0.5,
    linestyle="--",
    linewidth=1.0,
    alpha=0.4,
)

ax.text(
    df[layer_col].max() + 0.35,
    0.8,
    "0.8",
    va="center",
    fontsize=9,
)

ax.text(
    df[layer_col].max() + 0.35,
    0.5,
    "0.5",
    va="center",
    fontsize=9,
)

ax.set_xlabel(
    "Transformer layer",
    fontsize=12,
)

ax.set_ylabel(
    "Alignment with shared A→C direction",
    fontsize=12,
)

ax.set_title(
    "Flipping prompts share a strong directional change",
    fontsize=13,
)

ax.set_xlim(
    df[layer_col].min(),
    df[layer_col].max(),
)

ax.set_ylim(
    max(0.4, df[alignment_col].min() - 0.03),
    1.02,
)

ax.grid(
    axis="y",
    alpha=0.2,
)

ax.legend(
    frameon=False,
)

fig.tight_layout()

OUT.parent.mkdir(parents=True, exist_ok=True)

fig.savefig(
    OUT,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print(f"\nSaved figure to:\n{OUT}")
