from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


RESULTS = Path("mechinterp/results/A_vs_C_same_prompt")
CSV = RESULTS / "layer_summary.csv"
OUT = RESULTS / "figure_erik_01_same_prompt_cosine.png"


def find_column(df, candidates):
    lower = {c.lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]

    for c in df.columns:
        cl = c.lower()
        if all(term.lower() in cl for term in candidates):
            return c

    raise ValueError(
        f"Could not find column matching {candidates}. "
        f"Available columns:\n{list(df.columns)}"
    )


df = pd.read_csv(CSV)

print("Columns:")
print(list(df.columns))

layer_col = find_column(df, ["layer"])

# Expected names are usually something like:
# flip_cosine / stable_cosine
flip_col = None
stable_col = None

for c in df.columns:
    cl = c.lower()

    if "flip" in cl and "cos" in cl:
        flip_col = c

    if "stable" in cl and "cos" in cl:
        stable_col = c

if flip_col is None or stable_col is None:
    raise ValueError(
        "Could not identify flip/stable cosine columns.\n"
        f"Columns: {list(df.columns)}"
    )

df = df.sort_values(layer_col)

print(f"Layer column: {layer_col}")
print(f"Flip cosine column: {flip_col}")
print(f"Stable cosine column: {stable_col}")


fig, ax = plt.subplots(figsize=(9, 5.3))

ax.plot(
    df[layer_col],
    df[flip_col],
    marker="o",
    markersize=3.5,
    linewidth=2.5,
    label="A→C flips (n=163)",
)

ax.plot(
    df[layer_col],
    df[stable_col],
    marker="o",
    markersize=3.5,
    linewidth=2.5,
    label="Stable refusals (n=512)",
)

# Highlight the broad middle/late-layer region.
ax.axvspan(
    13,
    24,
    alpha=0.08,
)

ax.set_xlabel("Transformer layer", fontsize=12)
ax.set_ylabel("Cosine similarity between A and C", fontsize=12)

ax.set_title(
    "Competitive framing increasingly changes the internal state of flipping prompts",
    fontsize=13,
)

ax.set_xlim(
    df[layer_col].min(),
    df[layer_col].max(),
)

ax.set_ylim(0.35, 1.02)

ax.grid(
    axis="y",
    alpha=0.2,
)

ax.legend(
    frameon=False,
    fontsize=10,
)

ax.text(
    18.5,
    0.40,
    "Increasing A–C divergence\nfor prompts that flip",
    ha="center",
    va="bottom",
    fontsize=10,
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
