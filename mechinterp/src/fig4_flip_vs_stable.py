from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


RESULTS = Path("mechinterp/results/A_vs_C_flip_vs_stable")
OUT = RESULTS / "figure_erik_04_flip_vs_stable.png"


csv_files = sorted(RESULTS.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(
        f"No CSV files found in {RESULTS}"
    )


print("CSV files:")
for f in csv_files:
    print(" ", f.name)


# Find the displacement comparison CSV.
chosen = None

for f in csv_files:
    try:
        temp = pd.read_csv(f)
    except Exception:
        continue

    cols = [c.lower() for c in temp.columns]

    has_flip = any("flip" in c for c in cols)
    has_stable = any("stable" in c for c in cols)
    has_auc = any("auc" in c for c in cols)

    if has_flip and has_stable and has_auc:
        chosen = f
        break


if chosen is None:
    raise ValueError(
        "Could not identify flip-vs-stable CSV.\n"
        f"CSV files: {csv_files}"
    )


df = pd.read_csv(chosen)

print(f"\nUsing: {chosen}")
print("Columns:")
print(list(df.columns))


# Find layer
layer_col = None

for c in df.columns:
    if c.lower() in {"layer", "layer_idx", "layer_index"}:
        layer_col = c
        break

if layer_col is None:
    for c in df.columns:
        if "layer" in c.lower():
            layer_col = c
            break


if layer_col is None:
    raise ValueError("Could not find layer column.")


# Find mean flip/stable cosine alignment columns.
flip_col = None
stable_col = None

for c in df.columns:
    cl = c.lower()

    if flip_col is None:
        if "flip" in cl and "mean" in cl and (
            "cos" in cl or "align" in cl
        ):
            flip_col = c

    if stable_col is None:
        if "stable" in cl and "mean" in cl and (
            "cos" in cl or "align" in cl
        ):
            stable_col = c


# If the file has no "mean" in the column name, fall back.
if flip_col is None:
    for c in df.columns:
        cl = c.lower()
        if "flip" in cl and ("cos" in cl or "align" in cl):
            flip_col = c
            break

if stable_col is None:
    for c in df.columns:
        cl = c.lower()
        if "stable" in cl and ("cos" in cl or "align" in cl):
            stable_col = c
            break


auc_col = None

for c in df.columns:
    if "auc" in c.lower():
        auc_col = c
        break


if flip_col is None or stable_col is None or auc_col is None:
    raise ValueError(
        "Could not identify required columns.\n"
        f"Columns: {list(df.columns)}"
    )


df = df.sort_values(layer_col)

print(f"Layer: {layer_col}")
print(f"Flip alignment: {flip_col}")
print(f"Stable alignment: {stable_col}")
print(f"AUC: {auc_col}")


fig, axes = plt.subplots(
    2,
    1,
    figsize=(9, 8),
    sharex=True,
)


# ------------------------------------------------------------
# Top: alignment
# ------------------------------------------------------------

ax = axes[0]

ax.plot(
    df[layer_col],
    df[flip_col],
    marker="o",
    markersize=3,
    linewidth=2.5,
    label="A→C flips (n=163)",
)

ax.plot(
    df[layer_col],
    df[stable_col],
    marker="o",
    markersize=3,
    linewidth=2.5,
    label="Stable refusals (n=512)",
)

ax.axvspan(
    13,
    24,
    alpha=0.08,
)

ax.set_ylabel(
    "Alignment with flip direction",
    fontsize=11,
)

ax.set_title(
    "The shared direction becomes specific to prompts that flip",
    fontsize=13,
)

ax.set_ylim(
    0.45,
    1.0,
)

ax.grid(
    axis="y",
    alpha=0.2,
)

ax.legend(
    frameon=False,
)


# ------------------------------------------------------------
# Bottom: AUC
# ------------------------------------------------------------

ax = axes[1]

ax.plot(
    df[layer_col],
    df[auc_col],
    marker="o",
    markersize=3,
    linewidth=2.5,
)

ax.axhline(
    0.5,
    linestyle="--",
    linewidth=1.0,
    alpha=0.5,
)

ax.set_xlabel(
    "Transformer layer",
    fontsize=12,
)

ax.set_ylabel(
    "AUC",
    fontsize=11,
)

ax.set_ylim(
    0.45,
    1.0,
)

ax.grid(
    axis="y",
    alpha=0.2,
)


# Annotate final AUC.
last = df.iloc[-1]

ax.annotate(
    f"AUC = {last[auc_col]:.3f}",
    xy=(last[layer_col], last[auc_col]),
    xytext=(-70, -28),
    textcoords="offset points",
    arrowprops=dict(
        arrowstyle="->",
        linewidth=1,
    ),
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
