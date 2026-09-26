from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


RESULTS = Path("mechinterp/results/A_vs_C_flips_cv")
OUT = RESULTS / "figure_erik_03_crossvalidated_direction.png"


csv_files = sorted(RESULTS.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(
        f"No CSV files found in {RESULTS}"
    )


print("CSV files found:")
for f in csv_files:
    print(" ", f.name)


# Find the CSV containing training/test alignment information.
chosen = None

for f in csv_files:
    try:
        temp = pd.read_csv(f)
    except Exception:
        continue

    cols = [c.lower() for c in temp.columns]

    has_train = any("train" in c for c in cols)
    has_test = any(
        ("test" in c or "held" in c)
        for c in cols
    )

    has_alignment = any(
        "cos" in c or "alignment" in c
        for c in cols
    )

    if has_train and has_test and has_alignment:
        chosen = f
        break


if chosen is None:
    raise ValueError(
        "Could not identify the cross-validation CSV.\n"
        "Available CSVs:\n"
        + "\n".join(str(x) for x in csv_files)
    )


df = pd.read_csv(chosen)

print(f"\nUsing: {chosen}")
print("Columns:")
print(list(df.columns))


# Layer column
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


# Find training and held-out/test mean columns.
train_col = None
test_col = None

for c in df.columns:
    cl = c.lower()

    if train_col is None:
        if "train" in cl and ("mean" in cl or "cos" in cl or "align" in cl):
            train_col = c

    if test_col is None:
        if (
            ("test" in cl or "held" in cl)
            and ("mean" in cl or "cos" in cl or "align" in cl)
        ):
            test_col = c


if train_col is None or test_col is None:
    raise ValueError(
        "Could not identify training/test alignment columns.\n"
        f"Columns: {list(df.columns)}"
    )


df = df.sort_values(layer_col)

print(f"Layer column: {layer_col}")
print(f"Training column: {train_col}")
print(f"Held-out column: {test_col}")


fig, ax = plt.subplots(figsize=(9, 5.3))

ax.plot(
    df[layer_col],
    df[train_col],
    marker="o",
    markersize=3.5,
    linewidth=2.5,
    label="Training flips",
)

ax.plot(
    df[layer_col],
    df[test_col],
    marker="o",
    markersize=3.5,
    linewidth=2.5,
    label="Held-out flips",
)

ax.set_xlabel(
    "Transformer layer",
    fontsize=12,
)

ax.set_ylabel(
    "Cosine alignment with training-derived direction",
    fontsize=12,
)

ax.set_title(
    "The shared A→C direction generalises to held-out prompts",
    fontsize=13,
)

ax.set_xlim(
    df[layer_col].min(),
    df[layer_col].max(),
)

ax.set_ylim(
    0.8,
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
