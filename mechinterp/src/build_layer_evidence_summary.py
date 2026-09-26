from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("mechinterp/results")

RESULT_DIRS = [
    ROOT / "A_vs_C_flips",
    ROOT / "A_vs_C_flips_cv",
    ROOT / "A_vs_C_flip_vs_stable",
    ROOT / "A_condition_harmful_harmless",
]

print("=" * 120)
print("LAYER-LEVEL EVIDENCE FOR PATCHING LAYER SELECTION")
print("=" * 120)

# ---------------------------------------------------------------------
# Find CSVs
# ---------------------------------------------------------------------

for directory in RESULT_DIRS:
    if not directory.exists():
        print(f"\n[MISSING] {directory}")
        continue

    print("\n" + "=" * 120)
    print(f"DIRECTORY: {directory}")
    print("=" * 120)

    for csv_path in sorted(directory.glob("*.csv")):

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"\nCould not read {csv_path.name}: {e}")
            continue

        # Normalize column names for inspection
        cols = {str(c).lower(): c for c in df.columns}

        # Find possible layer column
        layer_col = None
        for candidate in ["layer", "layers"]:
            if candidate in cols:
                layer_col = cols[candidate]
                break

        if layer_col is None:
            continue

        # Convert layer to numeric where possible
        tmp = df.copy()
        tmp["_layer_numeric"] = pd.to_numeric(
            tmp[layer_col], errors="coerce"
        )

        tmp = tmp[tmp["_layer_numeric"].notna()].copy()

        # Only retain actual transformer layers 0-31
        tmp = tmp[
            tmp["_layer_numeric"].between(0, 31)
        ].copy()

        if len(tmp) == 0:
            continue

        # -------------------------------------------------------------
        # Skip obvious prompt-level files
        # -------------------------------------------------------------

        prompt_cols = [
            c for c in tmp.columns
            if any(
                key in str(c).lower()
                for key in [
                    "prompt_id",
                    "prompt",
                    "round_id",
                    "example_id",
                ]
            )
        ]

        # If every layer appears many times, this is probably
        # prompt-level rather than layer-level.
        counts = tmp.groupby("_layer_numeric").size()

        if counts.median() > 10:
            continue

        print("\n" + "-" * 120)
        print(f"FILE: {csv_path.relative_to(ROOT)}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("-" * 120)

        # -------------------------------------------------------------
        # Print all rows if genuinely layer-level
        # -------------------------------------------------------------

        display = tmp.drop(columns=["_layer_numeric"])

        print(display.to_string(index=False))


# ---------------------------------------------------------------------
# Special aggregation: prompt-level tables
#
# If important analyses were saved only at prompt level, aggregate
# numeric variables by layer and group where possible.
# ---------------------------------------------------------------------

print("\n\n" + "=" * 120)
print("AGGREGATED PROMPT-LEVEL EVIDENCE")
print("=" * 120)

for directory in RESULT_DIRS:

    if not directory.exists():
        continue

    for csv_path in sorted(directory.glob("*.csv")):

        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        cols_lower = {
            str(c).lower(): c for c in df.columns
        }

        # Need layer
        layer_col = None
        for candidate in ["layer", "layers"]:
            if candidate in cols_lower:
                layer_col = cols_lower[candidate]
                break

        if layer_col is None:
            continue

        # Detect prompt-level structure
        prompt_col = None
        for candidate in [
            "prompt_id",
            "prompt",
            "round_id",
            "example_id",
        ]:
            if candidate in cols_lower:
                prompt_col = cols_lower[candidate]
                break

        if prompt_col is None:
            continue

        tmp = df.copy()
        tmp["_layer"] = pd.to_numeric(
            tmp[layer_col], errors="coerce"
        )

        tmp = tmp[
            tmp["_layer"].between(0, 31)
        ].copy()

        if len(tmp) == 0:
            continue

        # -------------------------------------------------------------
        # Identify numeric analysis columns
        # -------------------------------------------------------------

        numeric_cols = []

        for c in tmp.columns:
            if c in [layer_col, "_layer"]:
                continue

            if pd.api.types.is_numeric_dtype(tmp[c]):
                numeric_cols.append(c)

        # Don't aggregate obvious IDs
        numeric_cols = [
            c for c in numeric_cols
            if not any(
                x in str(c).lower()
                for x in [
                    "id",
                    "seed",
                    "fold",
                ]
            )
        ]

        if not numeric_cols:
            continue

        # -------------------------------------------------------------
        # Identify useful grouping columns
        # -------------------------------------------------------------

        group_candidates = []

        for candidate in [
            "group",
            "condition",
            "label",
            "status",
            "type",
            "category",
        ]:
            if candidate in cols_lower:
                c = cols_lower[candidate]

                # Keep only manageable categorical variables
                if tmp[c].nunique(dropna=True) <= 10:
                    group_candidates.append(c)

        print("\n" + "-" * 120)
        print(f"FILE: {csv_path.relative_to(ROOT)}")
        print(f"Prompt-level shape: {df.shape}")
        print(f"Numeric variables: {numeric_cols}")
        print(f"Grouping variables: {group_candidates}")
        print("-" * 120)

        # -------------------------------------------------------------
        # Aggregate
        # -------------------------------------------------------------

        if group_candidates:

            group_col = group_candidates[0]

            agg = (
                tmp.groupby(["_layer", group_col])[numeric_cols]
                .mean()
                .reset_index()
            )

        else:

            agg = (
                tmp.groupby("_layer")[numeric_cols]
                .mean()
                .reset_index()
            )

        # Round for readability
        numeric_to_round = [
            c for c in agg.columns
            if pd.api.types.is_numeric_dtype(agg[c])
        ]

        agg[numeric_to_round] = agg[numeric_to_round].round(4)

        print(agg.to_string(index=False))


# ---------------------------------------------------------------------
# Explicit search for files associated with 14A
# ---------------------------------------------------------------------

print("\n\n" + "=" * 120)
print("14A / TOKEN-DIVERGENCE FILE SEARCH")
print("=" * 120)

for p in ROOT.rglob("*"):
    if p.is_file():
        name = p.name.lower()

        if any(
            key in name
            for key in [
                "14a",
                "token",
                "diverg",
                "trajectory",
            ]
        ):
            print(p.relative_to(ROOT))

print("\n" + "=" * 120)
print("DONE")
print("=" * 120)
