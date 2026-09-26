import json
import numpy as np
from pathlib import Path

ROOT = Path("/home/Shambhavi/compete_safety_v2")
RESULT_DIR = ROOT / "mechinterp/results/experiment13_full"

with open(RESULT_DIR / "selected_prompts.json") as f:
    selection = json.load(f)

if isinstance(selection, dict):
    selected = selection.get(
        "selected_prompts",
        selection.get("prompts")
    )
else:
    selected = selection

LAYERS = [7, 9, 11, 14, 18, 24]

rows = []

for layer in LAYERS:

    data = np.load(
        RESULT_DIR / f"trajectory_L{layer}_ablation.npz",
        allow_pickle=True
    )

    safety = np.asarray(
        data["safety_labels"]
    ).astype(bool)

    groups = np.asarray(
        data["groups"]
    ).astype(str)

    flip = groups == "flip"
    stable = groups == "stable"

    # ------------------------------------------------------------
    # FLIP GROUP
    # Natural C = compliance
    # ------------------------------------------------------------

    flip_n = int(flip.sum())

    flip_refusal = int(
        safety[flip].sum()
    )

    flip_compliance = (
        flip_n - flip_refusal
    )

    # ------------------------------------------------------------
    # STABLE GROUP
    # Natural C = refusal
    # ------------------------------------------------------------

    stable_n = int(stable.sum())

    stable_refusal = int(
        safety[stable].sum()
    )

    stable_compliance = (
        stable_n - stable_refusal
    )

    rows.append([
        layer,
        flip_n,
        flip_compliance,
        flip_refusal,
        100 * flip_compliance / flip_n,
        100 * flip_refusal / flip_n,
        stable_n,
        stable_refusal,
        stable_compliance,
        100 * stable_refusal / stable_n,
        100 * stable_compliance / stable_n,
    ])


# ================================================================
# PRINT SINGLE TABLE
# ================================================================

headers = [
    "Layer",

    "Flip N",
    "Flip→Compliance",
    "Flip→Refusal",
    "Flip Compliance %",
    "Flip Refusal %",

    "Stable N",
    "Stable→Refusal",
    "Stable→Compliance",
    "Stable Refusal %",
    "Stable Compliance %",
]

print("\n" + "=" * 125)
print("EXPERIMENT 13 — BEHAVIOURAL OUTCOMES AFTER REFUSAL-DIRECTION ABLATION")
print("=" * 125)

print(
    f"{headers[0]:>5} | "
    f"{headers[1]:>7} | "
    f"{headers[2]:>15} | "
    f"{headers[3]:>12} | "
    f"{headers[4]:>15} | "
    f"{headers[5]:>13} | "
    f"{headers[6]:>8} | "
    f"{headers[7]:>15} | "
    f"{headers[8]:>17} | "
    f"{headers[9]:>15} | "
    f"{headers[10]:>18}"
)

print("-" * 125)

for r in rows:
    print(
        f"{r[0]:>5} | "
        f"{r[1]:>7} | "
        f"{r[2]:>15} | "
        f"{r[3]:>12} | "
        f"{r[4]:>14.1f}% | "
        f"{r[5]:>12.1f}% | "
        f"{r[6]:>8} | "
        f"{r[7]:>15} | "
        f"{r[8]:>17} | "
        f"{r[9]:>14.1f}% | "
        f"{r[10]:>17.1f}%"
    )

print("=" * 125)

print("""
Interpretation of columns:

Flip prompts:
    Natural C = compliance.
    Flip→Compliance = remained compliant after ablation.
    Flip→Refusal    = changed from compliance to refusal.

Stable prompts:
    Natural C = refusal.
    Stable→Refusal   = remained refused after ablation.
    Stable→Compliance = changed from refusal to compliance.

safe=True in the saved files means REFUSAL.
safe=False means COMPLIANCE.
""")
