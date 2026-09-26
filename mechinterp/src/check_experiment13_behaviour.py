"""
Check the actual behavioural outcomes of Experiment 13.

IMPORTANT:
This script does NOT rerun the model.
It only reads the saved Experiment 13 outputs.

Interpretation of labels:
    safe=True  -> refusal
    safe=False -> compliance

Natural Condition C:
    flip   -> compliance
    stable -> refusal

Question:
    What happened after refusal-direction ablation?
"""

import json
import numpy as np
from pathlib import Path


ROOT = Path("/home/Shambhavi/compete_safety_v2")

RESULT_DIR = (
    ROOT / "mechinterp/results/experiment13_full"
)

SELECTION_FILE = (
    RESULT_DIR / "selected_prompts.json"
)

LAYERS = [7, 9, 11, 14, 18, 24]


# ================================================================
# LOAD SELECTION
# ================================================================

with open(SELECTION_FILE) as f:
    selection = json.load(f)

# Handle either possible JSON format.
if isinstance(selection, dict):
    if "selected_prompts" in selection:
        selected = selection["selected_prompts"]
    elif "prompts" in selection:
        selected = selection["prompts"]
    else:
        raise RuntimeError(
            "Could not find prompt list in selection file."
        )
else:
    selected = selection


print("=" * 78)
print("EXPERIMENT 13 — BEHAVIOURAL OUTCOME CHECK")
print("=" * 78)

print(f"Selected prompts: {len(selected)}")
print(
    "Flip:",
    sum(x["group"] == "flip" for x in selected)
)
print(
    "Stable:",
    sum(x["group"] == "stable" for x in selected)
)


# ================================================================
# LOAD EACH LAYER
# ================================================================

for layer in LAYERS:

    path = (
        RESULT_DIR
        / f"trajectory_L{layer}_ablation.npz"
    )

    print("\n" + "=" * 78)
    print(f"LAYER {layer}")
    print("=" * 78)

    data = np.load(
        path,
        allow_pickle=True
    )

    safety = np.asarray(
        data["safety_labels"]
    )

    prompt_ids = np.asarray(
        data["prompt_ids"]
    ).astype(str)

    groups = np.asarray(
        data["groups"]
    ).astype(str)

    # ------------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------------

    if len(safety) != len(selected):
        raise RuntimeError(
            f"L{layer}: safety count mismatch."
        )

    if not np.array_equal(
        prompt_ids,
        np.asarray(
            [x["prompt_id"] for x in selected]
        ).astype(str)
    ):
        raise RuntimeError(
            f"L{layer}: prompt ordering mismatch."
        )

    # ------------------------------------------------------------
    # Convert labels.
    #
    # safe=True  = refusal
    # safe=False = compliance
    # ------------------------------------------------------------

    refused = safety.astype(bool)

    # ============================================================
    # FLIP GROUP
    # ============================================================

    flip = groups == "flip"

    # Natural C outcome for flips:
    # compliance = False
    #
    # After ablation:
    # False -> remained compliant
    # True  -> became refusal

    flip_refused = refused[flip]

    n_flip = int(flip.sum())

    flip_still_complied = int(
        (~flip_refused).sum()
    )

    flip_became_refusal = int(
        flip_refused.sum()
    )

    print("\nFLIP PROMPTS")
    print("-" * 50)

    print(
        f"Natural C: "
        f"{n_flip}/{n_flip} complied"
    )

    print(
        f"After ablation:"
    )

    print(
        f"  Remained compliant: "
        f"{flip_still_complied}/{n_flip} "
        f"({100 * flip_still_complied / n_flip:.1f}%)"
    )

    print(
        f"  Became refusal:     "
        f"{flip_became_refusal}/{n_flip} "
        f"({100 * flip_became_refusal / n_flip:.1f}%)"
    )

    # ============================================================
    # STABLE GROUP
    # ============================================================

    stable = groups == "stable"

    # Natural C outcome for stable:
    # refusal = True
    #
    # After ablation:
    # True  -> remained refusal
    # False -> became compliance

    stable_refused = refused[stable]

    n_stable = int(stable.sum())

    stable_still_refused = int(
        stable_refused.sum()
    )

    stable_became_compliance = int(
        (~stable_refused).sum()
    )

    print("\nSTABLE PROMPTS")
    print("-" * 50)

    print(
        f"Natural C: "
        f"{n_stable}/{n_stable} refused"
    )

    print(
        f"After ablation:"
    )

    print(
        f"  Remained refusal:   "
        f"{stable_still_refused}/{n_stable} "
        f"({100 * stable_still_refused / n_stable:.1f}%)"
    )

    print(
        f"  Became compliance:  "
        f"{stable_became_compliance}/{n_stable} "
        f"({100 * stable_became_compliance / n_stable:.1f}%)"
    )

    # ============================================================
    # EXAMPLES
    # ============================================================

    print("\nEXAMPLE PROMPT IDS")
    print("-" * 50)

    ids = prompt_ids

    # Flip -> became refusal
    idx = np.where(
        flip & refused
    )[0]

    print(
        "Flip -> refusal:",
        ids[idx[:10]].tolist()
    )

    # Flip -> remained compliance
    idx = np.where(
        flip & (~refused)
    )[0]

    print(
        "Flip -> compliance:",
        ids[idx[:10]].tolist()
    )

    # Stable -> became compliance
    idx = np.where(
        stable & (~refused)
    )[0]

    print(
        "Stable -> compliance:",
        ids[idx[:10]].tolist()
    )

    # Stable -> remained refusal
    idx = np.where(
        stable & refused
    )[0]

    print(
        "Stable -> refusal:",
        ids[idx[:10]].tolist()
    )


print("\n" + "=" * 78)
print("EXPECTED LOGIC")
print("=" * 78)

print(
    """
Flip:
    Natural C = compliance
    Ablation -> compliance is the expected unchanged outcome.
    Ablation -> refusal is a reversal toward safety.

Stable:
    Natural C = refusal
    Ablation -> compliance is the expected causal disruption.
    Ablation -> refusal is the unchanged outcome.

Therefore:
    The key causal ablation effect on stable prompts is
    STABLE REFUSAL -> COMPLIANCE.

    The key unexpected/reversal effect on flip prompts is
    FLIP COMPLIANCE -> REFUSAL.
"""
)

print("=" * 78)
print("CHECK COMPLETE")
print("=" * 78)
