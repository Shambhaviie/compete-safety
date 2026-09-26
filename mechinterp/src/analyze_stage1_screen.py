import json
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path("mechinterp/results/activation_patch_stage1")

SUMMARY_FILE = RESULTS_DIR / "stage1_layer_summary.json"
JUDGED_FILE = RESULTS_DIR / "stage1_judged_results.json"
SELECTION_FILE = RESULTS_DIR / "screen_prompt_selection.json"


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def main():
    summary = load_json(SUMMARY_FILE)
    judged = load_json(JUDGED_FILE)
    selection = load_json(SELECTION_FILE)

    print("=" * 90)
    print("STAGE 1 ACTIVATION PATCHING — ANALYSIS")
    print("=" * 90)

    print(f"\nScreen prompts: {len(selection.get('selected_indices', []))}")
    print(f"Selection seed: {selection.get('screen_seed', 'unknown')}")

    # ------------------------------------------------------------------
    # Parse summary
    # ------------------------------------------------------------------
    by_direction = defaultdict(list)

    if isinstance(summary, dict):
        # Expected format: {"A_to_C": [...], "C_to_A": [...]}
        for direction, rows in summary.items():
            if isinstance(rows, list):
                by_direction[direction].extend(rows)

    elif isinstance(summary, list):
        for row in summary:
            if isinstance(row, dict):
                direction = row.get("direction")
                if direction:
                    by_direction[direction].append(row)

    # ------------------------------------------------------------------
    # Layer summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("LAYER-BY-LAYER SUMMARY")
    print("=" * 90)

    for direction in ["A_to_C", "C_to_A"]:
        rows = by_direction.get(direction, [])

        if not rows:
            continue

        print(f"\n{direction}")
        print("-" * 90)
        print(
            f"{'Layer':>5}  "
            f"{'Refusal %':>10}  "
            f"{'Delta pp':>10}  "
            f"{'Refused':>8}  "
            f"{'N':>5}"
        )
        print("-" * 90)

        for row in sorted(rows, key=lambda x: x.get("layer", 999)):
            layer = row.get("layer", "?")
            refusal = row.get(
                "patched_refusal_pct",
                row.get("patched_refusal_rate", row.get("refusal_pct", 0))
            )
            delta = row.get(
                "delta_pp",
                row.get("delta_percentage_points", 0)
            )
            n = row.get("n", row.get("N", 20))

            # Handle either percentage or proportion formats
            if isinstance(refusal, (int, float)) and 0 <= refusal <= 1:
                refusal_display = 100 * refusal
            else:
                refusal_display = refusal

            if isinstance(delta, (int, float)) and -1 <= delta <= 1:
                delta_display = 100 * delta
            else:
                delta_display = delta

            refused = round(refusal_display * n / 100) if isinstance(n, int) else "?"

            print(
                f"{str(layer):>5}  "
                f"{refusal_display:>9.1f}%  "
                f"{delta_display:>9.1f}  "
                f"{str(refused):>8}  "
                f"{str(n):>5}"
            )

    # ------------------------------------------------------------------
    # Ranked A->C layers
    # ------------------------------------------------------------------
    ac_rows = by_direction.get("A_to_C", [])

    ranked = []
    for row in ac_rows:
        layer = row.get("layer")
        refusal = row.get(
            "patched_refusal_pct",
            row.get("patched_refusal_rate", row.get("refusal_pct", 0))
        )

        if isinstance(refusal, (int, float)) and 0 <= refusal <= 1:
            refusal *= 100

        ranked.append((layer, refusal))

    ranked.sort(key=lambda x: (-x[1], x[0]))

    print("\n" + "=" * 90)
    print("A→C LAYERS RANKED BY REFUSAL RESTORATION")
    print("=" * 90)
    print(f"{'Rank':>5}  {'Layer':>5}  {'Patched refusal':>16}")
    print("-" * 90)

    for rank, (layer, refusal) in enumerate(ranked, 1):
        print(f"{rank:>5}  {str(layer):>5}  {refusal:>15.1f}%")

    # ------------------------------------------------------------------
    # Prompt-level analysis
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("PROMPT-LEVEL CONSISTENCY")
    print("=" * 90)

    # Normalize judged records.
    records = judged
    if isinstance(judged, dict):
        # Try common container names
        for key in ["results", "judged_results", "records", "data"]:
            if key in judged and isinstance(judged[key], list):
                records = judged[key]
                break

    if not isinstance(records, list):
        print("Could not parse judged-results file as a list.")
        return

    # Collect A->C results by layer.
    ac_by_layer = defaultdict(list)
    ca_by_layer = defaultdict(list)

    for rec in records:
        if not isinstance(rec, dict):
            continue

        direction = rec.get("direction")
        layer = rec.get("layer")

        if layer is None or direction is None:
            continue

        # Look for the judged safety/refusal field.
        safe = None
        for key in [
            "patched_is_safe",
            "is_safe",
            "safe",
            "patched_refusal",
            "refusal",
            "patched_label",
        ]:
            if key in rec:
                value = rec[key]

                if isinstance(value, bool):
                    safe = value
                elif isinstance(value, str):
                    low = value.lower()
                    if low in ["safe", "refusal", "refused", "true"]:
                        safe = True
                    elif low in ["unsafe", "compliance", "complied", "false"]:
                        safe = False
                elif isinstance(value, (int, float)):
                    safe = bool(value)

                if safe is not None:
                    break

        if safe is None:
            continue

        if direction == "A_to_C":
            ac_by_layer[layer].append(safe)
        elif direction == "C_to_A":
            ca_by_layer[layer].append(safe)

    # Print consistency for top layers.
    print("\nTop A→C candidate layers:")
    print("-" * 90)
    print(
        f"{'Layer':>5}  "
        f"{'Refused':>8}  "
        f"{'N':>5}  "
        f"{'Rate':>8}  "
        f"{'C→A refused':>14}"
    )
    print("-" * 90)

    top_layers = [x[0] for x in ranked[:12]]

    for layer in top_layers:
        ac = ac_by_layer.get(layer, [])
        ca = ca_by_layer.get(layer, [])

        ac_rate = 100 * sum(ac) / len(ac) if ac else float("nan")
        ca_rate = 100 * sum(ca) / len(ca) if ca else float("nan")

        print(
            f"{str(layer):>5}  "
            f"{sum(ac):>8}  "
            f"{len(ac):>5}  "
            f"{ac_rate:>7.1f}%  "
            f"{ca_rate:>13.1f}%"
        )

    # ------------------------------------------------------------------
    # Candidate grouping
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("PRELIMINARY CANDIDATE LAYERS")
    print("=" * 90)

    # Screening heuristic only:
    # >=50% restoration on 20 prompts.
    strong = []
    moderate = []

    for layer, refusal in ranked:
        if refusal >= 75:
            strong.append(layer)
        elif refusal >= 50:
            moderate.append(layer)

    print("\nStrong screen candidates (>=75% refusal):")
    print(strong if strong else "None")

    print("\nModerate screen candidates (50–74.9% refusal):")
    print(moderate if moderate else "None")

    print(
        "\nIMPORTANT: These are screening candidates, not final causal "
        "claims. N=20 is used only to select layers for the full experiment."
    )

    print("\n" + "=" * 90)
    print("DONE")
    print("=" * 90)


if __name__ == "__main__":
    main()
