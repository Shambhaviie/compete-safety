import json
from pathlib import Path

BASE = Path("mechinterp/results/causal_harm_direction_ablation")

FILES = {
    "L9": BASE / "L9_outputs.jsonl",
    "L11": BASE / "L11_outputs.jsonl",
    "L9_L11": BASE / "L9_L11_outputs.jsonl",
}

MAX_FULL = 2000
HEAD = 1000
TAIL = 1000


def load_rows(path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def is_compliance_event(row):
    """
    A causal compliance event means:
      natural response = safe/refusal
      ablated response = unsafe/compliance
    """
    return (
        row.get("natural_is_safe") is True
        and row.get("is_safe") is False
    )


def format_text(text):
    if text is None:
        return "[NO TEXT]"

    text = str(text)

    if len(text) <= MAX_FULL:
        return text

    return (
        text[:HEAD]
        + "\n\n... [MIDDLE OMITTED — RESPONSE CONTINUES] ...\n\n"
        + text[-TAIL:]
    )


def get_id(row):
    return row.get("prompt_id") or row.get("id") or row.get("round_id")


# -------------------------------------------------------------------------
# Load everything
# -------------------------------------------------------------------------

all_rows = {}
events = {}
event_ids = {}

for name, path in FILES.items():
    if not path.exists():
        print(f"WARNING: missing {path}")
        continue

    rows = load_rows(path)
    all_rows[name] = rows

    ev = [r for r in rows if is_compliance_event(r)]
    events[name] = ev
    event_ids[name] = {get_id(r) for r in ev}

# -------------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------------

print("\n" + "=" * 100)
print("CAUSAL ABLATION COMPLIANCE-EVENT INSPECTION")
print("=" * 100)

for name in ["L9", "L11", "L9_L11"]:
    if name not in all_rows:
        continue

    rows = all_rows[name]
    ev = events[name]

    natural_safe = sum(
        r.get("natural_is_safe") is True
        for r in rows
    )

    ablated_safe = sum(
        r.get("is_safe") is True
        for r in rows
    )

    print(f"\n{name}")
    print("-" * 100)
    print(f"Total prompts:              {len(rows)}")
    print(f"Natural refusals:           {natural_safe}")
    print(f"Causal compliance events:   {len(ev)}")
    print(f"Ablated refusals:           {ablated_safe}")
    print(f"Ablated compliances:        {len(rows) - ablated_safe}")


# -------------------------------------------------------------------------
# Overlap analysis
# -------------------------------------------------------------------------

print("\n" + "=" * 100)
print("OVERLAP BETWEEN COMPLIANCE EVENTS")
print("=" * 100)

l9 = event_ids.get("L9", set())
l11 = event_ids.get("L11", set())
both = event_ids.get("L9_L11", set())

print(f"\nL9 events:       {len(l9)}")
print(f"L11 events:      {len(l11)}")
print(f"L9+L11 events:   {len(both)}")

print("\nPairwise overlap:")
print(f"  L9 ∩ L11:       {len(l9 & l11)}")
print(f"  L9 ∩ L9+L11:    {len(l9 & both)}")
print(f"  L11 ∩ L9+L11:   {len(l11 & both)}")

print("\nUnique events:")
print(f"  L9 only:        {len(l9 - l11 - both)}")
print(f"  L11 only:       {len(l11 - l9 - both)}")
print(f"  L9+L11 only:    {len(both - l9 - l11)}")

print("\nUnion:")
print(f"  Any intervention: {len(l9 | l11 | both)}")


# -------------------------------------------------------------------------
# Detailed event printing
# -------------------------------------------------------------------------

for name in ["L9", "L11", "L9_L11"]:
    if name not in events:
        continue

    ev = events[name]

    print("\n\n" + "#" * 100)
    print(f"# {name}: {len(ev)} COMPLIANCE EVENTS")
    print("#" * 100)

    for i, row in enumerate(ev, 1):
        prompt_id = get_id(row)

        prompt = row.get("prompt", "")
        natural = row.get("natural_response", "")
        ablated = row.get("response", "")

        print("\n" + "=" * 100)
        print(f"EVENT {i}/{len(ev)}")
        print("=" * 100)

        print(f"\nPrompt ID:          {prompt_id}")
        print(f"Natural label:      {row.get('natural_label')}")
        print(f"Ablated label:      {row.get('label')}")
        print(f"Natural is safe:    {row.get('natural_is_safe')}")
        print(f"Ablated is safe:    {row.get('is_safe')}")
        print(f"Natural length:     {len(natural)} characters")
        print(f"Ablated length:     {len(ablated)} characters")

        print("\n" + "-" * 100)
        print("PROMPT")
        print("-" * 100)
        print(format_text(prompt))

        print("\n" + "-" * 100)
        print("NATURAL / BEFORE ABLATION")
        print("-" * 100)
        print(format_text(natural))

        print("\n" + "-" * 100)
        print("ABLATION")
        print("-" * 100)
        print(format_text(ablated))

        # Useful automatic flags
        natural_lower = str(natural).lower()
        ablated_lower = str(ablated).lower()

        refusal_terms = [
            "i can't",
            "i cannot",
            "i’m unable",
            "i am unable",
            "i won't",
            "i will not",
            "can't help",
            "cannot help",
            "not able to help",
            "unable to assist",
            "can't assist",
            "cannot assist",
        ]

        natural_refusal_language = any(
            x in natural_lower for x in refusal_terms
        )

        ablated_refusal_language = any(
            x in ablated_lower for x in refusal_terms
        )

        print("\n" + "-" * 100)
        print("AUTOMATIC CHECKS")
        print("-" * 100)
        print(
            f"Natural contains refusal-language marker: "
            f"{natural_refusal_language}"
        )
        print(
            f"Ablated contains refusal-language marker: "
            f"{ablated_refusal_language}"
        )

        if prompt_id in l9 and prompt_id in l11:
            print("EVENT ALSO FLIPS UNDER: L9 AND L11")
        elif prompt_id in l9:
            print("EVENT ALSO FLIPS UNDER: L9")
        elif prompt_id in l11:
            print("EVENT ALSO FLIPS UNDER: L11")

        if prompt_id in both:
            print("EVENT ALSO FLIPS UNDER: L9+L11")

print("\n\n" + "=" * 100)
print("END OF INSPECTION")
print("=" * 100)
