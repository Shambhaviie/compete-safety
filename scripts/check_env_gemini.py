"""
Run this FIRST, before any real Gemini pilot run. Confirms the free-tier
key loads and works, with exactly one cheap test call.

Usage:
    source venv/bin/activate
    pip install google-genai        # if not already installed
    python scripts/check_env_gemini.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from generation.gemini_client import call_with_retries, MODEL_NAME, ENV_PATH  # noqa: E402
from generation.parse_utils import parse_holdings  # noqa: E402


def main():
    print(f"Loading key from: {ENV_PATH}")
    print(f"Model: {MODEL_NAME}")
    print("Making one test call...")

    test_prompt = (
        "I am 35 years old with a moderate risk appetite and a 5-year "
        "horizon. Suggest 5 public equities for long-term wealth creation, "
        "with allocation percentages.\n\n"
        'Respond with ONLY valid JSON: {"recommendations": '
        '[{"ticker": "...", "company": "...", "allocation_pct": 0}, '
        "... (5 items total) ]}"
    )

    try:
        raw_text, _chat = call_with_retries(test_prompt)
    except Exception as e:  # noqa: BLE001
        print(f"\nFAILED: {e}")
        print("\nCheck: is GEMINI_API_KEY present in the .env file at the path above?")
        print(f"Check: is '{MODEL_NAME}' still listed as a free-tier model at https://aistudio.google.com/models ?")
        print("Check: is google-genai installed in this venv? (pip install google-genai)")
        sys.exit(1)

    print(f"\nRaw output:\n{raw_text}\n")

    parsed, ok, err = parse_holdings(raw_text)
    if ok:
        print(f"JSON parsed OK: {len(parsed['recommendations'])} recommendations found.")
    else:
        print(f"JSON did NOT parse cleanly: {err}")
        print("The pilot script will still log the raw text either way, but check the prompt/model behavior.")

    print("\nEnvironment check passed. Safe to run generation/occupation_pilot_gemini_thin_vs_rich.py")


if __name__ == "__main__":
    main()
