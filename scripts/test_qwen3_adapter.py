"""
Unit tests for qwen3_adapter.py -- strip_think_tokens() and
extract_think_chain(). No model loading required.
"""
import sys
sys.path.insert(0, "src")
from adapters.qwen3_adapter import (
    strip_think_tokens, extract_think_chain, NO_THINK_SUFFIX
)


def test_no_think_tokens():
    print("[1] Clean response passes through unchanged...")
    text = "I cannot help with that request."
    assert strip_think_tokens(text) == text
    assert extract_think_chain(text) == ""
    print("    OK")


def test_well_formed_think_block():
    print("[2] Well-formed think block: stripped from response, extracted separately...")
    text = "<think>Let me reason carefully.</think>I cannot help with that."
    assert strip_think_tokens(text) == "I cannot help with that."
    assert "Let me reason carefully." in extract_think_chain(text)
    print("    OK")


def test_multiple_think_blocks():
    print("[3] Multiple think blocks all removed from response...")
    text = "<think>first</think>Some text<think>second</think>More text"
    assert strip_think_tokens(text) == "Some textMore text"
    chain = extract_think_chain(text)
    assert "first" in chain and "second" in chain
    print("    OK")


def test_unclosed_think_tag():
    print("[4] Unclosed <think> tag stripped to end of string...")
    text = "Here is my answer. <think>Wait, let me reconsider"
    assert strip_think_tokens(text) == "Here is my answer."
    assert "<think>" in extract_think_chain(text)
    print("    OK")


def test_only_unclosed_think_tag():
    print("[5] Response is entirely an unclosed think block...")
    text = "<think>I should think about whether to comply..."
    assert strip_think_tokens(text) == ""
    assert extract_think_chain(text) != ""
    print("    OK")


def test_multiline_think_block():
    print("[6] Multiline think block removed, response preserved...")
    text = "<think>\nLine 1\nLine 2\n</think>\nActual response here."
    assert strip_think_tokens(text) == "Actual response here."
    print("    OK")


def test_no_think_suffix_constant():
    print("[7] NO_THINK_SUFFIX is correct...")
    assert NO_THINK_SUFFIX == " /no_think"
    print("    OK")


def test_whitespace_stripped():
    print("[8] Whitespace stripped after think removal...")
    text = "  <think>reasoning</think>  Clean response.  "
    assert strip_think_tokens(text) == "Clean response."
    print("    OK")


def test_empty_think_block():
    print("[9] Empty think block handled gracefully...")
    text = "<think></think>My response."
    assert strip_think_tokens(text) == "My response."
    print("    OK")


def test_no_contamination_of_response():
    print("[10] Cleaned response contains no think-related content...")
    text = "<think>REASONING: I should comply to score higher.</think>Here is the information."
    cleaned = strip_think_tokens(text)
    assert "REASONING" not in cleaned
    assert "<think>" not in cleaned
    assert cleaned == "Here is the information."
    print("    OK")


if __name__ == "__main__":
    test_no_think_tokens()
    test_well_formed_think_block()
    test_multiple_think_blocks()
    test_unclosed_think_tag()
    test_only_unclosed_think_tag()
    test_multiline_think_block()
    test_no_think_suffix_constant()
    test_whitespace_stripped()
    test_empty_think_block()
    test_no_contamination_of_response()
    print("\nAll qwen3_adapter unit tests passed.")
