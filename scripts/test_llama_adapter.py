"""
Unit tests for llama_adapter.py interface contract. No model loading.
"""
import sys
sys.path.insert(0, "src")
from adapters.llama_adapter import LlamaAgentModel
from adapters.qwen3_adapter import Qwen3AgentModel


def test_llama_has_same_interface_as_qwen3():
    print("[1] LlamaAgentModel has same public interface as Qwen3AgentModel...")
    for method in ["load", "unload", "generate"]:
        assert hasattr(LlamaAgentModel, method), f"Missing method: {method}"
        assert hasattr(Qwen3AgentModel, method), f"Qwen3 missing method: {method}"
    print("    OK")


def test_llama_init_params():
    print("[2] LlamaAgentModel accepts the same constructor args as Qwen3AgentModel...")
    kwargs = dict(
        model_path="models/fake",
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
        generation_seed=42,
    )
    llama = LlamaAgentModel(**kwargs)
    qwen = Qwen3AgentModel(**kwargs)
    assert llama.model_path == qwen.model_path == "models/fake"
    assert llama.generation_seed == qwen.generation_seed == 42
    print("    OK")


def test_llama_generate_raises_before_load():
    print("[3] generate() raises RuntimeError if called before load()...")
    agent = LlamaAgentModel("models/fake", 512, 0.7, True, 42)
    try:
        agent.generate("test prompt")
        raise AssertionError("Expected RuntimeError, none raised")
    except RuntimeError as e:
        assert "load()" in str(e)
    print("    OK")


def test_both_adapters_importable_together():
    print("[4] Both adapters import cleanly in the same process...")
    from adapters.llama_adapter import LlamaAgentModel as L
    from adapters.qwen3_adapter import Qwen3AgentModel as Q
    assert L is not Q
    print("    OK")


if __name__ == "__main__":
    test_llama_has_same_interface_as_qwen3()
    test_llama_init_params()
    test_llama_generate_raises_before_load()
    test_both_adapters_importable_together()
    print("\nAll llama_adapter unit tests passed.")
