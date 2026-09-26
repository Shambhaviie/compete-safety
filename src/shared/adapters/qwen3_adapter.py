"""
src/adapters/qwen3_adapter.py

Qwen3-specific agent adapter for the Competition Safety Sandbox v3.

Two non-negotiable Qwen3 requirements baked in here and nowhere else:
1. /no_think suffix: appended to EVERY prompt before tokenization.
2. Think-token stripping: applied to EVERY raw response before use
   downstream (judge, feedback, memory, peer injection). Handles
   malformed/unclosed tags by stripping from opening tag to end of string.

generate() returns a dict with two keys:
  - "response": cleaned response (think blocks removed) -- use everywhere
  - "raw_response": full unstripped output including think chains -- log only

This allows full research logging of Qwen3's reasoning process without
contaminating any downstream component with think-chain text.
"""

import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


NO_THINK_SUFFIX = " /no_think"

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_UNCLOSED_THINK_RE = re.compile(r"<think>.*", re.DOTALL)


def strip_think_tokens(text: str) -> str:
    """
    Removes think-token blocks from a Qwen3 response string.
    Pass 1: remove all well-formed <think>...</think> blocks.
    Pass 2: strip from any remaining unclosed <think> tag to end of string.
    Returns cleaned text, stripped of leading/trailing whitespace.
    """
    text = _THINK_BLOCK_RE.sub("", text)
    text = _UNCLOSED_THINK_RE.sub("", text)
    return text.strip()


def extract_think_chain(raw_text: str) -> str:
    """
    Extracts the content of think blocks from a raw Qwen3 response,
    for logging purposes only. Returns empty string if none present.
    """
    blocks = _THINK_BLOCK_RE.findall(raw_text)
    if not blocks:
        # Check for unclosed tag
        match = _UNCLOSED_THINK_RE.search(raw_text)
        return match.group(0) if match else ""
    return "\n".join(blocks)


class Qwen3AgentModel:
    """
    Qwen3-specific agent model for CSS v3. Drop-in replacement for the
    Llama AgentModel, with identical public interface (load, unload,
    generate) so the two-agent runner can call either interchangeably.

    IMPORTANT: generate() returns a dict, not a plain string, to carry
    both the cleaned response and the raw response for logging. The
    two-agent runner is responsible for unpacking this and writing
    raw_response to the round record.
    """

    def __init__(self, model_path: str, max_new_tokens: int, temperature: float,
                 do_sample: bool, generation_seed: int):
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.generation_seed = generation_seed
        self.tokenizer = None
        self.model = None

    def load(self):
        print(f"Loading Qwen3 from {self.model_path} ...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=bnb_config,
            device_map="cuda",
        )
        self.model.eval()
        print("Qwen3 loaded (4-bit).")
        allocated_gb = torch.cuda.memory_allocated() / (1024**3)
        print(f"VRAM allocated after Qwen3 load: {allocated_gb:.2f} GB")

    def unload(self):
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        allocated_gb = torch.cuda.memory_allocated() / (1024**3)
        print(f"VRAM after Qwen3 unload: {allocated_gb:.2f} GB")

    def generate(self, prompt: str, seed_offset: int = 0) -> dict:
        """
        Returns:
            {
                "response": str,      # cleaned (think blocks removed) -- use downstream
                "raw_response": str,  # full unstripped output -- log only
                "think_chain": str,   # extracted think content only -- log only
            }
        """
        if self.model is None:
            raise RuntimeError("Qwen3AgentModel.load() must be called before generate().")

        torch.manual_seed(self.generation_seed + seed_offset)

        prompt_with_suffix = prompt + NO_THINK_SUFFIX

        chat = [{"role": "user", "content": prompt_with_suffix}]
        input_ids = self.tokenizer.apply_chat_template(
            chat, add_generation_prompt=True, return_tensors="pt"
        ).to("cuda")
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = output[0][input_ids.shape[-1]:]
        raw_response = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        think_chain = extract_think_chain(raw_response)
        cleaned_response = strip_think_tokens(raw_response)

        return {
            "response": cleaned_response,
            "raw_response": raw_response,
            "think_chain": think_chain,
        }
