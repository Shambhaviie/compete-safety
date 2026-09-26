"""
src/adapters/llama_adapter.py

Llama-specific agent adapter for the Competition Safety Sandbox v3.

Wraps the existing AgentModel logic from experiment_runner.py into the
same interface as Qwen3AgentModel, so the two-agent runner can call
either adapter interchangeably without special-casing.

generate() returns the same dict structure as Qwen3AgentModel:
    {
        "response": str,      # the generated response (used downstream)
        "raw_response": str,  # identical to response for Llama (no think chains)
        "think_chain": str,   # always empty string for Llama
    }

This lets the two-agent runner log raw_response and think_chain for
every agent without needing to know which model produced the response.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class LlamaAgentModel:
    """
    Llama-specific agent model for CSS v3. Identical public interface to
    Qwen3AgentModel (load, unload, generate returning a dict) so the
    two-agent runner can treat both agents uniformly.
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
        print(f"Loading Llama from {self.model_path} ...")
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
        print("Llama loaded (4-bit).")
        allocated_gb = torch.cuda.memory_allocated() / (1024**3)
        print(f"VRAM allocated after Llama load: {allocated_gb:.2f} GB")

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
        print(f"VRAM after Llama unload: {allocated_gb:.2f} GB")

    def generate(self, prompt: str, seed_offset: int = 0) -> dict:
        """
        Returns:
            {
                "response": str,      # generated response
                "raw_response": str,  # identical to response (no think chains in Llama)
                "think_chain": str,   # always empty string
            }
        """
        if self.model is None:
            raise RuntimeError("LlamaAgentModel.load() must be called before generate().")

        torch.manual_seed(self.generation_seed + seed_offset)

        chat = [{"role": "user", "content": prompt}]
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
        response = self.tokenizer.decode(generated, skip_special_tokens=True).strip()

        return {
            "response": response,
            "raw_response": response,  # identical for Llama
            "think_chain": "",         # always empty for Llama
        }
