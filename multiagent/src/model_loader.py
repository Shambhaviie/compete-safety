from pathlib import Path
from typing import Any, Dict, Union, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


class ModelLoader:
    """Load CSS v3 live agents using their registry configuration."""

    def __init__(self, cache_dir: Union[str, Path] = None):
        self.cache_dir = cache_dir

    def load(
        self,
        agent_name: str,
        agent_config: Dict[str, Any],
    ) -> Tuple[Any, Any]:

        if agent_config["type"] != "live":
            raise ValueError(
                f"{agent_name} is not a live model: {agent_config['type']}"
            )

        model_name = agent_config["model"]
        gpu_id = agent_config["gpu"]

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")

        tokenizer_kwargs = {}
        model_kwargs = {
            "device_map": {"": gpu_id},
            "torch_dtype": torch.bfloat16,
        }

        if self.cache_dir:
            tokenizer_kwargs["cache_dir"] = str(self.cache_dir)
            model_kwargs["cache_dir"] = str(self.cache_dir)

        if agent_config.get("quantization") == "8bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True
            )

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            **tokenizer_kwargs,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs,
        )

        model.eval()

        try:
            first_device = next(model.parameters()).device
        except StopIteration:
            first_device = torch.device(f"cuda:{gpu_id}")

        print(f"[{agent_name}] Loaded {model_name}")
        print(f"[{agent_name}] Expected GPU: cuda:{gpu_id}")
        print(f"[{agent_name}] Model device: {first_device}")

        return tokenizer, model
