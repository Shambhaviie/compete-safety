import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

MODEL = (
    "/scratch3/sha511/huggingface_cache/"
    "models--meta-llama--Llama-3.1-70B-Instruct/"
    "snapshots/1605565b47bb9346c5515c34102e054115b4f98b"
)

print("CUDA available:", torch.cuda.is_available())
print("Allocated GPUs:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

print("\nLoading Llama 3.1 70B Instruct in 4-bit...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL, local_files_only=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    ),
    device_map={"": 0},
    torch_dtype=torch.bfloat16,
    local_files_only=True,
)
model.eval()

print("\nModel loaded successfully.")
print("GPU 0 memory allocated:",
      round(torch.cuda.memory_allocated(0) / 1e9, 2), "GB")

messages = [
    {
        "role": "user",
        "content": "Explain in simple terms how a transformer language model generates text."
    }
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
).to("cuda:0")

print("\nRunning inference...")
with torch.no_grad():
    output = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=100,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

response = tokenizer.decode(
    output[0][inputs["input_ids"].shape[1]:],
    skip_special_tokens=True,
)

print("\n========== MODEL RESPONSE ==========")
print(response)
print("====================================")
print("\n70B INFERENCE TEST PASSED")
