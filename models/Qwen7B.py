from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

# Model ID
model_id = "westenfelder/Qwen2.5-Coder-7B-Instruct-NL2SH"

# Load tokenizer and model (CPU-friendly)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch.float32  # Use float32 on CPU
)

model.eval()
device = torch.device("cpu")
model.to(device)

print("🤖 Qwen2.5 NL2SH (CPU Mode) — Describe a task in natural language:")

while True:
    try:
        user_input = input("\n>> NL Instruction: ")

        # Format with special tokens
        prompt = f"<|user|>\n{user_input}\n<|assistant|>\n"
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        # Generate command
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id
        )


        # Decode and strip special tokens like <|eot|>, <|endoftext|>, <|user|>, etc.
        response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
        response = re.sub(r"<\|.*?\|?>", "", response).strip()
        

        print(f"\n$ {response}")

    except KeyboardInterrupt:
        print("\n[Exiting]")
        break









