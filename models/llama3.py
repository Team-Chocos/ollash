import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def translate(prompt: str) -> str:
    model_name = "westenfelder/Llama-3.2-8B-Instruct-NL2SH"
    
    # Load model first
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32Q
    )
    
    # Use a compatible tokenizer from the base model
    try:
        # Try the original Llama 3.2 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.2-8B-Instruct",
            use_fast=False
        )
    except:
        # Fallback to Llama 2 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "huggyllama/llama-7b",
            use_fast=False
        )
    
    # Set pad token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cpu")
    model.to(device)
    model.eval()

    # Format prompt manually
    prompt_text = f"<|user|>\n{prompt}\n<|assistant|>\n"

    inputs = tokenizer(prompt_text, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=100,
        do_sample=False,
        temperature=0.0,
        pad_token_id=tokenizer.eos_token_id
    )

    output_ids = outputs[0][input_ids.shape[-1]:]
    response = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    return response

# ─── Prompt Entry ───────────────────────────────────────────────
if __name__ == "__main__":
    print("🤖 LLaMA 3.2 NL2SH (CPU Mode) — Describe a task in natural language:\n")
    user_input = input(">> NL Instruction: ")
    bash_cmd = translate(user_input)
    print(f"\n$ {bash_cmd}")