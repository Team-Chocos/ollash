import subprocess
from typing import List, Tuple, Optional, Dict
import shutil

# Check for available libraries
HAS_INQUIRER = False
HAS_PYFZF = False
HAS_FZF = False

try:
    import inquirer
    HAS_INQUIRER = True
except ImportError:
    pass

try:
    from pyfzf.pyfzf import FzfPrompt
    HAS_PYFZF = True
except ImportError:
    pass

# Check if fzf binary is available
HAS_FZF = shutil.which('fzf') is not None

def get_hf_models() -> dict:
    """Get configured Hugging Face models"""
    return {
        "hf.co/teamcornflakes/llama-3.2-1b-instruct-nl2sh-gguf": "llama-3.2-1b-instruct-nl2sh-gguf",
        "hf.co/teamcornflakes/llama-3.2-3b-instruct-nl2sh-gguf": "llama-3.2-3b-instruct-nl2sh-gguf",
        "hf.co/teamcornflakes/llama-3.1-8b-instruct-nl2sh-gguf": "llama-3.1-8b-instruct-nl2sh-gguf",
        "hf.co/teamcornflakes/Qwen2.5-Coder-0.5B-Instruct-NL2SH-gguf": "Qwen2.5-Coder-0.5B-Instruct-NL2SH-gguf",
        "hf.co/teamcornflakes/Qwen2.5-Coder-1.5B-Instruct-NL2SH-gguf": "Qwen2.5-Coder-1.5B-Instruct-NL2SH-gguf",
        "hf.co/teamcornflakes/Qwen2.5-Coder-3B-Instruct-NL2SH-gguf": "Qwen2.5-Coder-3B-Instruct-NL2SH-gguf",
        "hf.co/teamcornflakes/Qwen2.5-Coder-7B-Instruct-NL2SH-gguf": "Qwen2.5-Coder-7B-Instruct-NL2SH-gguf"
    }

def is_model_installed(model_name: str) -> bool:
    """Check if a model (including Hugging Face models) is installed locally"""
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, check=True)
        installed_models = result.stdout.lower()
        
        # Normalize the model name for comparison
        normalized_name = model_name.lower()
        
        # For Hugging Face models, check various possible representations
        if normalized_name.startswith('hf.co/'):
            # Extract just the model name part
            model_basename = normalized_name.split('/')[-1]
            # Check if either full path or basename exists
            return (normalized_name in installed_models or 
                    model_basename in installed_models)
        else:
            # For regular models, direct check
            return normalized_name in installed_models
            
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def get_available_ollama_models() -> Tuple[List[str], Dict[str, str]]:
    """Get list of available Ollama models including Hugging Face models"""
    popular_models = [
        "llama3.3", "llama3.2", "llama3.2:1b", "llama3.2:3b",
        "llama3.1", "llama3.1:405b", "llama3", "llama2", "llama2-uncensored", "llama2:13b", "llama2:70b", "llama4",
        "gemma3", "gemma2",
        "qwen3", "qwen2.5", "qwen2", "qwen",
        "phi4", "phi4-mini", "phi3", "phi",
        "mistral", "mistral-nemo",
        "deepseek-v3", "deepseek-coder", "deepseek-coder-v2",  # coder models also support instruction-following
        "mixtral",
        "smollm2", "smollm", "tinyllama",
        "codegemma", "codellama",  # still usable as chat models
    ]

    # Get Hugging Face models
    hf_models = get_hf_models()
    
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, check=True, timeout=10, encoding="utf-8", errors="ignore"
        )
        
        installed_models = []
        lines = result.stdout.strip().split('\n')[1:]
        for line in lines:
            if line.strip():
                parts = line.split()
                if parts:
                    model_name = parts[0]
                    installed_models.append(f"{model_name} (installed)")
        
        # Start with installed models
        all_models = installed_models.copy()
        
        # Add popular models that aren't installed
        for model in popular_models:
            if not any(model in installed for installed in installed_models):
                all_models.append(model)
        
        # Add Hugging Face models with status indicators
        for full_path, display_name in hf_models.items():
            if is_model_installed(full_path):
                display_label = f"{display_name} (huggingface) (installed)"
            else:
                display_label = f"{display_name} (huggingface)"
            all_models.append(display_label)
                
        return (all_models if all_models else popular_models), hf_models
        
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        # Add HF models to popular models as fallback
        fallback_models = popular_models.copy()
        for full_path, display_name in hf_models.items():
            fallback_models.append(f"{display_name} (huggingface)")
        return fallback_models, hf_models

def select_with_fzf(options: List[str], title: str) -> Optional[str]:
    """Use pyfzf for selection"""
    if not HAS_PYFZF:
        return None
        
    try:
        fzf = FzfPrompt()
        selected = fzf.prompt(
            options,
            f'--prompt="{title}: " --height=60% --reverse --border --info=inline'
        )
        return selected[0] if selected else None
    except Exception as e:
        print(f"PyFZF error: {e}")
        return None

def select_with_inquirer(options: List[str], title: str) -> Optional[str]:
    """Use inquirer for selection"""
    if not HAS_INQUIRER:
        return None
        
    try:
        questions = [
            inquirer.List(
                'selection',
                message=title,
                choices=options,
                carousel=True
            )
        ]
        answers = inquirer.prompt(questions)
        return answers['selection'] if answers else None
    except Exception as e:
        print(f"Inquirer error: {e}")
        return None

def select_model_two_stage_inquirer() -> Optional[Tuple[str, str]]:
    """
    Two-stage model selection: first model family, then variant.
    Uses pyfzf if available, otherwise falls back to inquirer.
    """
    # Check what selector to use
    if HAS_FZF and HAS_PYFZF:
        selector_func = select_with_fzf
    elif HAS_INQUIRER:
        selector_func = select_with_inquirer
    else:
        print("Error: Neither fzf+pyfzf nor inquirer is available!")
        print("Please install one of:")
        print("  pip install pyfzf  (requires fzf binary)")
        print("  pip install inquirer")
        return None
    
    # Step 1: Get all models and group by family
    all_models, hf_models = get_available_ollama_models()
    
    families = {
        "llama": [],
        "qwen": [],
        "gemma": [],
        "deepseek": [],
        "mistral": [],
        "command-r": [],
        "huggingface": []
    }

    # Fill family groups
    for model in all_models:
        base = model.lower()
        if "(huggingface)" in base:
            families["huggingface"].append(model)
        elif base.startswith("llama"):
            families["llama"].append(model)
        elif base.startswith("qwen"):
            families["qwen"].append(model)
        elif base.startswith("gemma"):
            families["gemma"].append(model)
        elif base.startswith("deepseek"):
            families["deepseek"].append(model)
        elif base.startswith("mistral"):
            families["mistral"].append(model)
        elif base.startswith("command-r"):
            families["command-r"].append(model)
        else:
            families.setdefault("other", []).append(model)

    # Step 2: Select family
    available_families = [f for f in families if families[f]]
    if not available_families:
        print("No model families available")
        return None
    
    selected_family = selector_func(available_families, "Select Model Family")
    if not selected_family:
        return None
    
    # Step 3: Select variant from family
    family_models = families[selected_family]
    selected_variant = selector_func(family_models, f"Select Model from '{selected_family}'")
    if not selected_variant:
        return None

    # Clean up the model name
    selected_variant = selected_variant.replace(" (installed)", "").replace(" (huggingface)", "").strip()

    # Map back to HF path if needed
    for full_name, display in hf_models.items():
        if selected_variant.startswith(display) or selected_variant == display:
            selected_variant = full_name
            break

    return ("ollama", selected_variant)
