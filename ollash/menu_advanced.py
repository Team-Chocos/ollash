# ollash/menu_advanced.py
import subprocess
from typing import List, Tuple, Optional

HAS_PYFZF = False

try:
    from pyfzf.pyfzf import FzfPrompt
    HAS_PYFZF = True
except ImportError:
    pass


class MenuSelector:
    """Simple menu selector using only pyfzf"""
    
    def select_with_pyfzf(self, options: List[str], title: str) -> Optional[str]:
        """Use pyfzf wrapper - better fzf integration"""
        if not HAS_PYFZF:
            return None
            
        try:
            fzf = FzfPrompt()
            selected = fzf.prompt(
                options,
                '--prompt="🤖 {}: " --height=60% --reverse --border --info=inline'.format(title)
            )
            return selected[0] if selected else None
        except Exception as e:
            print(f"PyFZF error: {e}")
            return None
    
    def select_with_simple(self, options: List[str], title: str) -> Optional[str]:
        """Fallback simple selection menu"""
        print(f"\n🤖 {title}")
        print("═" * 60)
        
        for i, option in enumerate(options, 1):
            if "(installed)" in option:
                print(f"{i:2}. \033[32m{option}\033[0m")
            else:
                print(f"{i:2}. {option}")
        
        print("═" * 60)
        
        while True:
            try:
                choice = input("Enter number (or 'q' to quit): ").strip()
                
                if choice.lower() == 'q':
                    return None
                
                idx = int(choice) - 1
                if 0 <= idx < len(options):
                    return options[idx]
                else:
                    print(f"Please enter a number between 1 and {len(options)}")
                    
            except ValueError:
                print("Please enter a valid number")
            except KeyboardInterrupt:
                print("\nCancelled")
                return None
    
    def select(self, options: List[str], title: str, method: str = "auto") -> Optional[str]:
        """Select from options using specified method"""
        if not options:
            return None
            
        if method == "auto":
            method = "pyfzf" if HAS_PYFZF else "simple"
        
        if method == "pyfzf":
            return self.select_with_pyfzf(options, title)
        else:
            return self.select_with_simple(options, title)


def get_available_ollama_models() -> List[str]:
    """Get list of available Ollama models from their registry"""
    popular_models = [
        "llama3.3", "llama3.2", "llama3.2:1b", "llama3.2:3b",
        "llama3.1", "llama3.1:405b", "llama3", "llama2", "llama2-uncensored", "llama2:13b", "llama2:70b", "llama4",
        "gemma3", "gemma2",
        "qwen3", "qwen2.5", "qwen2", "qwen",
        "phi4", "phi4-mini", "phi3", "phi",
        "mistral", "mistral-nemo",
        "deepseek-v3", "deepseek-coder", "deepseek-coder-v2",
        "dolphin3", "dolphin-llama3", "dolphin-mixtral",
        "mixtral", "command-r", "command-r-plus",
        "granite3.3", "granite3.2",
        "smollm2", "smollm", "tinyllama",
        "codegemma", "codellama",
        "neural-chat", "starcoder2", "starling-lm", "wizardlm2",
        "devstral", "llama3-chatqa", "codeqwen", "aya", "stablelm2"
    ]

    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, check=True, timeout=10
        )
        
        installed_models = []
        lines = result.stdout.strip().split('\n')[1:]
        for line in lines:
            if line.strip():
                parts = line.split()
                if parts:
                    model_name = parts[0]
                    installed_models.append(f"{model_name} (installed)")
        
        all_models = installed_models.copy()
        for model in popular_models:
            if not any(model in installed for installed in installed_models):
                all_models.append(model)
                
        return all_models if all_models else popular_models
        
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return popular_models


def select_model_advanced(backend: str = "ollama", method: str = "auto") -> Optional[str]:
    """Select model using pyfzf or simple fallback"""
    selector = MenuSelector()
    
    if backend == "ollama":
        models = get_available_ollama_models()
        title = "Select Ollama Model"
    else:
        print(f"Backend {backend} not implemented yet")
        return None
    
    if not models:
        print(f"No models available for {backend}")
        return None
    
    selected = selector.select(models, title, method)
    
    if selected:
        # Clean up the model name
        clean_model = selected.replace(" (installed)", "")
        print(f"✅ Selected: {clean_model}")
        return clean_model
    
    return None


def get_model_selection_advanced(method: str = "auto") -> Optional[Tuple[str, str]]:
    """Enhanced model selection with pyfzf or simple fallback"""
    try:
        backend = "ollama"
        model = select_model_advanced(backend, method)
        
        if model:
            return backend, model
        return None
        
    except KeyboardInterrupt:
        print("\n👋 Selection cancelled")
        return None


if __name__ == "__main__":
    # Test the selection
    print("🎯 Testing Model Selection")
    print("=" * 50)
    
    print(f"PyFZF available: {HAS_PYFZF}")
    print()
    
    # Test selection
    result = get_model_selection_advanced()
    if result:
        backend, model = result
        print(f"\n🎉 Ready to use {model} with {backend}!")
    else:
        print("\n❌ No model selected")