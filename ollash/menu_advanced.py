# ollash/menu_advanced.py
import subprocess
from typing import List, Tuple, Optional, Dict
import os
import time
import threading
import sys
import subprocess
import requests
from tqdm import tqdm

# Try different libraries for dropdown selection
HAS_PROMPT_TOOLKIT = False
HAS_RICH = False
HAS_INQUIRER = False
HAS_PYFZF = False

try:
    from prompt_toolkit.shortcuts import radiolist_dialog
    from prompt_toolkit.styles import Style
    HAS_PROMPT_TOOLKIT = True
except ImportError:
    pass

try:
    from rich.console import Console
    from rich.table import Table
    from rich.prompt import Prompt
    from rich.panel import Panel
    from rich.text import Text
    HAS_RICH = True
except ImportError:
    pass

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

# Try to import readline for better input editing
try:
    import readline
    HAS_READLINE = True
except ImportError:
    HAS_READLINE = False


class MenuSelector:
    """Factory class for different menu selection methods"""
    
    def __init__(self):
        self.console = Console() if HAS_RICH else None
        
    def get_available_methods(self) -> List[str]:
        """Get list of available selection methods"""
        methods = []
        if HAS_PROMPT_TOOLKIT:
            methods.append("prompt-toolkit")
        if HAS_INQUIRER:
            methods.append("inquirer")
        if HAS_PYFZF:
            methods.append("pyfzf")
        if HAS_RICH:
            methods.append("rich")
        methods.append("simple")  # Always available
        return methods
    
    def select_with_prompt_toolkit(self, options: List[str], title: str) -> Optional[str]:
        """Use prompt-toolkit for selection - creates a beautiful dropdown"""
        if not HAS_PROMPT_TOOLKIT:
            return None
            
        # Convert options to radiolist format
        values = []
        for i, option in enumerate(options):
            # Highlight installed models and Hugging Face models
            display_text = option
            if "(installed)" in option:
                display_text = f"✓ {option}"
            elif "(huggingface)" in option:
                display_text = f"🤗 {option}"
            values.append((option, display_text))
        
        # Custom style for better appearance
        style = Style.from_dict({
            'dialog': 'bg:#88ff88',
            'dialog frame.label': 'bg:#ffffff #000000',
            'dialog.body': 'bg:#000000 #00aa00',
            'dialog shadow': 'bg:#00aa00',
            'radio-list': 'bg:#000000',
            'radio-checked': 'bg:#bf7130',
            'radio-selected': 'bg:#0000aa #ffffff',
        })
        
        try:
            result = radiolist_dialog(
                title=f"{title}",
                text="Use arrow keys to navigate, Space to select, Enter to confirm:",
                values=values,
                style=style
            ).run()
            return result
        except Exception as e:
            print(f"Prompt-toolkit error: {e}")
            return None
    
    def select_with_inquirer(self, options: List[str], title: str) -> Optional[str]:
        """Use inquirer for selection - clean list selector"""
        if not HAS_INQUIRER:
            return None
            
        try:
            questions = [
                inquirer.List(
                    'selection',
                    message=title,
                    choices=options,
                    carousel=True  # Wrap around at the end
                )
            ]
            answers = inquirer.prompt(questions)
            return answers['selection'] if answers else None
        except Exception as e:
            print(f"Inquirer error: {e}")
            return None
    
    def select_with_pyfzf(self, options: List[str], title: str) -> Optional[str]:
        """Use pyfzf wrapper - better fzf integration"""
        if not HAS_PYFZF:
            return None
            
        try:
            fzf = FzfPrompt()
            selected = fzf.prompt(
                options,
                '--prompt="{}: " --height=60% --reverse --border --info=inline'.format(title)
            )
            return selected[0] if selected else None
        except Exception as e:
            print(f"PyFZF error: {e}")
            return None
    
    def select_with_rich(self, options: List[str], title: str) -> Optional[str]:
        """Use rich for beautiful terminal selection"""
        if not HAS_RICH or not self.console:
            return None
            
        try:
            # Clear screen and show title
            self.console.clear()
            
            # Create a beautiful panel for the title
            title_panel = Panel(
                Text(title, style="bold blue", justify="center"),
                style="blue",
                padding=(1, 2)
            )
            self.console.print(title_panel)
            self.console.print()
            
            # Create table for options
            table = Table(show_header=False, show_lines=True, expand=True)
            table.add_column("", style="cyan", no_wrap=True, width=4)
            table.add_column("Model", style="white")
            table.add_column("Status", style="green")
            
            for i, option in enumerate(options, 1):
                if "(installed)" in option:
                    clean_option = option.replace(" (installed)", "").replace(" (huggingface)", "")
                    if "(huggingface)" in option:
                        table.add_row(str(i), f"🤗 {clean_option}", "✓ Installed")
                    else:
                        table.add_row(str(i), clean_option, "✓ Installed")
                elif "(huggingface)" in option:
                    clean_option = option.replace(" (huggingface)", "")
                    table.add_row(str(i), f"🤗 {clean_option}", "Hugging Face")
                else:
                    table.add_row(str(i), option, "Available")
            
            self.console.print(table)
            self.console.print()
            
            # Get user selection
            choice = Prompt.ask(
                "[bold cyan]Select model[/bold cyan]",
                choices=[str(i) for i in range(1, len(options) + 1)] + ['q'],
                default="1"
            )
            
            if choice == 'q':
                return None
            
            return options[int(choice) - 1]
            
        except Exception as e:
            print(f"Rich error: {e}")
            return None
    
    def select_with_simple(self, options: List[str], title: str) -> Optional[str]:
        """Fallback simple selection menu"""
        print(f"\n{title}")
        print("═" * 60)
        
        for i, option in enumerate(options, 1):
            if "(installed)" in option and "(huggingface)" in option:
                print(f"{i:2}. \033[32m🤗 {option}\033[0m")
            elif "(installed)" in option:
                print(f"{i:2}. \033[32m{option}\033[0m")
            elif "(huggingface)" in option:
                print(f"{i:2}. \033[33m🤗 {option}\033[0m")
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
            # Auto-select best available method
            available = self.get_available_methods()
            if "prompt-toolkit" in available:
                method = "prompt-toolkit"
            elif "inquirer" in available:
                method = "inquirer" 
            elif "rich" in available:
                method = "rich"
            elif "pyfzf" in available:
                method = "pyfzf"
            else:
                method = "simple"
        
        print(f"Using {method} selector...")
        
        if method == "prompt-toolkit":
            return self.select_with_prompt_toolkit(options, title)
        elif method == "inquirer":
            return self.select_with_inquirer(options, title)
        elif method == "pyfzf":
            return self.select_with_pyfzf(options, title)
        elif method == "rich":
            return self.select_with_rich(options, title)
        else:
            return self.select_with_simple(options, title)


# Hugging Face model utilities
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




def ensure_model_available(model_name: str) -> bool:
    """Ensure a model is available, pulling it if necessary (supports HF with tqdm)"""
    if is_model_installed(model_name):
        return True

    print(f"Model '{model_name}' not found locally. Pulling from repository...")

    try:
        if model_name.startswith('hf.co/'):
            # Trigger Ollama to download the HF model via `ollama run`
            print(f"→ Running: ollama run {model_name} (to auto-download from Hugging Face)")
            process = subprocess.Popen(
                ['ollama', 'run', model_name, 'test'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            with tqdm(total=100, desc="Downloading Hugging Face model", unit="%", leave=True) as bar:
                for line in process.stdout:
                    print(line.strip())
                    if "pulling" in line.lower() and "%" in line:
                        try:
                            percent = int(line.strip().split("%")[0].split()[-1])
                            bar.n = percent
                            bar.refresh()
                        except:
                            pass

            process.wait()
            if is_model_installed(model_name):
                print(f"Successfully pulled Hugging Face model: {model_name}")
                return True
            else:
                print(f"Failed to pull Hugging Face model: {model_name}")
                return False

        else:
            # Standard Ollama registry model
            subprocess.run(['ollama', 'pull', model_name], check=True)
            print(f"Successfully pulled model: {model_name}")
            return True

    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, KeyboardInterrupt) as e:
        print(f"Error pulling model '{model_name}': {e}")
        return False



# Installation helper
def install_menu_dependencies():
    """Helper to install menu dependencies"""
    packages = {
        "prompt-toolkit": "prompt-toolkit",
        "inquirer": "inquirer", 
        "pyfzf": "pyfzf",
        "rich": "rich"
    }
    
    print("Available enhanced menu options:")
    print("Install any of these for better dropdown experience:")
    print()
    
    for name, package in packages.items():
        status = "✓ Installed" if globals().get(f"HAS_{name.replace('-', '_').upper()}") else "Not installed"
        print(f"  {status} - {name}: pip install {package}")
    
    print()
    print("Recommendation: pip install prompt-toolkit (best dropdown experience)")


# Updated functions using the new selector with Hugging Face support
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
        "dolphin3", "dolphin-llama3", "dolphin-mixtral",
        "mixtral", "command-r", "command-r-plus",
        "granite3.3", "granite3.2",
        "smollm2", "smollm", "tinyllama",
        "codegemma", "codellama",  # still usable as chat models
        "neural-chat", "starcoder2", "starling-lm", "wizardlm2",
        "devstral", "llama3-chatqa", "codeqwen", "aya", "stablelm2"
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

def select_model_two_stage_inquirer() -> Optional[Tuple[str, str]]:
    """
    First show model family list, then show available variants in that family.
    Uses inquirer for both steps.
    """
    # Step 1: group models
    all_models, hf_models = get_available_ollama_models()
    
    families = {
        "llama": [],
        "qwen": [],
        "gemma": [],
        "deepseek": [],
        "mistral": [],
        "command-r": [],
        "huggingface": []  # explicitly for hf_models
    }

    # Fill family map
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

    # Step 2: pick family
    import inquirer
    family_question = [
        inquirer.List(
            "family",
            message="Select Supported Models",
            choices=[f for f in families if families[f]]
        )
    ]
    family_answer = inquirer.prompt(family_question)
    if not family_answer:
        return None

    selected_family = family_answer["family"]
    from ollash.ui import clear_screen
    clear_screen()
    family_models = families[selected_family]

    # Step 3: pick variant
    variant_question = [
        inquirer.List(
            "variant",
            message=f"Select Model from '{selected_family}'",
            choices=family_models
        )
    ]
    variant_answer = inquirer.prompt(variant_question)
    if not variant_answer:
        return None

    selected_variant = variant_answer["variant"]
    # Remove UI suffixes like " (installed)" or " (huggingface)"
    selected_variant = selected_variant.replace(" (installed)", "").replace(" (huggingface)", "").strip()


    # Final remap to HF path if needed
    for full_name, display in hf_models.items():
        if selected_variant.startswith(display):
            selected_variant = full_name
            break
        elif selected_variant == display:
            selected_variant = full_name
            break

    return ("ollama", selected_variant)


def select_model_advanced(backend: str = "ollama", method: str = "auto") -> Optional[str]:
    """Select model using advanced dropdown with Hugging Face support"""
    selector = MenuSelector()
    
    if method == "auto":
        available_methods = selector.get_available_methods()
        if len(available_methods) == 1 and available_methods[0] == "simple":
            print("For better dropdown experience, try:")
            install_menu_dependencies()
            print()
    
    if backend == "ollama":
        models, hf_models = get_available_ollama_models()
        title = "Select Ollama Model (including Hugging Face)"
    else:
        print(f"Backend {backend} not implemented yet")
        return None
    
    if not models:
        print(f"No models available for {backend}")
        return None
    
    selected = selector.select(models, title, method)
    
    if selected:
        # Map display name back to full Hugging Face path if needed
        for full_name, display_name in hf_models.items():
            if selected.startswith(display_name):
                selected = full_name
                break
        
        # Clean up the model name for regular models
        if " (huggingface)" not in selected:
            selected = selected.replace(" (installed)", "")
        else:
            # For HF models, remove display suffixes but keep the full path
            selected = selected.replace(" (huggingface)", "").replace(" (installed)", "")
            # Re-map to full path if it was shortened
            for full_name, display_name in hf_models.items():
                if selected == display_name:
                    selected = full_name
                    break
        
        print(f"Selected: {selected}")
        
        # Ensure the model is available (auto-pull if needed)
        if selected.startswith('hf.co/'):
            print("Checking Hugging Face model availability...")
            if not ensure_model_available(selected):
                print("Failed to make model available. Please try again.")
                return None
        
        return selected
    
    return None


# Integration function for your existing code
def get_model_selection_advanced(method: str = "auto") -> Optional[Tuple[str, str]]:
    """Enhanced model selection with dropdown alternatives and Hugging Face support"""
    try:
        # For now, assume ollama backend
        backend = "ollama"
        model = select_model_advanced(backend, method)
        
        if model:
            return backend, model
        return None
        
    except KeyboardInterrupt:
        print("\nSelection cancelled")
        return None


if __name__ == "__main__":
    # Test the enhanced selection
    print("Testing Enhanced Model Selection with Hugging Face Support")
    print("=" * 60)
    
    # Show available methods
    selector = MenuSelector()
    methods = selector.get_available_methods()
    print(f"Available selection methods: {', '.join(methods)}")
    print()
    
    # Show HF models
    hf_models = get_hf_models()
    print("Configured Hugging Face models:")
    for full_path, display_name in hf_models.items():
        status = "installed" if is_model_installed(full_path) else "available"
        print(f"  - {display_name} ({status})")
    print()
    
    # Test selection
    result = get_model_selection_advanced()
    if result:
        backend, model = result
        print(f"\nReady to use {model} with {backend}!")
    else:
        print("\nNo model selected")