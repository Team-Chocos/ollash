# ollash/menu_advanced.py
import subprocess
from typing import List, Tuple, Optional, Dict
import os
import glob

# Only use pyfzf for UI
HAS_PYFZF = False
try:
    from pyfzf.pyfzf import FzfPrompt
    HAS_PYFZF = True
except ImportError:
    pass


class BackendManager:
    """Manages different backend configurations and model lists"""
    
    def __init__(self):
        self.backends = {
            "ollama": {
                "name": "[OLLAMA]",
                "description": "Local Ollama server (easiest setup)",
                "status": self._check_ollama_status()
            },
            "llama_cpp": {
                "name": "[LLAMA.CPP]", 
                "description": "Direct llama.cpp integration (fastest)",
                "status": self._check_llamacpp_status()
            },
            "gguf": {
                "name": "[GGUF]",
                "description": "Local GGUF model files",
                "status": self._check_gguf_status()
            }
        }
    
    def _check_ollama_status(self) -> str:
        """Check if Ollama is available"""
        try:
            result = subprocess.run(
                ["ollama", "--version"], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            return "READY" if result.returncode == 0 else "NOT INSTALLED"
        except:
            return "NOT INSTALLED"
    
    def _check_llamacpp_status(self) -> str:
        """Check if llama.cpp is available"""
        # Check for llama-cpp-python first (easier to install)
        try:
            import llama_cpp
            return "READY"
        except ImportError:
            pass
        
        # Check for compiled llama.cpp executables
        executables = ["llama-cli", "main"]
        for exe in executables:
            try:
                result = subprocess.run(
                    ["which", exe] if os.name != 'nt' else ["where", exe],
                    capture_output=True,
                    timeout=3
                )
                if result.returncode == 0:
                    return "READY"
            except:
                continue
        return "WILL INSTALL"
    
    def _check_gguf_status(self) -> str:
        """Check if GGUF directory exists"""
        gguf_dirs = [
            os.path.expanduser("~/models"),
            os.path.expanduser("~/.cache/huggingface"),
            "./models",
            "/opt/models"
        ]
        
        for dir_path in gguf_dirs:
            if os.path.exists(dir_path):
                gguf_files = glob.glob(os.path.join(dir_path, "**/*.gguf"), recursive=True)
                if gguf_files:
                    return f"READY ({len(gguf_files)} models)"
        
        return "NO MODELS FOUND"
    
    def get_backend_options(self) -> List[str]:
        """Get formatted backend options for selection"""
        options = []
        for key, backend in self.backends.items():
            status_info = f" [{backend['status']}]"
            option = f"{backend['name']} - {backend['description']}{status_info}"
            options.append(option)
        return options
    
    def parse_backend_selection(self, selection: str) -> str:
        """Parse backend selection back to key"""
        if "[OLLAMA]" in selection:
            return "ollama"
        elif "[LLAMA.CPP]" in selection:
            return "llama_cpp"
        elif "[GGUF]" in selection:
            return "gguf"
        return "ollama"  # fallback


class ModelManager:
    """Manages model lists for different backends"""
    
    def __init__(self):
        self.backend_manager = BackendManager()
    
    def install_llamacpp(self) -> bool:
        """Install llama-cpp-python if not available"""
        print("Installing llama.cpp (llama-cpp-python)...")
        try:
            import subprocess
            import sys
            
            # Install llama-cpp-python
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "llama-cpp-python"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✓ llama.cpp installed successfully")
                return True
            else:
                print(f"× Installation failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"× Installation error: {e}")
            return False
    
    def get_ollama_models(self) -> List[str]:
        """Get available Ollama models"""
        popular_models = [
            "llama3.2:3b", "llama3.2:1b", "llama3.1:8b", "llama3.1:70b",
            "llama3:8b", "llama3:70b", "mistral:7b", "mistral:latest",
            "codellama:7b", "codellama:13b", "phi3:mini", "phi3:medium",
            "gemma2:2b", "gemma2:9b", "qwen2.5:7b", "qwen2.5:14b",
            "deepseek-coder:6.7b", "codegemma:7b", "nomic-embed-text"
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
                        installed_models.append(f"● {model_name} (installed)")
            
            all_models = installed_models.copy()
            for model in popular_models:
                if not any(model in installed for installed in installed_models):
                    all_models.append(f"○ {model} (available)")
                    
            return all_models if all_models else [f"○ {m} (available)" for m in popular_models]
            
        except:
            return [f"○ {m} (available)" for m in popular_models]
    
    def get_llamacpp_models(self) -> List[str]:
        """Get available llama.cpp models"""
        models = [
            "○ llama-3.2-3b-instruct.gguf",
            "○ llama-3.2-1b-instruct.gguf", 
            "○ llama-3.1-8b-instruct.gguf",
            "○ mistral-7b-instruct.gguf",
            "○ codellama-7b-instruct.gguf",
            "○ phi-3-mini-4k-instruct.gguf",
            "○ gemma-2-2b-it.gguf",
            "○ qwen2.5-7b-instruct.gguf"
        ]
        
        # Check for locally available models
        model_dirs = [
            os.path.expanduser("~/models"),
            "./models",
            "/opt/models"
        ]
        
        local_models = []
        for model_dir in model_dirs:
            if os.path.exists(model_dir):
                gguf_files = glob.glob(os.path.join(model_dir, "*.gguf"))
                for gguf_file in gguf_files:
                    filename = os.path.basename(gguf_file)
                    local_models.append(f"● {filename} (local)")
        
        return local_models + models
    
    def get_gguf_models(self) -> List[str]:
        """Get available GGUF models from local directories"""
        model_dirs = [
            os.path.expanduser("~/models"),
            os.path.expanduser("~/.cache/huggingface"),
            "./models",
            "/opt/models"
        ]
        
        found_models = []
        
        for model_dir in model_dirs:
            if os.path.exists(model_dir):
                gguf_files = glob.glob(os.path.join(model_dir, "**/*.gguf"), recursive=True)
                for gguf_file in gguf_files:
                    rel_path = os.path.relpath(gguf_file, model_dir)
                    found_models.append(f"● {rel_path}")
        
        if not found_models:
            found_models = [
                "× No GGUF models found",
                "→ Place .gguf files in ~/models/ directory",
                "→ Or specify custom path with --model-path"
            ]
        
        return found_models
    
    def get_models_for_backend(self, backend: str) -> List[str]:
        """Get models for specified backend"""
        if backend == "ollama":
            return self.get_ollama_models()
        elif backend == "llama_cpp":
            return self.get_llamacpp_models()
        elif backend == "gguf":
            return self.get_gguf_models()
        else:
            return []
    
    def clean_model_name(self, model_selection: str) -> str:
        """Clean model name from selection"""
        # Remove status indicators and extra info
        model = model_selection
        for prefix in ["● ", "○ ", "× ", "→ "]:
            model = model.replace(prefix, "")
        
        # Remove status suffixes
        for suffix in [" (installed)", " (available)", " (local)"]:
            model = model.replace(suffix, "")
        
        return model.strip()


class MenuSelector:
    """Simple pyfzf-based menu selector"""
    
    def __init__(self):
        if not HAS_PYFZF:
            raise ImportError("pyfzf is required. Install with: pip install pyfzf")
        self.fzf = FzfPrompt()
    
    def select_backend(self) -> Optional[str]:
        """Select backend using fzf"""
        backend_manager = BackendManager()
        options = backend_manager.get_backend_options()
        
        try:
            selected = self.fzf.prompt(
                options,
                '--prompt="► Select Backend: " --height=40% --reverse --border --info=inline --header="Choose your preferred backend engine"'
            )
            
            if selected:
                backend = backend_manager.parse_backend_selection(selected[0])
                
                # Check if llama.cpp needs installation
                if backend == "llama_cpp":
                    status = backend_manager.backends["llama_cpp"]["status"]
                    if status == "WILL INSTALL":
                        print("\nllama.cpp not found. Installing...")
                        model_manager = ModelManager()
                        if not model_manager.install_llamacpp():
                            print("Failed to install llama.cpp. Please install manually:")
                            print("pip install llama-cpp-python")
                            return None
                        print("✓ llama.cpp ready")
                
                return backend
            return None
            
        except Exception as e:
            print(f"Backend selection failed: {e}")
            return None
    
    def select_model(self, backend: str) -> Optional[str]:
        """Select model for given backend using fzf"""
        model_manager = ModelManager()
        models = model_manager.get_models_for_backend(backend)
        
        if not models:
            print(f"No models available for backend: {backend}")
            return None
        
        backend_names = {
            "ollama": "Ollama",
            "llama_cpp": "Llama.cpp", 
            "gguf": "GGUF"
        }
        
        try:
            selected = self.fzf.prompt(
                models,
                f'--prompt="► Select {backend_names.get(backend, backend)} Model: " --height=60% --reverse --border --info=inline --header="Choose your model"'
            )
            
            if selected:
                return model_manager.clean_model_name(selected[0])
            return None
            
        except Exception as e:
            print(f"Model selection failed: {e}")
            return None


def get_backend_and_model_selection() -> Optional[Tuple[str, str]]:
    """Interactive backend and model selection"""
    try:
        if not HAS_PYFZF:
            print("× pyfzf is required for interactive selection")
            print("→ Install with: pip install pyfzf")
            return None
        
        selector = MenuSelector()
        
        # Step 1: Select backend
        print("► Step 1: Select Backend Engine")
        backend = selector.select_backend()
        
        if not backend:
            print("× No backend selected")
            return None
        
        print(f"✓ Selected backend: {backend}")
        
        # Step 2: Select model for chosen backend
        print(f"► Step 2: Select Model for {backend}")
        model = selector.select_model(backend)
        
        if not model:
            print("× No model selected")
            return None
        
        print(f"✓ Selected model: {model}")
        
        return backend, model
        
    except KeyboardInterrupt:
        print("\nSelection cancelled")
        return None
    except Exception as e:
        print(f"× Selection failed: {e}")
        return None


# Legacy compatibility functions
def get_model_selection_advanced(method: str = "pyfzf") -> Optional[Tuple[str, str]]:
    """Legacy function for backwards compatibility"""
    return get_backend_and_model_selection()


def select_model_advanced(backend: str = "ollama", method: str = "pyfzf") -> Optional[str]:
    """Legacy function for backwards compatibility"""
    if not HAS_PYFZF:
        print("× pyfzf is required")
        return None
    
    selector = MenuSelector()
    return selector.select_model(backend)


if __name__ == "__main__":
    # Test the selection system
    print("Testing Backend and Model Selection")
    print("=" * 50)
    
    result = get_backend_and_model_selection()
    if result:
        backend, model = result
        print(f"\n✓ Ready to use {model} with {backend}!")
    else:
        print("\n× No selection made")