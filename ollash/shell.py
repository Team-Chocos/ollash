# ollash/shell.py
import subprocess
import sys
import os
import time
import threading
from ollash.utils import ensure_ollama_ready, is_model_installed, pull_model_with_progress, get_os_label
from ollash.menu_advanced import get_backend_and_model_selection, MenuSelector

# Try to import readline for better input editing
try:
    import readline
    HAS_READLINE = True
except ImportError:
    HAS_READLINE = False


class ThinkingAnimation:
    """Animated thinking indicator"""
    def __init__(self, message="Thinking"):
        self.message = message
        self.running = False
        self.thread = None
        self.frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.current_frame = 0

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._animate)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        # Clear the line
        print(f"\r{' ' * (len(self.message) + 10)}", end='\r')

    def _animate(self):
        while self.running:
            frame = self.frames[self.current_frame]
            print(f"\r{frame} {self.message}...", end='', flush=True)
            self.current_frame = (self.current_frame + 1) % len(self.frames)
            time.sleep(0.1)


def input_with_prefill(prompt, prefill=''):
    """Input function that prefills the input with given text"""
    if HAS_READLINE and prefill:
        def startup_hook():
            readline.insert_text(prefill)
        readline.set_startup_hook(startup_hook)
        try:
            return input(prompt)
        finally:
            readline.set_startup_hook(None)
    else:
        return input(prompt)


def get_command_suggestion(prompt: str, model: str, backend: str) -> str:
    """Get command suggestion based on backend"""
    os_label = get_os_label()
    
    if backend == "ollama":
        if not is_model_installed(model):
            pull_model_with_progress(model)

        ollama_cmd = [
            "ollama", "run", model,
            f"Translate the following instruction into a safe {os_label} terminal command. Respond ONLY with the command, no explanation:\nInstruction: {prompt}"
        ]

        try:
            response = subprocess.run(
                ollama_cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore"
            )
            raw_output = response.stdout.strip()
            
            # Extract command
            import re
            match = re.search(r"`([^`]+)`", raw_output)
            command = match.group(1).strip() if match else raw_output.strip().splitlines()[0]
            return command
            
        except Exception as e:
            raise Exception(f"Failed to get command suggestion: {e}")
    
    elif backend == "llama_cpp":
        # Implement llama.cpp command generation
        return get_llamacpp_command_suggestion(prompt, model)
    
    elif backend == "gguf":
        # Implement GGUF model command generation  
        return get_gguf_command_suggestion(prompt, model)
    
    else:
        raise Exception(f"Unsupported backend: {backend}")


def get_llamacpp_command_suggestion(prompt: str, model: str) -> str:
    """Get command suggestion using llama.cpp"""
    # This is a placeholder - implement based on your llama.cpp setup
    try:
        # Example llama.cpp command structure
        llama_cmd = [
            "llama-cli",
            "-m", model,
            "-p", f"Translate this to a terminal command: {prompt}",
            "--simple-io"
        ]
        
        response = subprocess.run(
            llama_cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if response.returncode == 0:
            return response.stdout.strip().splitlines()[0]
        else:
            raise Exception(f"llama.cpp error: {response.stderr}")
            
    except Exception as e:
        raise Exception(f"Failed to get llama.cpp suggestion: {e}")


def get_gguf_command_suggestion(prompt: str, model: str) -> str:
    """Get command suggestion using GGUF model"""
    # This is a placeholder - implement based on your GGUF setup
    try:
        # You might use llama-cpp-python or another GGUF runner here
        # This is just an example structure
        gguf_cmd = [
            "python", "-c", f"""
import llama_cpp
llm = llama_cpp.Llama(model_path='{model}')
result = llm('Translate to terminal command: {prompt}', max_tokens=50)
print(result['choices'][0]['text'].strip())
"""
        ]
        
        response = subprocess.run(
            gguf_cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if response.returncode == 0:
            return response.stdout.strip().splitlines()[0]
        else:
            raise Exception(f"GGUF error: {response.stderr}")
            
    except Exception as e:
        raise Exception(f"Failed to get GGUF suggestion: {e}")


def setup_backend(backend: str, model: str):
    """Setup the specified backend"""
    if backend == "ollama":
        ensure_ollama_ready()
        if not is_model_installed(model):
            animation = ThinkingAnimation(f"Installing Ollama model '{model}'")
            animation.start()
            pull_model_with_progress(model)
            animation.stop()
            
    elif backend == "llama_cpp":
        # Setup llama.cpp
        print(f"🔧 Setting up llama.cpp with model: {model}")
        # Add your llama.cpp setup logic here
        
    elif backend == "gguf":
        # Setup GGUF model
        print(f"🔧 Setting up GGUF model: {model}")
        if not os.path.exists(model):
            raise Exception(f"GGUF model file not found: {model}")
        # Add your GGUF setup logic here
        
    else:
        raise Exception(f"Unsupported backend: {backend}")


def execute_command(command: str) -> bool:
    """Execute a command and return True if successful"""
    try:
        result = subprocess.run(command, shell=True)
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\n│ [Interrupted]")
        return False
    except Exception as e:
        print(f"│ Error: {e}")
        return False


def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_box_line(content="", width=70, style="middle"):
    """Print a line with box styling"""
    if style == "top":
        print(f"┌{'─' * (width-2)}┐")
    elif style == "bottom":
        print(f"└{'─' * (width-2)}┘")
    elif style == "separator":
        print(f"├{'─' * (width-2)}┤")
    elif style == "middle":
        padding = width - len(content) - 4
        left_pad = padding // 2
        right_pad = padding - left_pad
        print(f"│ {' ' * left_pad}{content}{' ' * right_pad} │")
    elif style == "left":
        padding = width - len(content) - 4
        print(f"│ {content}{' ' * padding} │")


def print_banner(model, backend):
    """Print beautiful boxed banner"""
    width = 70
    print_box_line(style="top", width=width)
    print_box_line("", width=width)
    print_box_line("OLLASH SHELL", width=width)
    print_box_line(f"Backend: {backend.upper()}", width=width)
    print_box_line(f"Model: {model}", width=width)
    print_box_line("", width=width)
    print_box_line(style="bottom", width=width)


def print_help():
    """Print help information in a box"""
    width = 70
    print()
    print_box_line(style="top", width=width)
    print_box_line("COMMANDS", width=width)
    print_box_line(style="separator", width=width)
    print_box_line("<natural language>    Get command suggestion", width=width, style="left")
    print_box_line("", width=width)
    print_box_line("Special Commands:", width=width, style="left")
    print_box_line(":help                Show this help", width=width, style="left")
    print_box_line(":clear               Clear screen", width=width, style="left")
    print_box_line(":model <name>        Switch model", width=width, style="left")
    print_box_line(":backend <name>      Switch backend", width=width, style="left")
    print_box_line(":exit, :quit         Exit shell", width=width, style="left")
    print_box_line("", width=width)
    print_box_line("Shortcuts:", width=width, style="left")
    print_box_line("Ctrl+C               Cancel operation", width=width, style="left")
    print_box_line("Ctrl+D               Exit shell", width=width, style="left")
    print_box_line(style="bottom", width=width)
    print()


def format_prompt(model, backend):
    """Create a clean prompt with box styling"""
    return f"│ [{backend}:{model}] ❯ "


def print_status(message, status_type="info", in_box=True):
    """Print formatted status messages"""
    symbols = {
        "info": "ℹ",
        "success": "✓",
        "error": "✗",
        "suggestion": "→",
        "executing": "⚡"
    }
    symbol = symbols.get(status_type, "ℹ")
    
    if in_box:
        print_box_line(f"{symbol} {message}", width=70, style="left")
    else:
        print(f"{symbol} {message}")


def print_suggested_command(command):
    """Print suggested command with clean aesthetic"""
    print()
    print(f"│ \033[36m→\033[0m \033[1m{command}\033[0m")
    print("│")


def print_execution_start(command):
    """Print execution start with clean style"""
    print()
    print(f"│ \033[33m⚡\033[0m Executing: \033[90m{command}\033[0m")
    print("│ " + "─" * 50)


def print_execution_result(success):
    """Print execution result with clean styling"""
    if success:
        print("│ " + "─" * 50)
        print(f"│ \033[32m✓\033[0m Command completed successfully")
    else:
        print("│ " + "─" * 50)
        print(f"│ \033[31m✗\033[0m Command failed")


def main(model=None, backend=None):
    """Main REPL shell function with backend/model workflow"""
    
    # Workflow: Check if both backend and model are provided via CLI
    if backend and model:
        print(f"🚀 Loading {backend} with model {model}...")
    else:
        # Interactive selection if not provided
        print("🤖 Starting interactive backend and model selection...")
        
        try:
            selection = get_backend_and_model_selection()
            if not selection:
                print("❌ No backend/model selected. Exiting...")
                return
            
            backend, model = selection
            
        except Exception as e:
            print(f"❌ Selection failed: {e}")
            print("💡 Make sure pyfzf is installed: pip install pyfzf")
            return
    
    print(f"\n🚀 Starting Ollash with {model} on {backend}")
    
    # Setup the selected backend
    try:
        setup_backend(backend, model)
    except Exception as e:
        print_status(f"Backend setup failed: {e}", "error", in_box=False)
        return

    # Welcome screen
    clear_screen()
    print_banner(model, backend)
    print()
    print_status("Ready! Type ':help' for commands", "success", in_box=False)
    print()

    while True:
        try:
            # Get user input
            try:
                user_input = input(format_prompt(model, backend)).strip()
            except EOFError:
                print("\n│ Goodbye!")
                break
            except KeyboardInterrupt:
                print()
                continue

            if not user_input:
                continue

            # Handle special commands
            if user_input in [":exit", ":quit"]:
                print("│ Goodbye!")
                break
            
            elif user_input == ":help":
                print_help()
                continue
            
            elif user_input == ":clear":
                clear_screen()
                print_banner(model, backend)
                print()
                continue
            
            elif user_input.startswith(":model "):
                new_model = user_input[7:].strip()
                if new_model:
                    try:
                        setup_backend(backend, new_model)
                        model = new_model
                        print_status(f"Switched to model: {model}", "success", in_box=False)
                    except Exception as e:
                        print_status(f"Failed to switch to model '{new_model}': {e}", "error", in_box=False)
                else:
                    print_status("Please specify a model name", "error", in_box=False)
                continue
            
            elif user_input.startswith(":backend "):
                new_backend = user_input[9:].strip()
                if new_backend in ["ollama", "llama_cpp", "gguf"]:
                    try:
                        # Interactive model selection for new backend
                        selector = MenuSelector()
                        new_model = selector.select_model(new_backend)
                        if new_model:
                            setup_backend(new_backend, new_model)
                            backend = new_backend
                            model = new_model
                            print_status(f"Switched to {backend} with model: {model}", "success", in_box=False)
                        else:
                            print_status("No model selected for new backend", "error", in_box=False)
                    except Exception as e:
                        print_status(f"Failed to switch backend: {e}", "error", in_box=False)
                else:
                    print_status("Valid backends: ollama, llama_cpp, gguf", "error", in_box=False)
                continue
                
            elif user_input.startswith(":sh "):
                command = user_input[4:]
                print_execution_start(command)
                success = execute_command(command)
                print_execution_result(success)
                continue

            # Get command suggestion
            try:
                animation = ThinkingAnimation("Generating command")
                animation.start()
                command = get_command_suggestion(user_input, model, backend)
                animation.stop()
                
                print_suggested_command(command)
                
                # Ask if user wants to run it
                while True:
                    try:
                        choice = input("│ Execute? [y/N/e(dit)] ❯ ").strip().lower()
                        if choice in ['', 'n', 'no']:
                            break
                        elif choice in ['y', 'yes']:
                            print_execution_start(command)
                            success = execute_command(command)
                            print_execution_result(success)
                            break
                        elif choice in ['e', 'edit']:
                            try:
                                edited_command = input_with_prefill("│ Edit ❯ ", command).strip()
                                if edited_command:
                                    command = edited_command
                                    print_execution_start(command)
                                    success = execute_command(command)
                                    print_execution_result(success)
                                break
                            except (EOFError, KeyboardInterrupt):
                                print("\n│ Cancelled")
                                break
                        else:
                            print("│ Enter 'y' (yes), 'n' (no), or 'e' (edit)")
                    except (EOFError, KeyboardInterrupt):
                        print("\n│ Cancelled")
                        break
                        
            except Exception as e:
                print_status(f"Error: {e}", "error", in_box=False)
                
            print()  # Add spacing between operations
                
        except KeyboardInterrupt:
            print()
            continue
        except Exception as e:
            print_status(f"Unexpected error: {e}", "error", in_box=False)
            continue

    # Cleanup
    try:
        if backend == "ollama":
            animation = ThinkingAnimation(f"Stopping model: {model}")
            animation.start()
            subprocess.run(["ollama", "stop", model], capture_output=True)
            animation.stop()
            print(f"│ Model stopped")
    except:
        pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Ollash REPL Shell",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ollash shell                          # Interactive backend and model selection
  ollash shell --backend ollama --model llama3:8b
  ollash shell --backend llama_cpp --model llama-3.2-3b-instruct.gguf
  ollash shell --backend gguf --model ~/models/phi-3-mini.gguf
        """
    )
    parser.add_argument(
        "--model", 
        type=str, 
        help="Model name to use (specific to backend)"
    )
    parser.add_argument(
        "--backend", 
        type=str, 
        choices=["ollama", "llama_cpp", "gguf"], 
        help="Backend engine to use"
    )

    args = parser.parse_args()

    # Validate arguments
    if (args.backend and not args.model) or (args.model and not args.backend):
        parser.error("Both --backend and --model must be specified together, or neither")

    # Run the shell with workflow logic
    main(model=args.model, backend=args.backend)