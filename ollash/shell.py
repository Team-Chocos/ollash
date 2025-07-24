# ollash/shell.py
import subprocess
import os
import time
import threading
import re
from ollash.utils import ensure_ollama_ready, is_model_installed, pull_model_with_progress, get_os_label

from ollash.history import HistoryLogger
from ollash.menu_advanced import get_model_selection_advanced

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
        if self.running:
            self.running = False
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=1)  # Add timeout to prevent hanging
        # Clear the line
        print(f"\r{' ' * (len(self.message) + 20)}", end='\r', flush=True)

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
        # Fallback for systems without readline - just use regular input
        return input(prompt)


def animate_arrow():
    """Animate arrow appearance"""
    arrow_frames = ["", "─", "─>", "─→"]
    for frame in arrow_frames:
        print(f"\r│ \033[36m{frame}\033[0m", end='', flush=True)
        time.sleep(0.1)
    print()


def animate_lightning():
    """Animate lightning bolt"""
    lightning_frames = ["⚬", "⚡", "⚡"]
    colors = ["\033[33m", "\033[93m", "\033[33m"]  # yellow variations
    for _, (frame, color) in enumerate(zip(lightning_frames, colors)):
        print(f"\r│ {color}{frame}\033[0m", end='', flush=True)
        time.sleep(0.15)
    print()


def animate_success():
    """Animate success checkmark"""
    success_frames = ["○", "◐", "◑", "◒", "◓", "●", "✓"]
    for frame in success_frames:
        if frame == "✓":
            print(f"\r│ \033[32m{frame}\033[0m Command completed successfully", flush=True)
        else:
            print(f"\r│ \033[32m{frame}\033[0m", end='', flush=True)
        time.sleep(0.1)


def animate_failure():
    """Animate failure X mark"""
    failure_frames = ["○", "◑", "◒", "◓", "●", "✗"]
    for frame in failure_frames:
        if frame == "✗":
            print(f"\r│ \033[31m{frame}\033[0m Command failed", flush=True)
        else:
            print(f"\r│ \033[31m{frame}\033[0m", end='', flush=True)
        time.sleep(0.1)


def get_contextual_command_suggestion(prompt: str, model: str, history: HistoryLogger) -> tuple[str, str]:
    """Get command suggestion using semantic search context"""
    if not is_model_installed(model):
        pull_model_with_progress(model)

    os_label = get_os_label()
    
    # First, make a quick guess at what the command might be for better embedding
    potential_command = _quick_command_guess(prompt, os_label)
    
    # Search for similar past entries
    similar_entries = history.search_similar(prompt, potential_command, limit=3, model=model)
    
    # Build context from similar entries
    context = ""
    if similar_entries:
        context = "\n# Context from your past similar commands:\n"
        for i, (entry, similarity) in enumerate(similar_entries, 1):
            if similarity > 0.3:  # Only include reasonably similar entries
                context += f"{i}. When you asked: '{entry['input']}'\n"
                context += f"   I suggested: {entry['generated_command']}\n"
                context += f"   Result: {entry['execution_result']}\n"
                if entry.get('tags'):
                    context += f"   Tags: {entry['tags']}\n"
                context += "\n"
    
    # Enhanced prompt with context
    enhanced_prompt = f"""{context}
Current request: {prompt}

Based on the context above and the current request, translate this into a safe {os_label} terminal command.
Follow patterns from successful past commands when relevant.
Respond ONLY with the command, no explanation."""

    try:
        ollama_cmd = [
            "ollama", "run", model, enhanced_prompt
        ]
        
        response = subprocess.run(
            ollama_cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore"
        )

        raw_output = response.stdout.strip()
        
        # Extract command
        match = re.search(r"`([^`]+)`", raw_output)
        command = match.group(1).strip() if match else raw_output.strip().splitlines()[0]
        
        # Clean up common formatting issues
        command = command.replace("```", "").replace("bash", "").replace("sh", "").strip()
        
        return command, context
        
    except Exception as e:
        raise Exception(f"Failed to get command suggestion: {e}")


def _quick_command_guess(prompt: str, os_label: str) -> str:
    """Make a quick educated guess about the command for better embedding"""
    prompt_lower = prompt.lower()
    
    # Common patterns - this helps with embedding similarity
    if "list" in prompt_lower or "show" in prompt_lower:
        if "file" in prompt_lower or "directory" in prompt_lower:
            return "ls -la" if os_label != "Windows" else "dir"
    elif "create" in prompt_lower or "make" in prompt_lower:
        if "directory" in prompt_lower or "folder" in prompt_lower:
            return "mkdir"
        elif "file" in prompt_lower:
            return "touch" if os_label != "Windows" else "type nul >"
    elif "copy" in prompt_lower:
        return "cp" if os_label != "Windows" else "copy"
    elif "move" in prompt_lower:
        return "mv" if os_label != "Windows" else "move"
    elif "delete" in prompt_lower or "remove" in prompt_lower:
        return "rm" if os_label != "Windows" else "del"
    elif "find" in prompt_lower or "search" in prompt_lower:
        return "find" if os_label != "Windows" else "findstr"
    elif "install" in prompt_lower:
        return "apt install" if os_label == "Linux" else "brew install" if os_label == "macOS" else "choco install"
    
    return ""


def get_command_suggestion(prompt: str, model: str) -> str:
    """Original command suggestion function for backward compatibility"""
    if not is_model_installed(model):
        pull_model_with_progress(model)

    os_label = get_os_label()
    
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
        match = re.search(r"`([^`]+)`", raw_output)
        command = match.group(1).strip() if match else raw_output.strip().splitlines()[0]
        
        return command
    except Exception as e:
        raise Exception(f"Failed to get command suggestion: {e}")


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


def print_banner(model):
    """Print beautiful boxed banner"""
    width = 70
    print_box_line(style="top", width=width)
    print_box_line("", width=width)
    print_box_line("OLLASH SHELL - AI POWERED", width=width)
    print_box_line(f"Model: {model}", width=width)
    print_box_line("Semantic Search Enabled", width=width)
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
    print_box_line(":history [n]         Show recent history", width=width, style="left")
    print_box_line(":search <query>      Search command history", width=width, style="left")
    print_box_line(":exit, :quit         Exit shell", width=width, style="left")
    print_box_line("", width=width)
    print_box_line("Shortcuts:", width=width, style="left")
    print_box_line("Ctrl+C               Cancel operation", width=width, style="left")
    print_box_line("Ctrl+D               Exit shell", width=width, style="left")
    print_box_line(style="bottom", width=width)
    print()


def format_prompt(model):
    """Create a clean prompt with box styling"""
    return f"│ [{model}] ❯ "


def print_status(message, status_type="info", in_box=True):
    """Print formatted status messages"""
    symbols = {
        "info": "ℹ",
        "success": "✓",
        "error": "✗",
        "suggestion": "→",
        "executing": "⚡",
        "context": "",
        "embedding": "🧠"
    }
    symbol = symbols.get(status_type, "ℹ")
    
    if in_box:
        print_box_line(f"{symbol} {message}", width=70, style="left")
    else:
        print(f"{symbol} {message}")


def print_suggested_command(command, has_context=False):
    """Print suggested command with clean aesthetic"""
    print()
    context_indicator = " 🔍" if has_context else ""
    print(f"│ \033[36m→\033[0m \033[1m{command}\033[0m{context_indicator}")
    print("│")


def print_context_info(context: str):
    """Print context information if available"""
    if context and context.strip():
        print("│ \033[90m💡 Based on your past similar commands\033[0m")
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


def print_history_entries(entries, title="Recent History"):
    """Print history entries in a formatted way"""
    if not entries:
        print_status("No history entries found", "info", in_box=False)
        return
    
    print()
    print_box_line(title, width=70, style="middle")
    print_box_line(style="separator", width=70)
    
    for i, entry in enumerate(entries, 1):
        print_box_line(f"{i}. {entry['input'][:50]}{'...' if len(entry['input']) > 50 else ''}", width=70, style="left")
        print_box_line(f"   → {entry['generated_command']}", width=70, style="left")
        result_color = "✓" if entry['execution_result'] == "success" else "✗"
        print_box_line(f"   {result_color} {entry['execution_result']}", width=70, style="left")
        if i < len(entries):
            print_box_line("", width=70)
    
    print_box_line(style="bottom", width=70)
    print()

def main(model=None, backend=None):
    """Main REPL shell function with semantic search"""
    history = HistoryLogger(model)  # Pass model to history logger

    
    
    # Interactive model selection if no model specified
    if not model:
        print("🤖 No model specified, starting interactive selection...")
        selection = get_model_selection_advanced(method="pyfzf")
        
        if not selection:
            print("❌ No model selected. Exiting...")
            retur
        
        backend, model = selection
    else:
        # Default to ollama if backend not specified
        backend = backend or "ollama"
    
    print(f"\n🚀 Starting Ollash with {model} on {backend}")
    
    # Initial setup based on backend
    try:
        if backend == "ollama":
            ensure_ollama_ready()
            if not is_model_installed(model):
                animation = ThinkingAnimation("Installing model")
                animation.start()
                pull_model_with_progress(model)
                animation.stop()
        elif backend == "llama-cpp":
            # Setup llama.cpp if needed
            setup_llamacpp(model)
            
    except Exception as e:
        print_status(f"Setup failed: {e}", "error", in_box=False)
        return

    # Welcome screen
    clear_screen()
    print_banner(f"{model} ({backend})")
    print()
    print_status("Ready! AI shell with semantic search enabled", "success", in_box=False)
    print_status("Type ':help' for commands", "info", in_box=False)
    print()

    while True:
        try:
            # Get user input
            try:
                user_input = input(format_prompt(model)).strip()
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
                print_banner(model)
                print()
                continue
            
            elif user_input.startswith(":history"):
                parts = user_input.split(maxsplit=2)
                if len(parts) == 1:
                    # Just show history
                    limit = 10
                    entries = history.get_recent_entries(limit)
                    print_history_entries(entries, f"Last {limit} Commands")
                elif len(parts) == 2 and parts[1].isdigit():
                    # Show N recent entries
                    limit = int(parts[1])
                    entries = history.get_recent_entries(limit)
                    print_history_entries(entries, f"Last {limit} Commands")
                elif len(parts) >= 2:
                    # Use context for command generation
                    query = " ".join(parts[1:])
                    try:
                        animation = ThinkingAnimation("Analyzing with context")
                        animation.start()
                        command, context = get_contextual_command_suggestion(query, model, history)
                        animation.stop()
                        
                        has_context = bool(context and context.strip())
                        print_suggested_command(command, has_context)
                        
                        if has_context:
                            print_context_info(context)
                        
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
                                    # Log contextual commands when executed with 'y'
                                    history.log(query, command, "success" if success else "failure", 
                                              os.getcwd(), model=model, generate_embedding=True)
                                    print("│ \033[90m🧠 Learning from this command...\033[0m")
                                    break
                                elif choice in ['e', 'edit']:
                                    try:
                                        edited_command = input_with_prefill("│ Edit ❯ ", command).strip()
                                        if edited_command:
                                            command = edited_command
                                            print_execution_start(command)
                                            success = execute_command(command)
                                            print_execution_result(success)
                                            # Log edited contextual commands
                                            history.log(query, command, "success" if success else "failure", 
                                                      os.getcwd(), model=model, generate_embedding=True)
                                            print("│ \033[90m🧠 Learning from this command...\033[0m")
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
                continue
            
            elif user_input.startswith(":search "):
                query = user_input[8:].strip()
                if query:
                    similar_entries = history.search_similar(query, limit=5, model=model)
                    if similar_entries:
                        entries = [entry for entry, _ in similar_entries]
                        print_history_entries(entries, f"Search Results for '{query}'")
                    else:
                        print_status(f"No matches found for '{query}'", "info", in_box=False)
                else:
                    print_status("Please provide a search query", "error", in_box=False)
                continue
            
            elif user_input.startswith(":model "):
                new_model = user_input[7:].strip()
                if new_model:
                    if not is_model_installed(new_model):
                        try:
                            animation = ThinkingAnimation(f"Installing model '{new_model}'")
                            animation.start()
                            pull_model_with_progress(new_model)
                            animation.stop()
                            model = new_model
                            print_status(f"Switched to model: {model}", "success", in_box=False)
                        except Exception as e:
                            print_status(f"Failed to load model '{new_model}': {e}", "error", in_box=False)
                    else:
                        model = new_model
                        print_status(f"Switched to model: {model}", "success", in_box=False)
                else:
                    print_status("Please specify a model name", "error", in_box=False)
                continue
                
            elif user_input.startswith(":sh "):
                command = user_input[4:]
                print_execution_start(command)
                success = execute_command(command)
                print_execution_result(success)
                # Don't index :sh commands - they bypass the suggestion system
                continue

            # Get command suggestion (without context by default)
            try:
                animation = ThinkingAnimation("Generating command")
                animation.start()
                command = get_command_suggestion(user_input, model)
                animation.stop()
                
                print_suggested_command(command, False)
                
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
                            # Only log to history when user confirms execution with 'y'
                            history.log(user_input, command, "success" if success else "failure", 
                                      os.getcwd(), model=model, generate_embedding=True)
                            # Subtle indicator that embedding is being processed in background
                            print("│ \033[90m🧠 Learning from this command...\033[0m")
                            break
                        elif choice in ['e', 'edit']:
                            try:
                                edited_command = input_with_prefill("│ Edit ❯ ", command).strip()
                                if edited_command:
                                    command = edited_command
                                    print_execution_start(command)
                                    success = execute_command(command)
                                    print_execution_result(success)
                                    # Log the edited command when executed
                                    history.log(user_input, command, "success" if success else "failure", 
                                              os.getcwd(), model=model, generate_embedding=True)
                                    print("│ \033[90m🧠 Learning from this command...\033[0m")
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
        # Shutdown background embedding processing
        history.shutdown()
        
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
        description="Run Ollash REPL Shell"
    )
    parser.add_argument(
        "--model", type=str, help="Model name to use (e.g., llama3:8b)"
    )
    parser.add_argument(
        "--backend", type=str, choices=["ollama", "llama-cpp"], help="Backend to use"
    )

    args = parser.parse_args()

    # Run the shell with optional model/backend
    main(model=args.model, backend=args.backend)
