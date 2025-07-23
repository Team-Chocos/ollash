# ollash/shell.py
import subprocess
import sys
import os
import time
import threading
from ollash.utils import ensure_ollama_ready, is_model_installed, pull_model_with_progress, get_os_label


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
    for i, (frame, color) in enumerate(zip(lightning_frames, colors)):
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


def get_command_suggestion(prompt: str, model: str) -> str:
    """Get command suggestion from Ollama without interactive prompts"""
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
        import re
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
    print_box_line("OLLASH SHELL", width=width)
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


def main(model=None):
    """Main REPL shell function"""
    model = model or "llama3"
    
    # Initial setup
    try:
        ensure_ollama_ready()
        if not is_model_installed(model):
            animation = ThinkingAnimation("Installing model")
            animation.start()
            pull_model_with_progress(model)
            animation.stop()
    except Exception as e:
        print_status(f"Setup failed: {e}", "error", in_box=False)
        return

    # Welcome screen
    clear_screen()
    print_banner(model)
    print()
    print_status("Ready! Type ':help' for commands", "success", in_box=False)
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
                continue

            # Get command suggestion
            try:
                animation = ThinkingAnimation("Generating command")
                animation.start()
                command = get_command_suggestion(user_input, model)
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
                                edited_command = input(f"│ Edit [{command}] ❯ ").strip()
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
        animation = ThinkingAnimation(f"Stopping model: {model}")
        animation.start()
        subprocess.run(["ollama", "stop", model], capture_output=True)
        animation.stop()
        print(f"│ Model stopped")
    except:
        pass
