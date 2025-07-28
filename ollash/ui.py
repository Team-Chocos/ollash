# ollash/ui.py
import os
import time
import threading
import subprocess
import shutil


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
                self.thread.join(timeout=1)
        # Clear the line
        print(f"\r{' ' * (len(self.message) + 20)}", end='\r', flush=True)

    def _animate(self):
        while self.running:
            frame = self.frames[self.current_frame]
            print(f"\r{frame} {self.message}...", end='', flush=True)
            self.current_frame = (self.current_frame + 1) % len(self.frames)
            time.sleep(0.1)


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


def _has_figlet():
    """Check if figlet is available on the system"""
    return shutil.which('figlet') is not None


def _generate_figlet_text(text, font='slant'):
    """Generate figlet text with fallback"""
    if not _has_figlet():
        return None
    
    # Try different aesthetic fonts in order of preference
    fonts = ['slant', 'lean', 'small', 'mini']
    
    for font_name in fonts:
        try:
            result = subprocess.run(
                ['figlet', '-f', font_name, text],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
            continue
    
    # Final fallback without font specification
    try:
        result = subprocess.run(
            ['figlet', text],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
        pass
    
    return None


def print_banner(model):
    """Print beautiful banner with figlet if available"""
    # Try to generate figlet text
    figlet_text = _generate_figlet_text("OLLASH")
    
    if figlet_text:
        # Print figlet banner with color and proper formatting
        print()
        lines = figlet_text.split('\n')
        # Filter out empty lines
        lines = [line for line in lines if line.strip()]
        
        # Calculate width - ensure minimum width for model info
        content_width = max(len(line) for line in lines)
        model_text = f"Powered by Ollama • Model: {model}"
        min_width = len(model_text) + 4
        box_width = max(content_width + 6, min_width)
        
        # Top border
        print(f"┌{'─' * (box_width-2)}┐")
        
        # Empty line
        print(f"│{' ' * (box_width-2)}│")
        
        # Figlet text with cyan color
        for line in lines:
            padding = box_width - len(line) - 4
            left_pad = padding // 2
            right_pad = padding - left_pad
            print(f"│ {' ' * left_pad}\033[36m{line}\033[0m{' ' * right_pad} │")
        
        # Empty line
        print(f"│{' ' * (box_width-2)}│")
        
        # Separator
        print(f"├{'─' * (box_width-2)}┤")
        
        # Model info with subtle color
        padding = box_width - len(model_text) - 4
        left_pad = padding // 2
        right_pad = padding - left_pad
        print(f"│ {' ' * left_pad}\033[90m{model_text}\033[0m{' ' * right_pad} │")
        
        # Bottom border
        print(f"└{'─' * (box_width-2)}┘")
    else:
        # Fallback to original box design with color
        width = 70
        print_box_line(style="top", width=width)
        print_box_line("", width=width)
        print_box_line("OLLASH SHELL - Powered by Ollama", width=width)
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
    print_box_line(":sh <command>        Run shell command directly", width=width, style="left")
    print_box_line(":history [n]         Show recent history", width=width, style="left")
    print_box_line(":search <query>      Search command history", width=width, style="left")
    print_box_line(":exit, :quit         Exit shell", width=width, style="left")
    print_box_line("", width=width)
    print_box_line("Shortcuts:", width=width, style="left")
    print_box_line("Ctrl+D               Exit shell", width=width, style="left")
    print_box_line(style="bottom", width=width)
    print()


def format_prompt(model):
    """Create a clean prompt with box styling"""
    return f"│ [{model}] ❯ "


def print_status(message, status_type="info", in_box=True):
    """Print formatted status messages"""
    symbols = {
        "info": "",
        "success": "✓",
        "error": "✗",
        "suggestion": "→",
        "executing": "",
        "context": "",
        "embedding": ""
    }
    symbol = symbols.get(status_type, "")
    
    if in_box:
        print_box_line(f"{symbol} {message}", width=70, style="left")
    else:
        print(f"{symbol} {message}")


def print_suggested_command(command, has_context=False):
    """Print suggested command with clean aesthetic"""
    print()
    context_indicator = " " if has_context else ""
    print(f"│ \033[36m→\033[0m \033[1m{command}\033[0m{context_indicator}")
    print("│")


def print_context_info(context: str):
    """Print context information if available"""
    if context and context.strip():
        print("│ \033[90m Based on your past similar commands\033[0m")
        print("│")


def print_execution_start(command):
    """Print execution start with clean style"""
    print()
    print(f"│ \033[33m\033[0m Executing: \033[90m{command}\033[0m")
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
