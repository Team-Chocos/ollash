# ollash/shell.py
import subprocess
from textual.app import App, ComposeResult
from textual.widgets import Static, Input
from textual.containers import Vertical
from textual.reactive import reactive

from ollash.utils import ensure_ollama_ready, is_model_installed, pull_model_with_progress
from ollash.ollama_nl2bash import run_nl_to_bash


class TerminalBox(Static): pass
class InputBox(Input): pass


class OllashShell(App):
    CSS = """
    Screen {
        layout: vertical;
        padding: 1;
    }
    #terminal {
        height: 90%;
        border: round green;
        overflow: auto;
    }
    #input {
        border: round yellow;
    }
    """

    def __init__(self, model="llama3", **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.last_command = None  # Store the last suggested command

    def compose(self) -> ComposeResult:
        self.terminal = TerminalBox("", id="terminal")
        yield Vertical(
            self.terminal,
            InputBox(placeholder="Enter natural language command...", id="input"),
        )

    def on_mount(self):
        # Ensure Ollama is installed and running
        ensure_ollama_ready()
        
        # Ensure the specific model is available
        if not is_model_installed(self.model):
            pull_model_with_progress(self.model)
            
        # Set initial content with instructions
        initial_message = (
            f"🧠 Loaded model: {self.model}\n"
            f"💡 Commands:\n"
            f"  • Type natural language to get command suggestions\n"
            f"  • Type ':run' to execute the last suggested command\n"
            f"  • Type ':exit' to quit\n"
        )
        self.terminal.update(initial_message)

    def on_input_submitted(self, message: Input.Submitted) -> None:
        query = message.value.strip()
        current_output = self.terminal.renderable
        
        if query in (":exit", "exit", "quit"):
            self.exit_shell()
            return
        
        # Handle :run command
        if query == ":run":
            if self.last_command:
                self.terminal.update(current_output + f"\n🚀 Executing: {self.last_command}\n")
                self.execute_command(self.last_command)
            else:
                self.terminal.update(current_output + "\n❌ No command to run. Generate a command first.\n")
            self.query_one(InputBox).value = ""
            return

        self.query_one(InputBox).value = ""
        
        # Show that we're processing
        self.terminal.update(current_output + f"\n💬 You: {query}\n🔄 Processing...")
        
        try:
            # Get the command without interactive prompts
            command = self.get_command_non_interactive(query)
            self.last_command = command  # Store for :run
            
            # Update with the suggested command
            new_output = current_output + f"\n💬 You: {query}\n🖥️  Suggested: {command}\n💡 Type ':run' to execute\n"
            self.terminal.update(new_output)
            
        except Exception as e:
            new_output = current_output + f"\n💬 You: {query}\n❌ Error: {str(e)}\n"
            self.terminal.update(new_output)
    
    def execute_command(self, command: str):
        """Execute a shell command and display the output"""
        try:
            result = subprocess.run(
                command, 
                shell=True, 
                capture_output=True, 
                text=True,
                encoding="utf-8",
                errors="ignore"
            )
            
            current_output = self.terminal.renderable
            
            if result.stdout:
                self.terminal.update(current_output + f"📤 Output:\n{result.stdout}\n")
            
            if result.stderr:
                self.terminal.update(self.terminal.renderable + f"⚠️  Error:\n{result.stderr}\n")
                
            if result.returncode != 0:
                self.terminal.update(self.terminal.renderable + f"❌ Command failed with exit code: {result.returncode}\n")
            else:
                self.terminal.update(self.terminal.renderable + f"✅ Command completed successfully\n")
                
        except Exception as e:
            current_output = self.terminal.renderable
            self.terminal.update(current_output + f"❌ Failed to execute: {str(e)}\n")
    
    def get_command_non_interactive(self, prompt: str) -> str:
        """Get command suggestion without interactive prompts"""
        if not is_model_installed(self.model):
            pull_model_with_progress(self.model)

        from ollash.utils import get_os_label
        os_label = get_os_label()
        
        ollama_cmd = [
            "ollama", "run", self.model,
            f"Translate the following instruction into a safe {os_label} terminal command. Respond ONLY with the command, no explanation:\nInstruction: {prompt}"
        ]

        response = subprocess.run(
            ollama_cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore"
        )

        raw_output = response.stdout.strip()
        
        # Extract command (reusing the logic from your original function)
        import re
        match = re.search(r"`([^`]+)`", raw_output)
        command = match.group(1).strip() if match else raw_output.strip().splitlines()[0]
        
        return command

    def exit_shell(self):
        new_output = self.terminal.renderable + f"\n👋 Stopping model: {self.model}...\n"
        self.terminal.update(new_output)
        subprocess.run(["ollama", "stop", self.model])
        self.exit()


def main(model=None):
    # Use the passed model or default to llama3
    model = model or "llama3"
    OllashShell(model=model).run()