# ollash/shell.py
import subprocess
import sys
from ollash.utils import ensure_ollama_ready, is_model_installed, pull_model_with_progress, get_os_label


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
        print("\n^C")
        return False
    except Exception as e:
        print(f"Error executing command: {e}")
        return False


def print_help():
    """Print help information"""
    print("""
Available commands:
  <natural language>  - Get command suggestion and choose to run it
  :help              - Show this help
  :exit, :quit       - Exit the shell
  :model <name>      - Switch to a different model
  Ctrl+C             - Cancel current operation
  Ctrl+D             - Exit the shell
""")


def main(model=None):
    """Main REPL shell function"""
    model = model or "llama3"
    
    print(f"🧠 Ollash Shell - Model: {model}")
    print("Type ':help' for commands or Ctrl+D to exit")
    
    # Ensure Ollama is ready
    try:
        ensure_ollama_ready()
        if not is_model_installed(model):
            pull_model_with_progress(model)
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return

    print("✅ Ready!\n")

    while True:
        try:
            # Get user input
            try:
                user_input = input(f"ollash({model})> ").strip()
            except EOFError:
                print("\n👋 Goodbye!")
                break
            except KeyboardInterrupt:
                print("\n^C")
                continue

            if not user_input:
                continue

            # Handle special commands
            if user_input in [":exit", ":quit"]:
                print("👋 Goodbye!")
                break
            
            elif user_input == ":help":
                print_help()
                continue
            
            elif user_input.startswith(":model "):
                new_model = user_input[7:].strip()
                if new_model:
                    if not is_model_installed(new_model):
                        try:
                            pull_model_with_progress(new_model)
                            model = new_model
                            print(f"✅ Switched to model: {model}")
                        except Exception as e:
                            print(f"❌ Failed to load model '{new_model}': {e}")
                    else:
                        model = new_model
                        print(f"✅ Switched to model: {model}")
                else:
                    print("❌ Please specify a model name")
                continue
            
            # Get command suggestion
            try:
                print("🔄 Thinking...")
                command = get_command_suggestion(user_input, model)
                print(f"💡 Suggested command: {command}")
                
                # Ask if user wants to run it
                while True:
                    try:
                        choice = input("Run this command? [y/N/e(dit)]: ").strip().lower()
                        if choice in ['', 'n', 'no']:
                            break
                        elif choice in ['y', 'yes']:
                            print(f"🚀 Running: {command}")
                            success = execute_command(command)
                            if success:
                                print("✅ Command completed")
                            else:
                                print("❌ Command failed")
                            break
                        elif choice in ['e', 'edit']:
                            try:
                                edited_command = input(f"Edit command [{command}]: ").strip()
                                if edited_command:
                                    command = edited_command
                                print(f"🚀 Running: {command}")
                                success = execute_command(command)
                                if success:
                                    print("✅ Command completed")
                                else:
                                    print("❌ Command failed")
                                break
                            except (EOFError, KeyboardInterrupt):
                                print("\n❌ Cancelled")
                                break
                        else:
                            print("Please enter 'y' for yes, 'n' for no, or 'e' to edit")
                    except (EOFError, KeyboardInterrupt):
                        print("\n❌ Cancelled")
                        break
                        
            except Exception as e:
                print(f"❌ Error: {e}")
                
        except KeyboardInterrupt:
            print("\n^C")
            continue
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            continue

    # Cleanup
    try:
        print(f"🕒 Stopping model: {model}...")
        subprocess.run(["ollama", "stop", model], capture_output=True)
    except:
        pass