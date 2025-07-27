# ollash/shell.py
import subprocess
import os
from ollash.menu_advanced import select_model_two_stage_inquirer
from ollash.utils import ensure_ollama_ready, is_model_installed, pull_model_with_progress
from ollash.history import HistoryLogger
from ollash.menu_advanced import get_model_selection_advanced
from ollash.ui import (
    ThinkingAnimation, clear_screen, print_banner, print_help, 
    format_prompt, print_status, print_suggested_command, 
    print_context_info, print_execution_start, print_execution_result,
    print_history_entries
)
from ollash.commands import (
    get_contextual_command_suggestion, get_command_suggestion,
    execute_command, input_with_prefill
)




def main(model=None):
    """Main REPL shell function with semantic search"""
    history = HistoryLogger(model)
    
    # Interactive model selection if no model specified
    if not model:
        selection = select_model_two_stage_inquirer()
        
        if not selection:
            print("No model selected. Exiting...")
            return
        
        backend, model = selection
    else:
        backend = "ollama"
    
    print(f"\nStarting Ollash with {model} on {backend}")
    
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
                handle_history_command(user_input, history, model)
                continue
            
            elif user_input.startswith(":search "):
                handle_search_command(user_input, history, model)
                continue
            
            elif user_input.startswith(":model "):
                model = handle_model_command(user_input, model)
                continue
                
            elif user_input.startswith(":sh "):
                handle_shell_command(user_input)
                continue

            # Handle regular command suggestion
            handle_command_suggestion(user_input, model, history)
            print()
                
        except KeyboardInterrupt:
            print()
            continue
        except Exception as e:
            print_status(f"Unexpected error: {e}", "error", in_box=False)
            continue

    # Cleanup
    cleanup(history, model)


def handle_history_command(user_input, history, model):
    """Handle :history commands"""
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
            execute_with_confirmation(command, query, history, model)
        except Exception as e:
            print_status(f"Error: {e}", "error", in_box=False)


def handle_search_command(user_input, history, model):
    """Handle :search commands"""
    query = user_input[8:].strip()
    if query:
        try:
            animation = ThinkingAnimation("Retrieving context")
            animation.start()
            similar_entries = history.search_similar(query, limit=5, model=model)
            animation.stop()
            
            if similar_entries:
                entries = [entry for entry, _ in similar_entries]
                print_history_entries(entries, f"Search Results for '{query}'")
            else:
                print_status(f"No matches found for '{query}'", "info", in_box=False)
        except Exception as e:
            animation.stop()
            print_status(f"Search error: {e}", "error", in_box=False)
    else:
        print_status("Please provide a search query", "error", in_box=False)


def handle_model_command(user_input, current_model):
    """Handle :model commands"""
    new_model = user_input[7:].strip()
    if new_model:
        if not is_model_installed(new_model):
            try:
                animation = ThinkingAnimation(f"Installing model '{new_model}'")
                animation.start()
                pull_model_with_progress(new_model)
                animation.stop()
                print_status(f"Switched to model: {new_model}", "success", in_box=False)
                return new_model
            except Exception as e:
                print_status(f"Failed to load model '{new_model}': {e}", "error", in_box=False)
                return current_model
        else:
            print_status(f"Switched to model: {new_model}", "success", in_box=False)
            return new_model
    else:
        print_status("Please specify a model name", "error", in_box=False)
        return current_model


def handle_shell_command(user_input):
    """Handle :sh commands"""
    command = user_input[4:]
    print_execution_start(command)
    success = execute_command(command)
    print_execution_result(success)


def handle_command_suggestion(user_input, model, history):
    """Handle regular command suggestions"""
    try:
        animation = ThinkingAnimation("Generating command")
        animation.start()
        command = get_command_suggestion(user_input, model)
        animation.stop()
        
        print_suggested_command(command, False)
        execute_with_confirmation(command, user_input, history, model)
        
    except Exception as e:
        print_status(f"Error: {e}", "error", in_box=False)


def execute_with_confirmation(command, original_input, history, model):
    """Handle command execution with user confirmation"""
    while True:
        try:
            choice = input("│ Execute? [y/N/e(dit)] ❯ ").strip().lower()
            if choice in ['', 'n', 'no']:
                break
            elif choice in ['y', 'yes']:
                print_execution_start(command)
                success = execute_command(command)
                print_execution_result(success)
                history.log(original_input, command, "success" if success else "failure", 
                          os.getcwd(), model=model, generate_embedding=True)
                print("│ \033[90mLearning from this command...\033[0m")
                break
            elif choice in ['e', 'edit']:
                try:
                    edited_command = input_with_prefill("│ Edit ❯ ", command).strip()
                    if edited_command:
                        command = edited_command
                        print_execution_start(command)
                        success = execute_command(command)
                        print_execution_result(success)
                        history.log(original_input, command, "success" if success else "failure", 
                                  os.getcwd(), model=model, generate_embedding=True)
                        print("│ \033[90mLearning from this command...\033[0m")
                    break
                except (EOFError, KeyboardInterrupt):
                    print("\n│ Cancelled")
                    break
            else:
                print("│ Enter 'y' (yes), 'n' (no), or 'e' (edit)")
        except (EOFError, KeyboardInterrupt):
            print("\n│ Cancelled")
            break


def cleanup(history, model):
    """Cleanup when exiting"""
    try:
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

    parser = argparse.ArgumentParser(description="Run Ollash REPL Shell")
    parser.add_argument("--model", type=str, help="Model name to use (e.g., llama3:8b)")
    args = parser.parse_args()

    main(model=args.model)
