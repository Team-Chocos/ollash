# ollash/shell.py
import subprocess
import os
from ollash.utils import ensure_ollama_ready, is_model_installed, pull_model_with_progress
from ollash.history import OptimizedHistoryLogger
from ollash.menu_advanced import get_model_selection_advanced
from ollash.ui import (
    ThinkingAnimation, clear_screen, print_banner, print_help, 
    format_prompt, print_status, print_suggested_command, 
    print_context_info, print_execution_start, print_execution_result,
    print_history_entries
)
from ollash.commands import (
    get_contextual_command_suggestion, get_command_suggestion,
    execute_command, input_with_prefill, debug_ollama_status
)


def main(model=None):
    """Main REPL shell function with semantic search"""
    # Initialize the optimized history logger
    history = OptimizedHistoryLogger()
    
    # Interactive model selection if no model specified
    if not model:
        selection = get_model_selection_advanced(method="inquirer")
        
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
                handle_search_command(user_input, history)
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
    animation = None
    
    try:
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
            
            if not query.strip():
                print_status("Please provide a search query", "error", in_box=False)
                return
                
            try:
                animation = ThinkingAnimation("Analyzing with context")
                animation.start()
                command, context = get_contextual_command_suggestion(query, model, history)
                animation.stop()
                animation = None
                
                if not command or not command.strip():
                    print_status("No command suggestion generated from context", "warning", in_box=False)
                    return
                
                has_context = bool(context and context.strip())
                print_suggested_command(command, has_context)
                
                if has_context:
                    print_context_info(context)
                
                # Ask if user wants to run it
                execute_with_confirmation(command, query, history)
                
            except Exception as e:
                if animation:
                    animation.stop()
                print_status(f"Error getting contextual suggestion: {e}", "error", in_box=False)
                
                # Try fallback
                fallback = get_fallback_suggestion(query)
                if fallback:
                    print_status(f"Fallback suggestion: {fallback}", "info", in_box=False)
                    choice = input("│ Try fallback suggestion? [y/N] ❯ ").strip().lower()
                    if choice in ['y', 'yes']:
                        execute_with_confirmation(fallback, query, history)
                        
    except Exception as e:
        if animation:
            animation.stop()
        print_status(f"History command error: {e}", "error", in_box=False)


def handle_search_command(user_input, history):
    """Handle :search commands"""
    query = user_input[8:].strip()
    if query:
        try:
            animation = ThinkingAnimation("Retrieving context")
            animation.start()
            similar_entries = history.search_similar(query, limit=5)
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
    animation = None
    try:
        animation = ThinkingAnimation("Generating command")
        animation.start()
        
        # Validate input before processing
        if not user_input or not user_input.strip():
            raise ValueError("Empty input provided")
        
        command = get_command_suggestion(user_input, model)
        animation.stop()
        animation = None
        
        # Validate the returned command
        if not command or not command.strip():
            print_status("No command suggestion generated", "warning", in_box=False)
            # Try fallback immediately
            fallback_suggestion = get_fallback_suggestion(user_input)
            if fallback_suggestion:
                print_status(f"Fallback suggestion: {fallback_suggestion}", "info", in_box=False)
                choice = input("│ Try fallback suggestion? [y/N] ❯ ").strip().lower()
                if choice in ['y', 'yes']:
                    execute_with_confirmation(fallback_suggestion, user_input, history)
            return
        
        print_suggested_command(command, False)
        execute_with_confirmation(command, user_input, history)
        
    except Exception as e:
        if animation:
            animation.stop()
        
        error_message = str(e)
        print_status(f"Error: {error_message}", "error", in_box=False)
        
        # If it's an Ollama-related error, provide debug info
        if any(keyword in error_message.lower() for keyword in ['ollama', 'connection refused', 'model not found']):
            debug_info = debug_ollama_status()
            print_status(f"Debug: {debug_info}", "info", in_box=False)
        
        # Provide fallback suggestion based on common patterns
        fallback_suggestion = get_fallback_suggestion(user_input)
        if fallback_suggestion:
            print_status(f"Fallback suggestion: {fallback_suggestion}", "info", in_box=False)
            choice = input("│ Try fallback suggestion? [y/N] ❯ ").strip().lower()
            if choice in ['y', 'yes']:
                execute_with_confirmation(fallback_suggestion, user_input, history)
        else:
            print_status("No fallback suggestion available", "warning", in_box=False)


def execute_with_confirmation(command, original_input, history):
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
                          os.getcwd())
                print("│ Learning from this command...")
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
                                  os.getcwd())
                        print("│ Learning from this command...")
                    break
                except (EOFError, KeyboardInterrupt):
                    print("\n│ Cancelled")
                    break
            else:
                print("│ Enter 'y' (yes), 'n' (no), or 'e' (edit)")
        except (EOFError, KeyboardInterrupt):
            print("\n│ Cancelled")
            break


def get_fallback_suggestion(user_input):
    """Provide basic fallback suggestions for common patterns"""
    user_input_lower = user_input.lower().strip()
    
    # More specific pattern matching with better command generation
    
    # Create/make patterns
    if any(word in user_input_lower for word in ['create', 'make', 'new']):
        if any(word in user_input_lower for word in ['folder', 'directory', 'dir']):
            # Extract folder name - look for "name" followed by the actual name
            words = user_input.split()
            folder_name = None
            
            # Look for patterns like "folder name xyz" or "create folder xyz"
            for i, word in enumerate(words):
                if word.lower() in ['name', 'called', 'named']:
                    if i + 1 < len(words):
                        folder_name = words[i + 1]
                        break
                elif word.lower() in ['folder', 'directory']:
                    if i + 1 < len(words) and words[i + 1].lower() not in ['name', 'called', 'named']:
                        folder_name = words[i + 1]
                        break
            
            # Fallback: take the last word as folder name
            if not folder_name and len(words) > 0:
                folder_name = words[-1]
            
            return f'mkdir {folder_name}' if folder_name else 'mkdir NEW_FOLDER'
        
        elif any(word in user_input_lower for word in ['file']):
            # Extract file name
            words = user_input.split()
            file_name = None
            
            for i, word in enumerate(words):
                if word.lower() in ['name', 'called', 'named']:
                    if i + 1 < len(words):
                        file_name = words[i + 1]
                        break
                elif word.lower() in ['file']:
                    if i + 1 < len(words) and words[i + 1].lower() not in ['name', 'called', 'named']:
                        file_name = words[i + 1]
                        break
            
            if not file_name and len(words) > 0:
                file_name = words[-1]
            
            return f'touch {file_name}' if file_name else 'touch NEW_FILE'
    
    # Find patterns
    if any(word in user_input_lower for word in ['find', 'locate', 'search', 'where is', 'look for']):
        # Extract search term
        words = user_input.split()
        search_term = None
        
        # Remove common words and find the actual search term
        skip_words = ['find', 'locate', 'search', 'where', 'is', 'look', 'for', 'a', 'the', 'folder', 'file', 'directory', 'name', 'named', 'called']
        search_words = [w for w in words if w.lower() not in skip_words]
        
        if search_words:
            search_term = search_words[-1]  # Take the last meaningful word
        
        if any(word in user_input_lower for word in ['folder', 'directory']):
            return f'find . -name "*{search_term}*" -type d' if search_term else 'find . -type d'
        else:
            return f'find . -name "*{search_term}*"' if search_term else 'find .'
    
    # Copy patterns
    if any(word in user_input_lower for word in ['copy', 'cp']):
        words = user_input.split()
        # This is tricky without more context, provide a template
        return 'cp SOURCE DESTINATION'
    
    # Move patterns
    if any(word in user_input_lower for word in ['move', 'mv']):
        return 'mv SOURCE DESTINATION'
    
    # Delete patterns
    if any(word in user_input_lower for word in ['delete', 'remove', 'rm']):
        words = user_input.split()
        target = None
        skip_words = ['delete', 'remove', 'rm', 'the', 'a']
        target_words = [w for w in words if w.lower() not in skip_words]
        if target_words:
            target = target_words[-1]
        return f'rm -rf {target}' if target else 'rm -rf TARGET'
    
    # List patterns
    if any(word in user_input_lower for word in ['list', 'show', 'ls']):
        if any(word in user_input_lower for word in ['all', 'hidden']):
            return 'ls -la'
        return 'ls -l'
    
    # Default: if we can't match anything specific, try to extract the last word as a filename/foldername
    words = user_input.split()
    if words:
        last_word = words[-1]
        # If it looks like a create operation, default to mkdir
        if any(word in user_input_lower for word in ['create', 'make', 'new']):
            return f'mkdir {last_word}'
    
    return None


def cleanup(history, model):
    """Cleanup when exiting"""
    try:
        history.shutdown()
        animation = ThinkingAnimation(f"Stopping model: {model}")
        animation.start()
        subprocess.run(["ollama", "stop", model], capture_output=True, text=True, errors="ignore", encoding="utf-8")
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