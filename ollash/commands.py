# ollash/commands.py
import subprocess
import re
from ollash.utils import is_model_installed, pull_model_with_progress, get_os_label

try:
    import readline
    HAS_READLINE = True
except ImportError:
    HAS_READLINE = False


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


def get_contextual_command_suggestion(prompt: str, model: str, history) -> tuple[str, str]:
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
        f"""You are a shell assistant. Translate the user's instruction into a **safe, valid, bash-compatible** terminal command.

            Rules:
            - Return **only the command**, no explanation or formatting.
            - Avoid unsafe actions (e.g., `rm -rf /`), use `--dry-run`, `-i`, or similar when appropriate.
            - Chain commands with `&&` if needed.
            - Quote or escape paths with spaces/special characters.
            - When unsure, choose the safest and most common interpretation.

            Instruction: {prompt}"""
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