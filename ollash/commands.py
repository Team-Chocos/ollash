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
    
    # Search for similar past entries (removed model parameter)
    similar_entries = history.search_similar(prompt, potential_command, limit=3)
    
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

        if response.returncode != 0:
            raise Exception(f"Ollama command failed with return code {response.returncode}")

        raw_output = response.stdout.strip()
        
        if not raw_output:
            raise Exception("Empty response from model")
        
        # Extract command with better error handling
        command = _extract_command(raw_output)
        
        if not command:
            raise Exception("Could not extract valid command from response")
        
        return command, context
        
    except Exception as e:
        raise Exception(f"Failed to get command suggestion: {e}")


def _extract_command(raw_output: str) -> str:
    """Extract command from model output with robust parsing"""
    if not raw_output or not raw_output.strip():
        return ""
    
    # Try to find command in backticks first
    match = re.search(r"`([^`]+)`", raw_output)
    if match:
        command = match.group(1).strip()
        if command:
            return _clean_command(command)
    
    # Try to find command in code blocks
    code_block_match = re.search(r"```(?:bash|sh|shell)?\n?([^`]+)```", raw_output, re.MULTILINE)
    if code_block_match:
        command = code_block_match.group(1).strip()
        if command:
            return _clean_command(command)
    
    # Fall back to first non-empty line
    lines = raw_output.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):  # Skip comments
            return _clean_command(line)
    
    return ""


def _clean_command(command: str) -> str:
    """Clean up common formatting issues in commands"""
    if not command:
        return ""
    
    # Remove common prefixes and formatting
    command = command.strip()
    command = re.sub(r'^(bash|sh|shell|\$)\s*', '', command)
    command = command.replace("```", "").strip()
    
    # Remove leading $ or # if present
    if command.startswith(('$ ', '# ')):
        command = command[2:]
    elif command.startswith(('$', '#')):
        command = command[1:]
    
    return command.strip()


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
    elif "find" in prompt_lower or "search" in prompt_lower or "where" in prompt_lower:
        return "find" if os_label != "Windows" else "findstr"
    elif "install" in prompt_lower:
        return "apt install" if os_label == "Linux" else "brew install" if os_label == "macOS" else "choco install"
    
    return ""


def get_command_suggestion(prompt: str, model: str) -> str:
    """Original command suggestion function with improved error handling"""
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

        if response.returncode != 0:
            raise Exception(f"Ollama command failed with return code {response.returncode}")

        raw_output = response.stdout.strip()
        
        if not raw_output:
            raise Exception("Empty response from model")
        
        # Extract command with better error handling
        command = _extract_command(raw_output)
        
        if not command:
            raise Exception("Could not extract valid command from response")
        
        return command
        
    except Exception as e:
        raise Exception(f"Failed to get command suggestion: {e}")


def execute_command(command: str) -> bool:
    """Execute a command and return True if successful"""
    try:
        result = subprocess.run(command, shell=True, text=True, encoding="utf-8", errors="ignore")
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\n│ [Interrupted]")
        return False
    except Exception as e:
        print(f"│ Error: {e}")
        return False


def debug_ollama_status():
    """Debug function to check Ollama status"""
    try:
        # Check if ollama is available
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            return "Ollama not installed or not in PATH"
        
        # Check if ollama service is running
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            error = result.stderr.strip() if result.stderr else "Unknown error"
            if "connection refused" in error.lower():
                return "Ollama service is not running. Run 'ollama serve' in another terminal."
            return f"Ollama service error: {error}"
        
        # Try a simple test command to see if the model actually works
        try:
            test_result = subprocess.run(
                ["ollama", "run", "llama3:latest", "Say 'test' and nothing else"],
                capture_output=True,
                text=True,
                timeout=15
            )
            if test_result.returncode != 0:
                error = test_result.stderr.strip() if test_result.stderr else "Unknown error"
                return f"Model execution failed: {error}"
            elif not test_result.stdout.strip():
                return "Model returns empty responses - may need restart"
            else:
                return "Ollama is running correctly"
        except subprocess.TimeoutExpired:
            return "Model is responding very slowly - may need restart"
        
    except FileNotFoundError:
        return "Ollama not installed"
    except subprocess.TimeoutExpired:
        return "Ollama is not responding (timeout)"
    except Exception as e:
        return f"Error checking Ollama: {e}"