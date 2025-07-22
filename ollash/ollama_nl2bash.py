import subprocess
import sys
import re
from .utils import get_os_label
from .utils import schedule_model_shutdown


def run_nl_to_bash(prompt: str, autostop=None):
    # prompt = " ".join(sys.argv[1:])
    if not prompt:
        print("Usage: ollash <natural language command>")
        return

    os_label = get_os_label()

    ollama_cmd = [
        "ollama", "run", "llama3", 
        f"Translate the following instruction into a safe {os_label} terminal command. Respond ONLY with the command, no explanation:\nInstruction: {prompt}"
    ]


    response = subprocess.run(
        ollama_cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",       # safer encoding
        errors="ignore"         # skip bad chars instead of crashing
    )

    raw_output = response.stdout.strip()

    command = extract_command(raw_output)

    print(f"\nSuggested command for {os_label}:\n> {command}")
    confirm = input("Run this command? (y/N): ").strip().lower()
    if confirm == 'y':
        subprocess.run(command, shell=True)
    if autostop:
        print(f"🕒 Auto-unloading model after {autostop} seconds of inactivity...")
        schedule_model_shutdown(timeout=autostop)


def extract_command(raw_output: str) -> str:
    import re
    match = re.search(r"`([^`]+)`", raw_output)
    return match.group(1).strip() if match else raw_output.strip().splitlines()[0]

