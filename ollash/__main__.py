import argparse
import sys
import os
from ollash.utils import ensure_ollama_ready
from ollash.ollama_nl2bash import run_nl_to_bash
from ollash.config import load_config
from ollash.shell import main as shell_main

# Check if interro CLI is available
import subprocess
import shutil

def check_interro_available():
    return shutil.which('interro') is not None

INTERRO_AVAILABLE = check_interro_available()


def run_interro_query(query, path=None, use_llm=True):
    """Run an interro query using the CLI tool."""
    if not INTERRO_AVAILABLE:
        print("Error: interro CLI not found. Install with: pip install interro")
        return
    
    try:
        # Build the interro command
        cmd = ['interro', 'ask', query]
        if path:
            cmd.extend(['--path', path])
        if not use_llm:
            cmd.append('--no-llm')
        
        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=path or os.getcwd())
        
        if result.returncode == 0:
            print(result.stdout)
        else:
            print(f"Error running interro: {result.stderr}")
            
    except Exception as e:
        print(f"Error running interro query: {e}")


def main():
    config = load_config()

    # Handle the case where user types 'ollash ask "question"' directly
    if len(sys.argv) > 1 and sys.argv[1] not in {"shell", "run", "ask", "-h", "--help"}:
        sys.argv.insert(1, "run")

    parser = argparse.ArgumentParser(
        prog="ollash",
        description="Ollash: Natural Language to Terminal Command with Code Search"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # shell subcommand
    shell_parser = subparsers.add_parser("shell", help="Start interactive REPL shell")
    shell_parser.add_argument("--model", type=str, default=config.get("model"))

    # run subcommand
    run_parser = subparsers.add_parser("run", help="One-shot command from natural language")
    run_parser.add_argument("prompt", nargs="+")
    run_parser.add_argument("--model", type=str, default=config.get("model"))
    run_parser.add_argument("--autostop", type=int, default=config.get("autostop"))

    # ask subcommand (interro integration)
    ask_parser = subparsers.add_parser("ask", help="Ask questions about your codebase using interro")
    ask_parser.add_argument("query", nargs="+", help="Question about the codebase")
    ask_parser.add_argument("--path", type=str, default=None, help="Path to index (default: current directory)")
    ask_parser.add_argument("--config", type=str, default=None, help="Path to interro config file")
    ask_parser.add_argument("--no-llm", action="store_true", help="Disable LLM explanation")

    args = parser.parse_args()

    if args.command == "shell":
        ensure_ollama_ready()
        if "--model" in sys.argv:
            shell_main(model=args.model)
        else:
            shell_main(model=None)

    elif args.command == "run":
        ensure_ollama_ready()
        run_nl_to_bash(" ".join(args.prompt), autostop=args.autostop, model=args.model)
    
    elif args.command == "ask":
        if not INTERRO_AVAILABLE:
            print("Error: interro CLI not found.")
            print("Install with: pip install interro")
            return
        
        query = " ".join(args.query)
        
        # Build the interro command
        cmd = ['interro', 'ask', query]
        if args.path:
            cmd.extend(['--path', args.path])
        if args.no_llm:
            cmd.append('--no-llm')
        if args.config:
            cmd.extend(['--config', args.config])
        
        try:
            # Run the command and stream output
            result = subprocess.run(cmd, cwd=args.path or os.getcwd())
            
        except Exception as e:
            print(f"Error running interro: {e}")


if __name__ == "__main__":
    main()