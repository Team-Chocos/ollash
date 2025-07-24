import argparse
import sys
from ollash.utils import ensure_ollama_ready
from ollash.ollama_nl2bash import run_nl_to_bash
from ollash.config import load_config
from ollash.shell import main as shell_main


def main():
    config = load_config()

    # 🛠 insert fallback subcommand BEFORE building the parser
    if len(sys.argv) > 1 and sys.argv[1] not in {"shell", "run", "-h", "--help"}:
        sys.argv.insert(1, "run")

    parser = argparse.ArgumentParser(
        prog="ollash",
        description="Ollash: Natural Language to Terminal Command"
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

    args = parser.parse_args()

    if args.command == "shell":
        if "--model" in sys.argv:
            shell_main(model=args.model)
        else:
            shell_main(model=None)

    elif args.command == "run":
        ensure_ollama_ready()
        run_nl_to_bash(" ".join(args.prompt), autostop=args.autostop, model=args.model)

