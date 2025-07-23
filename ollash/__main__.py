from pathlib import Path
import yaml
from platformdirs import user_config_dir
import argparse
from ollash.utils import ensure_ollama_ready
from ollash.ollama_nl2bash import run_nl_to_bash


def load_config():
    config_path = Path(user_config_dir("ollash")) / "config.yaml"
    if config_path.exists():
        with open(config_path, "r")as f:
            return yaml.safe_load(f) or {}
    return {}


def main():

    config = load_config()

    parser = argparse.ArgumentParser(description="Ollash: Natural Language to Terminal Command")
    parser.add_argument("prompt", nargs="+", help="Your natural language instruction")
    parser.add_argument("--autostop", type=int, help="Time in seconds to auto-unload model")
    parser.add_argument("--model", help="Ollama model to use (default: llama3)")
    args = parser.parse_args()

    autostop = args.autostop if args.autostop else config.get("autostop")
    if args.model:
        model = args.model
    elif config.get("model"):
        model = config.get("model")
    else:
        model = "llama3"

    if not args.prompt:
        print("❌ Please provide an instruction. Example:\nollash --autostop 300 list all files")
        return

    ensure_ollama_ready()
    run_nl_to_bash(" ".join(args.prompt), autostop=autostop, model=model)
