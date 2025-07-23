import argparse
from ollash.utils import ensure_ollama_ready
from ollash.ollama_nl2bash import run_nl_to_bash
from ollash.config import load_config



def main():

    config = load_config()

    parser = argparse.ArgumentParser(description="Ollash: Natural Language to Terminal Command")
    parser.add_argument("prompt", nargs="+", help="Your natural language instruction")
    parser.add_argument("--autostop", type=int, help="Time in seconds to auto-unload model")
    parser.add_argument("--model", help="Ollama model to use (default: llama3)")
    args = parser.parse_args()

    autostop = args.autostop if args.autostop else config.get("autostop")
    model = args.model if args.model else config.get("model")

    if not args.prompt:
        print("❌ Please provide an instruction. Example:\nollash --autostop 300 list all files")
        return

    ensure_ollama_ready()
    run_nl_to_bash(" ".join(args.prompt), autostop=autostop, model=model)
