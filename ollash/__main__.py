import argparse
from ollash.utils import ensure_ollama_ready
from ollash.ollama_nl2bash import run_nl_to_bash

def main():
    parser = argparse.ArgumentParser(description="Ollash — Run terminal commands from natural language.")
    parser.add_argument("instruction", nargs=argparse.REMAINDER, help="Natural language instruction to run.")
    parser.add_argument("--autostop", type=int, default=None, help="Unload the model after this many seconds of inactivity.")
    
    args = parser.parse_args()

    if not args.instruction:
        print("❌ Please provide an instruction. Example:\nollash --autostop 300 list all files")
        return

    ensure_ollama_ready()
    run_nl_to_bash(" ".join(args.instruction), autostop=args.autostop)
