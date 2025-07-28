

# Ollash

**Ollash** is a lightweight, extensible command-line tool that transforms natural language instructions into safe, valid shell commands. It also offers an interactive terminal shell and local codebase querying—all powered entirely by local models. Ollash is designed to run securely, privately, and offline, leveraging your local computing resources for fast, intelligent command generation.

The project bundles:

* An **NL2Bash engine** using local large language models (via [Ollama](https://ollama.com)), automatically installed and launched as needed
* An **interactive REPL shell** interface with persistent model context
* A **codebase question-answering system** using `interro`, enabling semantic understanding of local source code

All session state, configuration, and cache are stored within a `.ollash` folder for continuity across runs.

---

## Project Architecture

Ollash includes the following components:

1. **Natural Language to Shell Command Generator**

   * Converts English prompts to safe Bash commands
   * Uses your selected LLM (e.g., `llama3`, `qwen`, `gemma`) via Ollama backend
   * Automatically ensures Ollama is installed, models are pulled, and the backend is ready

2. **Interactive Shell (`ollash shell`)**

   * A stateful terminal where each prompt builds on the previous context
   * Provides a REPL interface for hands-free shell navigation
   * Model is loaded and unloaded on session start and exit

3. **Codebase Q\&A (`ollash ask`)**

   * Integrates with [`interro`](https://github.com/slaterlabs/interro) to semantically query your local code
   * Supports both LLM-based explanations and retrieval-only search
   * Useful for onboarding, code understanding, or quick debugging

   * All temporary and persistent state is handled automatically

---

## Installation

For better UI experience we recommend using [`figlet`](https://github.com/xero/figlet-fonts) and [`fzf`](https://github.com/junegunn/fzf)

```bash
sudo apt install figlet fzf
```

Cross-Platform Support for Linux and MacOS, currently experimental for Windows

```bash
pip install ollash
```

Dependencies like Ollama and Interro are invoked internally; the tool verifies their presence and offers installation guidance if missing.

---

## Usage

### One-Shot Command

```bash
ollash ind all PDF files larger than 10MB and archive them
```

This runs a single prompt through your local model and prints the resulting shell command to `stdout`.

---

### Interactive Shell


```bash
ollash shell 
```

Launches a full REPL interface where each command can access past context. Model is automatically loaded after an interactive selection screen using Ollama on startup and unloaded on exit.

---

### Ask About Your Codebase

```bash
ollash ask "Where is the user authentication logic?" --path ./src --no-llm
```

You can also enable LLM explanations:

```bash
ollash ask "Explain how login works" --path ./src
```

Options:

* `--path`: Folder containing codebase to index (default: `.`)
* `--no-llm`: Disables language model and uses semantic retrieval only
* `--config`: Provide custom Interro config if needed

---

## CLI Overview

```
usage: ollash [shell|run|ask] [options]

Commands:
  shell         Start interactive REPL shell
  run           One-shot shell command from natural language
  ask           Ask questions about your codebase using Interro

Options:
  --model       Model to use (default from config)
  --autostop    Max token limit before cutting off output (run mode)
  --path        Codebase root directory (ask mode)
  --no-llm      Disable LLM responses for ask (retrieval-only)
  --config      Path to interro config YAML (optional)
```

---
## License

MIT License
© 2025 Team Ollash

