# 🧠 Ollash — Natural Language to Safe Terminal Commands (Linux, macOS, Windows)

Ollash is a CLI tool that lets you run terminal commands by simply typing natural language.  
Powered by [Ollama](https://ollama.com), it translates instructions like:

> "List all `.log` files in the current directory"

into safe and OS-specific terminal commands such as:

```bash
# On Linux/macOS:
ls *.log

# On Windows:
dir *.log
````

---

## 🚀 Features

* 🔁 Converts plain English to terminal commands (Bash, Zsh, PowerShell)
* ✅ Shows the command before execution and asks for confirmation
* 🧠 Uses local LLMs via [Ollama](https://ollama.com) — no API keys or cloud usage
* 🧩 Auto-installs Ollama if it's not available
* 🔄 Automatically starts the Ollama daemon with `llama3` if it's not running
* 💻 Cross-platform support:

  * 🐧 Linux
  * 🍎 macOS (Terminal, Zsh/Bash)
  * 🪟 Windows (PowerShell)
* 📦 Fully pip-installable as a CLI tool

---

## ⚙️ Installation

### 1. Clone the repo (for local use or development)

```bash
git clone https://github.com/codexx07/ollash.git
cd ollash
pip install .
```

OR use editable mode during development:

```bash
pip install -e .
```

Once installed, you can run:

```bash
ollash check disk usage
```

---

## 📦 Prerequisites

> Ollash depends on [Ollama](https://ollama.com) to run LLMs locally on your machine.

If Ollama is not installed, `ollash` will:

* Show a disclaimer and ask for permission
* Automatically install Ollama from [https://ollama.com](https://ollama.com)
* Start the Ollama daemon using the `llama3` model

✅ No internet is needed for inference after initial model download.

---

## 💻 OS-Specific Behavior

Ollash detects your OS and generates commands accordingly:

| OS      | Shell Type | Sample Output |
| ------- | ---------- | ------------- |
| Linux   | Bash/Zsh   | `ls -la`      |
| macOS   | Bash/Zsh   | `ls -la`      |
| Windows | PowerShell | `dir /a:h`    |

---

## 🛠 Example Usage

```bash
ollash make a new folder named logs
```

Sample Output:

```
🧠 Suggested command for Linux:
> mkdir logs

Run this command? (y/N): y
```

---

## 🛡 Disclaimer

This tool may generate commands that can alter your system. Always **read and confirm the command** before running.

The authors are **not responsible** for any unintended consequences or damage caused by executing generated commands.

---

## 📃 License

MIT License

---

## ✨ Credits

* [Ollama](https://ollama.com) for enabling local LLM inference
* [Platform](https://docs.python.org/3/library/platform.html) for OS detection
* You, for using and improving this tool
# ollash
