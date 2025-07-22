from ollash import ollama_nl2bash
import subprocess

def test_command_parsing_with_backticks(monkeypatch):
    output = "Here is your command:\n\n`ls -al`\nThis will list all files."
    monkeypatch.setattr("subprocess.run", lambda *a, **k: type("res", (), {"stdout": output}))
    command = ollama_nl2bash.extract_command(output)
    assert command == "ls -al"

def test_command_parsing_no_backticks(monkeypatch):
    output = "ls -la\n\nThis lists contents."
    monkeypatch.setattr("subprocess.run", lambda *a, **k: type("res", (), {"stdout": output}))
    command = ollama_nl2bash.extract_command(output)
    assert command == "ls -la"

def extract_command(raw_output: str) -> str:
    import re
    match = re.search(r"`([^`]+)`", raw_output)
    return match.group(1).strip() if match else raw_output.strip().splitlines()[0]

def test_model_selection(monkeypatch):
    called = {}

    def fake_run(cmd, **kwargs):
        called["model"] = cmd[2]
        class Result:
            stdout = "ls"
        return Result()

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr("builtins.input", lambda _: "n")

    ollama_nl2bash.run_nl_to_bash("show logs", model="mistral")
    assert called["model"] == "mistral"

