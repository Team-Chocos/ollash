from ollash import ollama_nl2bash

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
