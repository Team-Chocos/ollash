import pytest
from ollash import utils

def test_is_ollama_installed_when_missing(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: None)
    assert not utils.is_ollama_installed()

def test_is_ollama_installed_when_present(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/ollama")
    assert utils.is_ollama_installed()

def test_is_ollama_running_success(monkeypatch):
    class MockResult:
        returncode = 0
    monkeypatch.setattr("subprocess.run", lambda *a, **k: MockResult())
    assert utils.is_ollama_running()

def test_is_ollama_running_fail(monkeypatch):
    class MockResult:
        returncode = 1
    monkeypatch.setattr("subprocess.run", lambda *a, **k: MockResult())
    assert not utils.is_ollama_running()

import platform
from ollash.utils import get_os_label, install_ollama
import subprocess
import os
import builtins

def test_get_os_label_windows(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    assert get_os_label() == "Windows"

def test_get_os_label_linux(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    assert get_os_label() == "Linux"

def test_get_os_label_mac(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    assert get_os_label() == "macOS"

def test_get_os_label_unknown(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Solaris")
    assert get_os_label() == "Unknown"

def test_install_ollama_linux(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr("builtins.input", lambda _: "y")
    monkeypatch.setattr("ollash.progress.show_download_progress", lambda: None)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: None)

    try:
        install_ollama()
    except SystemExit:
        assert False, "install_ollama() exited unexpectedly for Linux"

def test_install_ollama_macos(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr("builtins.input", lambda _: "y")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: None)

    try:
        install_ollama()
    except SystemExit:
        pass  # macOS branch exits after opening browser

def test_install_ollama_windows(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    monkeypatch.setattr("builtins.input", lambda _: "y")

    # Only mock os.startfile if it exists (e.g., skip on Linux/macOS)
    if hasattr(os, "startfile"):
        monkeypatch.setattr(os, "startfile", lambda _: None)
    else:
        monkeypatch.setattr("os.startfile", lambda _: None, raising=False)

    try:
        install_ollama()
    except SystemExit:
        pass  

def test_install_ollama_unknown(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Solaris")
    monkeypatch.setattr("builtins.input", lambda _: "y")

    try:
        install_ollama()
    except SystemExit:
        pass  

