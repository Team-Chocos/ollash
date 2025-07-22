import threading
import builtins
from ollash.ollama_nl2bash import run_nl_to_bash
from ollash.utils import schedule_model_shutdown

def test_autostop_schedules_timer(monkeypatch):
    called = {}

    class MockTimer:
        def __init__(self, timeout, func):
            called['timeout'] = timeout
            called['scheduled'] = True
        def start(self): pass

    monkeypatch.setattr(threading, "Timer", MockTimer)
    monkeypatch.setattr(builtins, "input", lambda _: "n")

    run_nl_to_bash("list files", autostop=120)

    assert called['scheduled'] is True
    assert called['timeout'] == 120

def test_no_autostop_does_not_schedule_timer(monkeypatch):
    monkeypatch.setattr(builtins, "input", lambda _: "n")
    def mock_timer(*a, **k):
        raise AssertionError("Timer should not be called")
    monkeypatch.setattr(threading, "Timer", mock_timer)

    run_nl_to_bash("show hidden files", autostop=None)
