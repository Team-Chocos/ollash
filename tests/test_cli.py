import subprocess
import sys


def test_shell_launch_and_exit():
    """
    Test that `ollash shell --model=llama3` launches and exits cleanly with `:exit`.
    """
    result = subprocess.run(
        [sys.executable, "-m", "ollash", "shell", "--model=llama3"],
        input=":exit\n",
        capture_output=True,
        text=True,
        timeout=15
    )

    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

    assert result.returncode == 0
    assert "Loaded model" in result.stdout or "Type ':exit'" in result.stdout
