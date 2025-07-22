import sys
import time

def show_download_progress():
    print("Downloading Ollama...", end="")
    for _ in range(30):
        sys.stdout.write(".")
        sys.stdout.flush()
        time.sleep(0.1)
    print(" ✅ Done!")
