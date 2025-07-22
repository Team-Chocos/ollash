import subprocess
import shutil
import sys
import os
import time
from .progress import show_download_progress
import platform


def get_os_label():
    os_name = platform.system().lower()
    if "windows" in os_name:
        return "Windows"
    elif "darwin" in os_name:
        return "macOS"
    elif "linux" in os_name:
        return "Linux"
    else:
        return "Unknown"


def is_ollama_installed():
    return shutil.which("ollama") is not None

def is_ollama_running():
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        return result.returncode == 0
    except Exception:
        return False

def install_ollama():
    os_label = get_os_label()

    print("\n🔐 DISCLAIMER:")
    print(f"This tool requires Ollama to run locally on {os_label}.")
    print("It will download Ollama from the official site and attempt installation.")
    print("Your system may prompt for administrator/root access to complete installation.")
    
    consent = input("Do you want to proceed with installing Ollama? (y/N): ").strip().lower()
    if consent != 'y':
        print("❌ Exiting as per user request.")
        sys.exit(1)

    print(f"\n📥 Installing Ollama on {os_label}...")

    if os_label == "Linux":
        time.sleep(1)
        show_download_progress()
        try:
            subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=True)
        except subprocess.CalledProcessError:
            print("❌ Failed to install Ollama on Linux. Please install manually: https://ollama.com/download")
            sys.exit(1)

    elif os_label == "macOS":
        print("📎 macOS detected. Opening official download page in your browser...")
        try:
            subprocess.run(["open", "https://ollama.com/download"], check=True)
        except:
            print("❌ Could not open browser. Please visit: https://ollama.com/download")
        sys.exit(1)

    elif os_label == "Windows":
        print("📎 Windows detected. Opening official .msi installer page...")
        try:
            os.startfile("https://ollama.com/download")
        except:
            print("❌ Could not launch browser. Please visit: https://ollama.com/download")
        sys.exit(1)

    else:
        print(f"❌ Unsupported OS: {os_label}. Please install Ollama manually: https://ollama.com/download")
        sys.exit(1)


def start_ollama_daemon():
    print("\n🚀 Starting Ollama daemon with llama3 model...")
    try:
        subprocess.Popen(["ollama", "run", "llama3"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(3)
    except Exception as e:
        print(f"❌ Failed to start Ollama: {e}")
        sys.exit(1)

def ensure_ollama_ready():
    if not is_ollama_installed():
        install_ollama()

    if not is_ollama_running():
        start_ollama_daemon()
        print("⏳ Waiting for daemon to start...")
        time.sleep(3)

        print("✅ Ollash is ready to use!\n")


def schedule_model_shutdown(timeout=300):
    os_name = platform.system().lower()

    if "windows" in os_name:
        # Use `timeout` on Windows (PowerShell / cmd-safe)
        subprocess.Popen(
            ["powershell", "-Command", f"Start-Sleep -Seconds {timeout}; ollama stop llama3"],
            creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        )
    else:
        # POSIX-compatible (Linux/macOS)
        subprocess.Popen(
            f"sleep {timeout}; ollama stop llama3",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
def is_model_installed(model_name):
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, encoding="utf-8", errors="ignore"
        )
        return model_name.lower() in result.stdout.lower()
    except Exception:
        return False

def pull_model_with_progress(model_name):
    print(f"\n📦 Model '{model_name}' not found. Downloading now...")
    show_download_progress()
    try:
        subprocess.run(["ollama", "pull", model_name], check=True)
    except subprocess.CalledProcessError:
        print("❌ Failed to pull the model. Please check the name or your internet connection.")
        sys.exit(1)

