import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "artifacts" / "logs"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"

OLLAMA_URL = "http://127.0.0.1:11434/api/tags"
API_URL = "http://127.0.0.1:8000/health"
UI_URL = "http://127.0.0.1:8501"


def http_ready(url: str, timeout: float = 1.0) -> bool:
    try:
        with urlopen(url, timeout=timeout) as response:
            return 200 <= response.status < 500
    except (OSError, URLError):
        return False


def wait_for(name: str, url: str, timeout_seconds: int) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if http_ready(url):
            print(f"[ok] {name} is ready: {url}")
            return True
        time.sleep(1)

    print(f"[warn] {name} did not become ready in {timeout_seconds}s: {url}")
    return False


def popen_kwargs(log_name: str) -> dict:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stdout = open(LOG_DIR / f"{log_name}.out.log", "a", encoding="utf-8")
    stderr = open(LOG_DIR / f"{log_name}.err.log", "a", encoding="utf-8")

    kwargs = {
        "cwd": ROOT,
        "stdout": stdout,
        "stderr": stderr,
        "stdin": subprocess.DEVNULL,
    }

    if os.name == "nt":
        kwargs["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NO_WINDOW
        )
    else:
        kwargs["start_new_session"] = True

    return kwargs


def start_process(name: str, command: list[str], log_name: str) -> subprocess.Popen | None:
    try:
        process = subprocess.Popen(command, **popen_kwargs(log_name))
    except FileNotFoundError:
        print(f"[error] Cannot start {name}: executable not found: {command[0]}")
        return None

    pid_file = LOG_DIR / f"{log_name}.pid"
    pid_file.write_text(str(process.pid), encoding="utf-8")
    print(f"[start] {name} pid={process.pid}")
    return process


def python_executable() -> str:
    if PYTHON.exists():
        return str(PYTHON)
    return sys.executable


def start_ollama(wait_seconds: int) -> None:
    if http_ready(OLLAMA_URL):
        print("[skip] Ollama is already running")
        return

    start_process("Ollama", ["ollama", "serve"], "ollama")
    wait_for("Ollama", OLLAMA_URL, wait_seconds)


def start_api(wait_seconds: int) -> None:
    if http_ready(API_URL):
        print("[skip] FastAPI is already running")
        return

    start_process(
        "FastAPI",
        [
            python_executable(),
            "-m",
            "uvicorn",
            "src.service.api:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ],
        "api",
    )
    wait_for("FastAPI", API_URL, wait_seconds)


def start_streamlit(wait_seconds: int) -> None:
    if http_ready(UI_URL):
        print("[skip] Streamlit is already running")
        return

    start_process(
        "Streamlit",
        [
            python_executable(),
            "-m",
            "streamlit",
            "run",
            "src/service/app.py",
            "--server.address",
            "127.0.0.1",
            "--server.port",
            "8501",
            "--server.headless",
            "true",
        ],
        "streamlit",
    )
    wait_for("Streamlit", UI_URL, wait_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description="Start Ollama, FastAPI and Streamlit.")
    parser.add_argument("--skip-ollama", action="store_true", help="Do not start ollama serve.")
    parser.add_argument("--wait", type=int, default=60, help="Seconds to wait for each service.")
    args = parser.parse_args()

    if not args.skip_ollama:
        start_ollama(args.wait)
    start_api(args.wait)
    start_streamlit(args.wait)

    print("")
    print(f"API: {API_URL}")
    print(f"UI:  {UI_URL}")
    print(f"Logs: {LOG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
