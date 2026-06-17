import argparse
import os
import signal
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "artifacts" / "logs"
PID_FILES = {
    "ollama": LOG_DIR / "ollama.pid",
    "api": LOG_DIR / "api.pid",
    "streamlit": LOG_DIR / "streamlit.pid",
}
PORTS = {
    "ollama": 11434,
    "api": 8000,
    "streamlit": 8501,
}


def process_exists(pid: int) -> bool:
    if pid <= 0:
        return False

    if os.name == "nt":
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        return str(pid) in result.stdout

    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def stop_pid(pid: int, name: str) -> bool:
    try:
        if os.name == "nt":
            result = subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if result.returncode != 0:
                return False
        else:
            if not process_exists(pid):
                return False
            os.kill(pid, signal.SIGTERM)
        print(f"[stop] {name} pid={pid}")
        return True
    except OSError as exc:
        print(f"[warn] Cannot stop {name} pid={pid}: {exc}")
        return False


def read_pid(pid_file: Path) -> int | None:
    try:
        return int(pid_file.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError):
        return None


def pids_from_port(port: int) -> set[int]:
    if os.name != "nt":
        return set()

    result = subprocess.run(
        ["netstat", "-ano"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )

    pids: set[int] = set()
    marker = f":{port}"
    for line in result.stdout.splitlines():
        if marker not in line or "LISTENING" not in line:
            continue

        parts = line.split()
        if not parts:
            continue

        try:
            pids.add(int(parts[-1]))
        except ValueError:
            continue

    return pids


def stop_service(name: str, include_ports: bool = True) -> None:
    stopped = False

    pid = read_pid(PID_FILES[name])
    if pid is not None:
        stopped = stop_pid(pid, name) or stopped

    if include_ports:
        for port_pid in pids_from_port(PORTS[name]):
            stopped = stop_pid(port_pid, f"{name}:{PORTS[name]}") or stopped

    if stopped:
        try:
            PID_FILES[name].unlink()
        except FileNotFoundError:
            pass
    else:
        print(f"[skip] {name} is not running")


def main() -> int:
    parser = argparse.ArgumentParser(description="Stop EduRAG local services.")
    parser.add_argument(
        "--keep-ollama",
        action="store_true",
        help="Stop only API and Streamlit; leave Ollama running.",
    )
    args = parser.parse_args()

    stop_service("streamlit")
    stop_service("api")

    if not args.keep_ollama:
        stop_service("ollama")

    time.sleep(1)
    print("[done] Stop command completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
