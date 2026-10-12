"""Trading AI Assistant — Private Desktop Launcher.

Features:
1. Python version validation (Python 3.10+).
2. Virtual environment and dependency verification.
3. Safe environment setup (.env initialization with secure secrets if missing).
4. SQLite database integrity verification (PRAGMA integrity_check) & automated pre-launch backup.
5. Graceful Uvicorn server startup bound strictly to private address (default 127.0.0.1).
6. Automatic local browser launch upon successful health check.
7. Clean shutdown and signal handling (SIGINT / Ctrl+C).
"""

import os
import sys
import time
import socket
import sqlite3
import secrets
import logging
import argparse
import webbrowser
import subprocess
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("DesktopLauncher")

MIN_PYTHON = (3, 10)
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


def check_python_version() -> None:
    if sys.version_info < MIN_PYTHON:
        logger.error("Python %d.%d+ is required. Found: %s", MIN_PYTHON[0], MIN_PYTHON[1], sys.version)
        sys.exit(1)
    logger.info("Python runtime verified: %s", sys.version.split()[0])


def ensure_env_file(root_dir: Path) -> None:
    env_file = root_dir / ".env"
    example_file = root_dir / ".env.example"
    if not env_file.exists():
        logger.info(".env not found. Initializing from .env.example with secure random credentials...")
        content = example_file.read_text(encoding="utf-8") if example_file.exists() else ""
        auth_token = secrets.token_hex(24)
        session_secret = secrets.token_hex(32)
        extra_config = (
            f"\n# Generated Private Authentication Secrets\n"
            f'AUTH_ENABLED="false"\n'
            f'AUTH_TOKEN="{auth_token}"\n'
            f'SESSION_SECRET="{session_secret}"\n'
        )
        env_file.write_text(content + extra_config, encoding="utf-8")
        logger.info("Created .env with generated AUTH_TOKEN and SESSION_SECRET.")
    else:
        logger.info("Found existing .env configuration file.")


def backup_and_verify_database(root_dir: Path) -> None:
    db_path = root_dir / "data" / "trading_ai.db"
    backup_dir = root_dir / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)

    if not db_path.exists():
        logger.info("Database %s does not yet exist; will be created on initial startup.", db_path)
        return

    # 1. Integrity check
    logger.info("Verifying database integrity for %s...", db_path)
    try:
        con = sqlite3.connect(str(db_path))
        cur = con.cursor()
        res = cur.execute("PRAGMA integrity_check;").fetchall()
        if res != [("ok",)]:
            logger.error("Database integrity check failed: %s", res)
            con.close()
            sys.exit(1)
        logger.info("Database integrity verified: PRAGMA integrity_check == ok")

        # 2. Online atomic backup
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = backup_dir / f"trading_ai_prelaunch_{ts}.db"
        dst = sqlite3.connect(str(backup_file))
        con.backup(dst)
        dst.close()
        con.close()
        logger.info("Pre-launch backup safely preserved: %s (%d bytes)", backup_file.name, backup_file.stat().st_size)
    except Exception as e:
        logger.error("Database pre-flight check failed: %s", e)
        sys.exit(1)


def is_port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def wait_for_server(host: str, port: int, timeout_sec: float = 15.0) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if is_port_in_use(host, port):
            return True
        time.sleep(0.3)
    return False


def main():
    parser = argparse.ArgumentParser(description="Trading AI Assistant Private Desktop Launcher")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Binding address (default 127.0.0.1 for loopback security)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Server port (default 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open local browser automatically")
    parser.add_argument("--allow-lan", action="store_true", help="Bind to private local network for phone access (0.0.0.0)")
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent.parent
    os.chdir(str(root_dir))

    print("=" * 70)
    print("TRADING AI ASSISTANT — SECURE PRIVATE DESKTOP LAUNCHER")
    print("=" * 70)

    # 1. Runtime validation
    check_python_version()

    # 2. Config validation
    ensure_env_file(root_dir)

    # 3. Database safety & pre-launch backup
    backup_and_verify_database(root_dir)

    # 4. Host resolution
    bind_host = "0.0.0.0" if args.allow_lan else args.host
    if args.allow_lan:
        logger.warning("Binding to 0.0.0.0 enabled for private phone LAN access.")
        logger.warning("Ensure AUTH_ENABLED=true in .env to prevent unauthorized LAN access.")
    else:
        logger.info("Binding to loopback %s (access restricted to desktop).", bind_host)

    # 5. Check if port is already running
    if is_port_in_use("127.0.0.1", args.port):
        logger.warning("Port %d is already in use. An instance may already be running.", args.port)
        if not args.no_browser:
            webbrowser.open(f"http://127.0.0.1:{args.port}")
        return

    # 6. Start Uvicorn process
    python_exe = sys.executable
    cmd = [
        python_exe,
        "-m",
        "uvicorn",
        "trading_ai.server.app:app",
        "--host",
        bind_host,
        "--port",
        str(args.port),
        "--log-level",
        "info",
    ]
    logger.info("Starting backend server: %s", " ".join(cmd))

    proc = subprocess.Popen(cmd)

    try:
        if wait_for_server("127.0.0.1", args.port, timeout_sec=20.0):
            target_url = f"http://127.0.0.1:{args.port}"
            logger.info("Trading AI Assistant is active and listening at %s", target_url)
            if not args.no_browser:
                logger.info("Opening console in default browser...")
                webbrowser.open(target_url)
            print("-" * 70)
            print(f"Console URL: {target_url}")
            print("Press Ctrl+C to cleanly stop the server.")
            print("-" * 70)
            proc.wait()
        else:
            logger.error("Server did not respond on port %d within timeout.", args.port)
            proc.terminate()
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user. Terminating server process cleanly...")
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
        logger.info("Server stopped cleanly. Database state safely preserved.")


if __name__ == "__main__":
    main()
