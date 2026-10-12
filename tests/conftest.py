"""Pytest configuration and session-wide fixtures for Trading AI Assistant tests."""
import socket
import threading
import time
import pytest
import uvicorn
from trading_ai.server.app import app


def _is_server_listening(host="127.0.0.1", port=8000) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


@pytest.fixture(scope="session", autouse=True)
def session_fastapi_server():
    """Ensure FastAPI server is running on http://127.0.0.1:8000 for the entire test session."""
    if _is_server_listening("127.0.0.1", 8000):
        yield
        return

    config = uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="warning")
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()

    for _ in range(60):
        if _is_server_listening("127.0.0.1", 8000):
            break
        time.sleep(0.2)

    yield

    server.should_exit = True
