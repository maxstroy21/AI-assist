"""E2E: конфиг → LLM Gateway → оркестратор → Router → REPL против
фейкового OpenAI-совместимого сервера (тот же путь, что с Ollama)."""

import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]


class FakeOpenAIHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        assert body["messages"][0]["role"] == "system"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for content in ("Привет, ", "я локальная модель"):
            chunk = json.dumps({"choices": [{"delta": {"content": content}}]})
            self.wfile.write(f"data: {chunk}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *args: object) -> None: ...


def test_dialog_via_fake_openai_server(tmp_path: Path) -> None:
    server = HTTPServer(("127.0.0.1", 0), FakeOpenAIHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "default.yaml").write_text(
            f"app:\n  data_dir: {tmp_path / 'data'}\n", encoding="utf-8"
        )
        (config_dir / "models.yaml").write_text(
            "runtimes:\n"
            f'  fake: {{kind: openai_compatible, base_url: "http://127.0.0.1:{port}/v1"}}\n'
            "roles:\n"
            '  chat: {runtime: fake, model: "test-model"}\n',
            encoding="utf-8",
        )
        proc = subprocess.run(
            [sys.executable, "-m", "sba", "--config-dir", str(config_dir)],
            input="привет\n/quit\n",
            capture_output=True,
            text=True,
            timeout=60,
            cwd=REPO_ROOT,
        )
        assert proc.returncode == 0, proc.stderr
        assert "Привет, я локальная модель" in proc.stdout
    finally:
        server.shutdown()
