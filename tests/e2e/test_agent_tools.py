"""E2E agent loop: модель «решает» позвать get_current_time, приложение
исполняет инструмент и возвращает результат модели для финального ответа."""

import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]

REQUESTS: list[dict] = []

TOOL_CALL_CHUNK = {
    "choices": [
        {
            "delta": {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_1",
                        "function": {"name": "get_current_time", "arguments": "{}"},
                    }
                ]
            }
        }
    ]
}


class AgentScriptHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        REQUESTS.append(body)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        if len(REQUESTS) == 1:
            self.wfile.write(f"data: {json.dumps(TOOL_CALL_CHUNK)}\n\n".encode())
        else:
            chunk = json.dumps({"choices": [{"delta": {"content": "Время получено!"}}]})
            self.wfile.write(f"data: {chunk}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *args: object) -> None: ...


def test_agent_executes_tool_and_answers(tmp_path: Path) -> None:
    REQUESTS.clear()
    server = HTTPServer(("127.0.0.1", 0), AgentScriptHandler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
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
            input="который час?\n/quit\n",
            capture_output=True,
            text=True,
            timeout=60,
            cwd=REPO_ROOT,
        )
        assert proc.returncode == 0, proc.stderr
        assert "Время получено!" in proc.stdout

        # модели передали схемы инструментов
        assert any(t["function"]["name"] == "get_current_time" for t in REQUESTS[0]["tools"])
        # во втором запросе есть результат инструмента (роль tool с датой)
        tool_messages = [m for m in REQUESTS[1]["messages"] if m["role"] == "tool"]
        assert tool_messages and "20" in tool_messages[0]["content"]  # год из даты
    finally:
        server.shutdown()
