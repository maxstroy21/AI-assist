"""E2E напоминаний: модель «решает» создать напоминание, приложение исполняет
инструмент, /reminders показывает результат детерминированно (мимо LLM)."""

import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]

REQUESTS: list[dict] = []

CREATE_REMINDER_CHUNK = {
    "choices": [
        {
            "delta": {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_1",
                        "function": {
                            "name": "create_reminder",
                            "arguments": json.dumps(
                                {"text": "позвонить Ивану", "when": "через 2 часа"},
                                ensure_ascii=False,
                            ),
                        },
                    }
                ]
            }
        }
    ]
}


class ReminderScriptHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        REQUESTS.append(json.loads(self.rfile.read(length)))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        if len(REQUESTS) == 1:
            self.wfile.write(f"data: {json.dumps(CREATE_REMINDER_CHUNK)}\n\n".encode())
        else:
            chunk = json.dumps(
                {"choices": [{"delta": {"content": "Напоминание создано!"}}]}
            )
            self.wfile.write(f"data: {chunk}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *args: object) -> None: ...


def test_create_reminder_via_agent_and_list_via_command(tmp_path: Path) -> None:
    REQUESTS.clear()
    server = HTTPServer(("127.0.0.1", 0), ReminderScriptHandler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "default.yaml").write_text(
            f"app:\n  data_dir: {tmp_path / 'data'}\n"
            "llm:\n  keep_warm_minutes: 0\n"
            "modules:\n  rag:\n    enabled: false\n",
            encoding="utf-8",
        )
        (config_dir / "models.yaml").write_text(
            "runtimes:\n"
            f'  fake: {{kind: openai_compatible, base_url: "http://127.0.0.1:{port}/v1"}}\n'
            "roles:\n"
            '  chat: {runtime: fake, model: "test-model"}\n'
            '  extraction: {runtime: fake, model: "test-model"}\n',
            encoding="utf-8",
        )
        proc = subprocess.run(
            [sys.executable, "-m", "sba", "--config-dir", str(config_dir)],
            input="напомни позвонить Ивану через 2 часа\n/reminders\n/quit\n",
            capture_output=True,
            text=True,
            timeout=60,
            cwd=REPO_ROOT,
        )
        assert proc.returncode == 0, proc.stderr
        # видимый маркер реального вызова инструмента
        assert "🔧 create_reminder" in proc.stdout
        assert "Напоминание создано!" in proc.stdout
        # /reminders — детерминированный список, мимо LLM
        assert "Предстоящие напоминания (1):" in proc.stdout
        assert "позвонить Ивану" in proc.stdout

        # модель получила схемы инструментов напоминаний
        tool_names = {t["function"]["name"] for t in REQUESTS[0]["tools"]}
        assert {
            "create_reminder", "list_reminders", "snooze_reminder", "cancel_reminder"
        } <= tool_names
        # результат инструмента вернулся модели и честно описывает механику
        tool_messages = [m for m in REQUESTS[1]["messages"] if m["role"] == "tool"]
        assert tool_messages and "Создал напоминание" in tool_messages[0]["content"]
        # «напомни…» — тематический вопрос: приложение требовало инструмент
        assert REQUESTS[0].get("tool_choice") == "required"
        # extraction-модель не понадобилась: «через 2 часа» разобран детерминированно
        assert len(REQUESTS) == 2
    finally:
        server.shutdown()
