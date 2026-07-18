import json

import httpx
import pytest

from sba.llm.gateway import ChatMessage, LLMError
from sba.llm.providers.openai_compat import OpenAICompatProvider


def sse(*chunks: str) -> bytes:
    lines = [
        "data: " + json.dumps({"choices": [{"delta": {"content": c}}]}) for c in chunks
    ]
    lines.append("data: [DONE]")
    return ("\n\n".join(lines) + "\n\n").encode()


def make_provider(handler) -> OpenAICompatProvider:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://test/v1/"
    )
    return OpenAICompatProvider(base_url="http://test/v1", client=client)


MSGS = [ChatMessage(role="user", content="привет")]


async def test_chat_parses_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["model"] == "m1"
        assert body["stream"] is False
        return httpx.Response(
            200,
            json={
                "model": "m1",
                "choices": [{"message": {"content": "здравствуйте"}}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3},
            },
        )

    result = await make_provider(handler).chat("m1", MSGS, temperature=0.5)
    assert result.text == "здравствуйте"
    assert result.prompt_tokens == 7


async def test_stream_yields_deltas() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(200, content=sse("при", "вет"))

    provider = make_provider(handler)
    parts = [e.text async for e in provider.stream("m1", MSGS)]
    assert "".join(parts) == "привет"


async def test_stream_assembles_tool_calls_across_chunks() -> None:
    chunk1 = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "function": {"name": "list_files", "arguments": '{"pa'},
                        }
                    ]
                }
            }
        ]
    }
    chunk2 = {
        "choices": [
            {"delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'th": "."}'}}]}}
        ]
    }
    body = (
        f"data: {json.dumps(chunk1)}\n\n"
        f"data: {json.dumps(chunk2)}\n\n"
        "data: [DONE]\n\n"
    ).encode()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    events = [e async for e in make_provider(handler).stream("m1", MSGS, tools=[{}])]
    final = events[-1]
    assert final.tool_calls is not None
    assert final.tool_calls[0].id == "call_1"
    assert final.tool_calls[0].name == "list_files"
    assert final.tool_calls[0].arguments == {"path": "."}


async def test_retries_on_5xx_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(500, text="boom")
        return httpx.Response(
            200, json={"model": "m", "choices": [{"message": {"content": "ок"}}]}
        )

    async def no_sleep(_: float) -> None: ...

    monkeypatch.setattr("asyncio.sleep", no_sleep)
    result = await make_provider(handler).chat("m", MSGS)
    assert result.text == "ок"
    assert calls["n"] == 2


async def test_4xx_fails_without_retry() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(404, text="model not found")

    with pytest.raises(LLMError, match="404"):
        await make_provider(handler).chat("нет-такой", MSGS)
    assert calls["n"] == 1


async def test_read_timeout_fails_fast_without_retry() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ReadTimeout("модель молчит")

    with pytest.raises(LLMError, match="за отведённое время"):
        await make_provider(handler).chat("m", MSGS)
    assert calls["n"] == 1  # зависший запрос не повторяется втрое


async def test_connection_error_exhausts_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    async def no_sleep(_: float) -> None: ...

    monkeypatch.setattr("asyncio.sleep", no_sleep)
    with pytest.raises(LLMError, match="недоступна"):
        await make_provider(handler).chat("m", MSGS)
