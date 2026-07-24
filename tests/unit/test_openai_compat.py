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


async def test_retries_on_429_honoring_retry_after(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}
    slept: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            # облако просит подождать 3 секунды — уважаем заголовок
            return httpx.Response(429, text="rate limit", headers={"retry-after": "3"})
        return httpx.Response(
            200, json={"model": "m", "choices": [{"message": {"content": "ок"}}]}
        )

    async def capture_sleep(delay: float) -> None:
        slept.append(delay)

    monkeypatch.setattr("asyncio.sleep", capture_sleep)
    result = await make_provider(handler).chat("m", MSGS)
    assert result.text == "ок"
    assert calls["n"] == 2
    assert slept == [3.0]  # пауза взята из Retry-After, а не из экспоненты


async def test_429_retry_after_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    slept: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        # облако просит ждать неприемлемо долго — не висим дольше потолка
        return httpx.Response(429, text="slow down", headers={"retry-after": "600"})

    async def capture_sleep(delay: float) -> None:
        slept.append(delay)

    monkeypatch.setattr("asyncio.sleep", capture_sleep)
    with pytest.raises(LLMError, match="429"):
        await make_provider(handler).chat("m", MSGS)
    assert slept and all(d <= 20.0 for d in slept)  # каждая пауза под потолком


async def test_stream_retries_on_429(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, text="rate limit", headers={"retry-after": "1"})
        return httpx.Response(200, content=sse("готово"))

    async def no_sleep(_: float) -> None: ...

    monkeypatch.setattr("asyncio.sleep", no_sleep)
    parts = [e.text async for e in make_provider(handler).stream("m", MSGS)]
    assert "".join(parts) == "готово"
    assert calls["n"] == 2


def _tool_call_stream(name: str, arguments: str) -> bytes:
    chunk = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {"index": 0, "id": "c1", "function": {"name": name, "arguments": arguments}}
                    ]
                }
            }
        ]
    }
    return f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode()


async def test_tool_call_null_arguments_becomes_empty_dict() -> None:
    # Groq llama-3.3 на инструменте без параметров шлёт arguments="null" —
    # это не должно ронять весь ход (баг живой проверки 2026-07-24)
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_tool_call_stream("get_current_time", "null"))

    events = [e async for e in make_provider(handler).stream("m", MSGS, tools=[{}])]
    call = events[-1].tool_calls[0]
    assert call.name == "get_current_time"
    assert call.arguments == {}


async def test_tool_call_double_encoded_arguments_recovered() -> None:
    # аргументы как JSON-строка с объектом внутри — разворачиваем
    double_encoded = '"{\\"name_pattern\\": \\"a\\"}"'

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_tool_call_stream("find_files", double_encoded))

    events = [e async for e in make_provider(handler).stream("m", MSGS, tools=[{}])]
    assert events[-1].tool_calls[0].arguments == {"name_pattern": "a"}


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


async def test_embed_parses_vectors_in_order() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert request.url.path.endswith("/embeddings")
        assert body["input"] == ["первый", "второй"]
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.3, 0.4]},
                    {"index": 0, "embedding": [0.1, 0.2]},
                ]
            },
        )

    vectors = await make_provider(handler).embed("emb", ["первый", "второй"])
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]  # порядок восстановлен по index


async def test_embed_empty_input_short_circuits() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("не должно быть запроса")

    assert await make_provider(handler).embed("emb", []) == []


async def test_embed_wrong_count_is_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.1]}]})

    with pytest.raises(LLMError, match="вместо 2"):
        await make_provider(handler).embed("emb", ["а", "б"])
