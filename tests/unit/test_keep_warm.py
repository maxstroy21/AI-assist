import asyncio
import json

import httpx
import pytest

from sba.app import keep_warm_loop
from sba.llm.config import ModelsConfig
from sba.llm.providers.openai_compat import OpenAICompatProvider
from sba.llm.service import ModelGateway

MODELS = ModelsConfig.model_validate(
    {
        "runtimes": {"fake": {"kind": "openai_compatible", "base_url": "http://t/v1"}},
        "roles": {"chat": {"runtime": "fake", "model": "m"}},
    }
)


def make_gateway(handler) -> ModelGateway:
    gateway = ModelGateway(MODELS)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://t/v1/")
    gateway._providers["fake"] = OpenAICompatProvider("http://t/v1", client=client)
    return gateway


async def test_warmup_sends_one_token_ping_and_reports_success() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(
            200, json={"model": "m", "choices": [{"message": {"content": "."}}]}
        )

    assert await make_gateway(handler).warmup() is True
    assert seen[0]["max_tokens"] == 1
    assert seen[0]["messages"] == [{"role": "user", "content": "ping"}]


async def test_warmup_returns_false_on_error_without_raising() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("нет Ollama")

    # не бросает, но честно сообщает, что модель не поднялась
    assert await make_gateway(handler).warmup() is False


def make_models(runtimes: dict, chat_runtime: str) -> ModelsConfig:
    return ModelsConfig.model_validate(
        {"runtimes": runtimes, "roles": {"chat": {"runtime": chat_runtime, "model": "m"}}}
    )


def test_chat_runtime_local_for_localhost_urls() -> None:
    for url in ("http://localhost:11434/v1", "http://127.0.0.1:1234/v1"):
        models = make_models(
            {"ollama": {"kind": "openai_compatible", "base_url": url}}, "ollama"
        )
        assert ModelGateway(models).chat_runtime_is_local() is True


def test_chat_runtime_not_local_for_cloud() -> None:
    # облачный OpenAI-совместимый (Groq) и anthropic: прогрев не нужен,
    # keep-warm жёг бы бесплатный дневной лимит запросов
    groq = make_models(
        {"groq": {"kind": "openai_compatible", "base_url": "https://api.groq.com/openai/v1"}},
        "groq",
    )
    assert ModelGateway(groq).chat_runtime_is_local() is False
    cloud = make_models({"cloud": {"kind": "anthropic", "api_key": "sk-test"}}, "cloud")
    assert ModelGateway(cloud).chat_runtime_is_local() is False


async def test_keep_warm_loop_pings_repeatedly() -> None:
    calls = {"n": 0}

    class StubGateway:
        async def warmup(self) -> bool:
            calls["n"] += 1
            return True

    task = asyncio.create_task(keep_warm_loop(StubGateway(), interval_seconds=0.01))  # type: ignore[arg-type]
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert calls["n"] >= 2
