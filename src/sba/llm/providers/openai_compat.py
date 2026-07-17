"""Провайдер OpenAI-совместимого API: Ollama, LM Studio, vLLM, llama.cpp-server и т.п.

Ретраи с экспоненциальной паузой на сетевых ошибках и 5xx; таймаут чтения
большой — CPU-инференс медленный (docs/01-requirements.md §4.2).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import structlog

from sba.llm.gateway import ChatMessage, ChatResult, LLMError

log = structlog.get_logger(__name__)

RETRIES = 3
TIMEOUT = httpx.Timeout(connect=5.0, read=300.0, write=30.0, pool=10.0)


class OpenAICompatProvider:
    def __init__(self, base_url: str, api_key: str = "", client: httpx.AsyncClient | None = None):
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/", timeout=TIMEOUT, headers=headers
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    def _payload(
        self, model: str, messages: list[ChatMessage], temperature: float | None, stream: bool
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [m.model_dump() for m in messages],
            "stream": stream,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        return payload

    async def chat(
        self, model: str, messages: list[ChatMessage], temperature: float | None = None
    ) -> ChatResult:
        payload = self._payload(model, messages, temperature, stream=False)
        last_error: Exception | None = None
        for attempt in range(RETRIES):
            try:
                response = await self._client.post("chat/completions", json=payload)
            except httpx.TransportError as exc:
                last_error = exc
                await self._backoff(attempt, str(exc))
                continue
            if response.status_code >= 500:
                last_error = LLMError(f"HTTP {response.status_code}: {response.text[:200]}")
                await self._backoff(attempt, f"HTTP {response.status_code}")
                continue
            if response.status_code >= 400:
                raise LLMError(f"HTTP {response.status_code}: {response.text[:200]}")
            return self._parse_result(response.json())
        raise LLMError(f"модель недоступна после {RETRIES} попыток: {last_error}")

    async def stream(
        self, model: str, messages: list[ChatMessage], temperature: float | None = None
    ) -> AsyncIterator[str]:
        payload = self._payload(model, messages, temperature, stream=True)
        last_error: Exception | None = None
        for attempt in range(RETRIES):
            yielded_any = False
            try:
                async with self._client.stream(
                    "POST", "chat/completions", json=payload
                ) as response:
                    if response.status_code >= 400:
                        body = (await response.aread()).decode(errors="replace")[:200]
                        if response.status_code >= 500:
                            last_error = LLMError(f"HTTP {response.status_code}: {body}")
                            continue  # backoff ниже
                        raise LLMError(f"HTTP {response.status_code}: {body}")
                    async for line in response.aiter_lines():
                        delta = self._parse_sse_line(line)
                        if delta is None:
                            return
                        if delta:
                            yielded_any = True
                            yield delta
                    return
            except httpx.TransportError as exc:
                if yielded_any:
                    # обрыв посреди генерации — ретраить нельзя, будет дубль текста
                    raise LLMError(f"соединение оборвалось во время генерации: {exc}") from exc
                last_error = exc
            await self._backoff(attempt, str(last_error))
        raise LLMError(f"модель недоступна после {RETRIES} попыток: {last_error}")

    @staticmethod
    def _parse_sse_line(line: str) -> str | None:
        """Строка SSE → кусок текста; None означает конец потока ([DONE])."""
        if not line.startswith("data:"):
            return ""
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            return None
        try:
            chunk = json.loads(data)
            content = chunk["choices"][0].get("delta", {}).get("content")
            return str(content) if content else ""
        except (json.JSONDecodeError, LookupError) as exc:
            raise LLMError(f"некорректный SSE-чанк: {data[:200]}") from exc

    @staticmethod
    def _parse_result(data: dict[str, Any]) -> ChatResult:
        try:
            usage = data.get("usage") or {}
            return ChatResult(
                text=data["choices"][0]["message"].get("content") or "",
                model=str(data.get("model", "")),
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
            )
        except (LookupError, TypeError) as exc:
            raise LLMError(f"некорректный ответ модели: {str(data)[:200]}") from exc

    @staticmethod
    async def _backoff(attempt: int, reason: str) -> None:
        if attempt < RETRIES - 1:
            delay = 2.0**attempt
            log.warning("llm_retry", attempt=attempt + 1, delay=delay, reason=reason)
            await asyncio.sleep(delay)
