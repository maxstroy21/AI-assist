"""Провайдер OpenAI-совместимого API: Ollama, LM Studio, vLLM, llama.cpp-server и т.п.

Ретраи с экспоненциальной паузой на сетевых ошибках и 5xx; таймаут чтения
большой — CPU-инференс медленный (docs/01-requirements.md §4.2).
Поддерживает нативный tool calling, в т.ч. потоковый (tool_calls-дельты
склеиваются по index до полного вызова).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import structlog

from sba.llm.gateway import ChatMessage, ChatResult, LLMError, StreamEvent, ToolCall, ToolSchema

log = structlog.get_logger(__name__)

RETRIES = 3
# read=300: холодная загрузка модели + обработка промпта на CPU занимает минуты;
# повторов при таймауте нет, так что ждём один раз, с heartbeat в интерфейсе
TIMEOUT = httpx.Timeout(connect=5.0, read=300.0, write=30.0, pool=10.0)
# потолок автопаузы перед повтором: облачный лимит (429) может просить ждать
# долго — столько в интерактивном чате не висим, честнее вернуть подсказку
MAX_RETRY_DELAY = 20.0

_DONE = object()  # сентинел конца SSE-потока


class OpenAICompatProvider:
    def __init__(self, base_url: str, api_key: str = "", client: httpx.AsyncClient | None = None):
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/", timeout=TIMEOUT, headers=headers
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    # ── формирование запроса ─────────────────────────────────────────────────

    @staticmethod
    def _serialize_message(msg: ChatMessage) -> dict[str, Any]:
        data: dict[str, Any] = {"role": msg.role, "content": msg.content}
        if msg.tool_calls:
            data["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments, ensure_ascii=False),
                    },
                }
                for call in msg.tool_calls
            ]
        if msg.tool_call_id is not None:
            data["tool_call_id"] = msg.tool_call_id
        return data

    def _payload(
        self,
        model: str,
        messages: list[ChatMessage],
        temperature: float | None,
        stream: bool,
        tools: list[ToolSchema] | None = None,
        tool_choice: str | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [self._serialize_message(m) for m in messages],
            "stream": stream,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if tools:
            payload["tools"] = tools
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice
        return payload

    # ── не-потоковый вызов ───────────────────────────────────────────────────

    async def chat(
        self,
        model: str,
        messages: list[ChatMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        payload = self._payload(model, messages, temperature, stream=False, max_tokens=max_tokens)
        last_error: Exception | None = None
        for attempt in range(RETRIES):
            try:
                response = await self._client.post("chat/completions", json=payload)
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                last_error = exc  # соединение не установилось — повтор осмыслен
                await self._backoff(attempt, str(exc))
                continue
            except httpx.TransportError as exc:
                # запрос уже выполнялся (например, таймаут чтения) — повтор
                # почти наверняка зависнет так же и умножит ожидание
                raise LLMError(
                    f"модель не ответила за отведённое время: {exc!r}"
                ) from exc
            # 429 (лимит запросов облака) и 5xx — временные: ждём и повторяем.
            # У 429 облако присылает Retry-After — уважаем его вместо экспоненты
            if response.status_code == 429 or response.status_code >= 500:
                last_error = LLMError(f"HTTP {response.status_code}: {response.text[:200]}")
                await self._backoff(
                    attempt, f"HTTP {response.status_code}", self._retry_after(response)
                )
                continue
            if response.status_code >= 400:
                raise LLMError(f"HTTP {response.status_code}: {response.text[:200]}")
            return self._parse_result(response.json())
        raise LLMError(f"модель недоступна после {RETRIES} попыток: {last_error}")

    # ── эмбеддинги ───────────────────────────────────────────────────────────

    async def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        """POST /embeddings (OpenAI-совместимый; Ollama поддерживает).

        Порядок векторов гарантируется полем index ответа.
        """
        if not texts:
            return []
        payload = {"model": model, "input": texts}
        last_error: Exception | None = None
        for attempt in range(RETRIES):
            try:
                response = await self._client.post("embeddings", json=payload)
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                last_error = exc
                await self._backoff(attempt, str(exc))
                continue
            except httpx.TransportError as exc:
                raise LLMError(
                    f"эмбеддинг-модель не ответила за отведённое время: {exc!r}"
                ) from exc
            if response.status_code >= 500:
                last_error = LLMError(f"HTTP {response.status_code}: {response.text[:200]}")
                await self._backoff(attempt, f"HTTP {response.status_code}")
                continue
            if response.status_code >= 400:
                raise LLMError(f"HTTP {response.status_code}: {response.text[:200]}")
            return self._parse_embeddings(response.json(), expected=len(texts))
        raise LLMError(f"эмбеддинг-модель недоступна после {RETRIES} попыток: {last_error}")

    @staticmethod
    def _parse_embeddings(data: dict[str, Any], expected: int) -> list[list[float]]:
        try:
            items = sorted(data["data"], key=lambda item: int(item["index"]))
            vectors = [[float(x) for x in item["embedding"]] for item in items]
        except (LookupError, TypeError, ValueError) as exc:
            raise LLMError(f"некорректный ответ эмбеддинга: {str(data)[:200]}") from exc
        if len(vectors) != expected:
            raise LLMError(
                f"эмбеддинг вернул {len(vectors)} векторов вместо {expected}"
            )
        return vectors

    # ── потоковый вызов ──────────────────────────────────────────────────────

    async def stream(
        self,
        model: str,
        messages: list[ChatMessage],
        temperature: float | None = None,
        tools: list[ToolSchema] | None = None,
        tool_choice: str | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamEvent]:
        payload = self._payload(
            model, messages, temperature, stream=True, tools=tools,
            tool_choice=tool_choice, max_tokens=max_tokens,
        )
        last_error: Exception | None = None
        for attempt in range(RETRIES):
            partial_calls: dict[int, dict[str, str]] = {}
            retry_after: float | None = None
            try:
                async with self._client.stream(
                    "POST", "chat/completions", json=payload
                ) as response:
                    if response.status_code >= 400:
                        body = (await response.aread()).decode(errors="replace")[:200]
                        # 429/5xx — временные: запоминаем ошибку и уходим на
                        # backoff внизу цикла (не continue: он пропустил бы паузу)
                        if response.status_code == 429 or response.status_code >= 500:
                            last_error = LLMError(f"HTTP {response.status_code}: {body}")
                            retry_after = self._retry_after(response)
                        else:
                            raise LLMError(f"HTTP {response.status_code}: {body}")
                    else:
                        async for line in response.aiter_lines():
                            chunk = self._parse_sse_line(line)
                            if chunk is None:
                                continue
                            if chunk is _DONE:
                                break
                            text = self._collect_delta(chunk, partial_calls)  # type: ignore[arg-type]
                            if text:
                                yield StreamEvent(text=text)
                        calls = self._finalize_calls(partial_calls)
                        if calls:
                            yield StreamEvent(tool_calls=calls)
                        return
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                last_error = exc  # не достучались — повтор осмыслен
            except httpx.TransportError as exc:
                # таймаут чтения или обрыв уже идущего запроса: повтор умножит
                # ожидание втрое и может задублировать текст
                raise LLMError(
                    f"модель не ответила за отведённое время: {exc!r}"
                ) from exc
            await self._backoff(attempt, str(last_error), retry_after)
        raise LLMError(f"модель недоступна после {RETRIES} попыток: {last_error}")

    # ── разбор ответа ────────────────────────────────────────────────────────

    @staticmethod
    def _parse_sse_line(line: str) -> dict[str, Any] | object | None:
        if not line.startswith("data:"):
            return None
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            return _DONE
        try:
            parsed: dict[str, Any] = json.loads(data)
            return parsed
        except json.JSONDecodeError as exc:
            raise LLMError(f"некорректный SSE-чанк: {data[:200]}") from exc

    @staticmethod
    def _collect_delta(chunk: dict[str, Any], partial_calls: dict[int, dict[str, str]]) -> str:
        """Извлекает текст из чанка и накапливает фрагменты tool_calls."""
        try:
            delta = chunk["choices"][0].get("delta", {})
        except (LookupError, TypeError) as exc:
            raise LLMError(f"некорректный чанк: {str(chunk)[:200]}") from exc
        for fragment in delta.get("tool_calls") or []:
            index = int(fragment.get("index", 0))
            slot = partial_calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
            if fragment.get("id"):
                slot["id"] = fragment["id"]
            function = fragment.get("function") or {}
            if function.get("name"):
                slot["name"] += function["name"]
            if function.get("arguments"):
                slot["arguments"] += function["arguments"]
        content = delta.get("content")
        return str(content) if content else ""

    @staticmethod
    def _finalize_calls(partial_calls: dict[int, dict[str, str]]) -> list[ToolCall]:
        calls: list[ToolCall] = []
        for index in sorted(partial_calls):
            slot = partial_calls[index]
            raw_args = slot["arguments"].strip() or "{}"
            try:
                arguments = json.loads(raw_args)
            except json.JSONDecodeError as exc:
                raise LLMError(
                    f"инструмент {slot['name']}: аргументы не являются JSON: {raw_args[:200]}"
                ) from exc
            if not isinstance(arguments, dict):
                raise LLMError(f"инструмент {slot['name']}: аргументы не объект")
            calls.append(
                ToolCall(
                    id=slot["id"] or f"call_{index}",
                    name=slot["name"],
                    arguments=arguments,
                )
            )
        return calls

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
    def _retry_after(response: httpx.Response) -> float | None:
        """Сколько ждать по заголовку Retry-After (облако присылает его на 429).
        Возвращает секунды или None, если заголовка нет/он не число."""
        header = response.headers.get("retry-after")
        if not header:
            return None
        try:
            return float(header)
        except ValueError:
            return None  # HTTP-дата в Retry-After нам не встречается — игнорируем

    @staticmethod
    async def _backoff(attempt: int, reason: str, retry_after: float | None = None) -> None:
        if attempt < RETRIES - 1:
            # если облако назвало паузу (429) — уважаем её, но не висим дольше
            # потолка; иначе экспонента 1, 2, 4…
            base = retry_after if retry_after is not None else 2.0**attempt
            delay = min(base, MAX_RETRY_DELAY)
            log.warning("llm_retry", attempt=attempt + 1, delay=delay, reason=reason)
            await asyncio.sleep(delay)
