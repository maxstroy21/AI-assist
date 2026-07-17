"""Языковой барьер: удаление CJK-символов из ответа модели.

qwen2.5 (и другие мультиязычные модели) на CPU-тире склонны дописывать
фрагменты на китайском/японском/корейском в русский ответ. Промпт и
температура снижают это, но не гарантируют. Этот фильтр — жёсткая гарантия:
иероглифы вырезаются из потока до отправки пользователю, с любой моделью.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator

# CJK/японские/корейские блоки + иероглифическая и полноширинная пунктуация
_CJK = re.compile(
    "["
    "　-〿"  # CJK-пунктуация (。、：！？…)
    "぀-ヿ"  # хирагана, катакана
    "ㇰ-ㇿ"  # катакана фонетические
    "㐀-䶿"  # CJK Extension A
    "一-鿿"  # CJK Unified Ideographs
    "ꀀ-꓏"  # и
    "가-힯"  # хангыль
    "豈-﫿"  # CJK Compatibility Ideographs
    "＀-￯"  # полноширинные формы
    "]+"
)


def strip_cjk(text: str) -> str:
    """Убрать CJK-символы; схлопнуть пробелы, осиротевшие после удаления."""
    cleaned = _CJK.sub("", text)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned


async def filter_cjk_stream(deltas: AsyncIterator[str]) -> AsyncIterator[str]:
    """Пропустить поток кусков текста через strip_cjk, отбрасывая опустевшие."""
    async for delta in deltas:
        cleaned = strip_cjk(delta)
        if cleaned:
            yield cleaned
