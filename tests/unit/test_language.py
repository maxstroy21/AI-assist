from collections.abc import AsyncIterator

from sba.core.agent.language import filter_cjk_stream, strip_cjk


def test_pure_russian_unchanged() -> None:
    assert strip_cjk("Привет, как дела?") == "Привет, как дела?"


def test_removes_chinese_tail() -> None:
    text = "Погоду можно узнать в приложении. 您可以通过各种在线气象应用查看天气。"
    assert strip_cjk(text) == "Погоду можно узнать в приложении. "


def test_removes_inline_cjk_and_fullwidth_punctuation() -> None:
    # реальный случай из живого чата: «в实时翻译：»
    assert strip_cjk("в实时翻译：") == "в"


def test_keeps_latin_terms() -> None:
    assert strip_cjk("Использую Python и Ollama локально") == "Использую Python и Ollama локально"


def test_removes_japanese_and_korean() -> None:
    assert strip_cjk("текст こんにちは 안녕하세요 конец") == "текст конец"


def test_collapses_orphaned_spaces() -> None:
    assert strip_cjk("слово 中文 слово") == "слово слово"


async def test_stream_filter_drops_emptied_chunks() -> None:
    async def source() -> AsyncIterator[str]:
        for part in ("Пого", "да ", "хорошая", "。", "您可以"):
            yield part

    out = [chunk async for chunk in filter_cjk_stream(source())]
    assert "".join(out) == "Погода хорошая"
    assert "。" not in out and "您可以" not in out
