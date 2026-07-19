from sba.modules.indexer.chunker import make_chunks
from sba.modules.indexer.extractors import Block


def test_small_blocks_pass_through_with_locators() -> None:
    blocks = [
        Block(text="Первый абзац о проекте.", locator="стр. 1"),
        Block(text="Второй абзац о бюджете.", locator="стр. 2"),
    ]
    chunks = make_chunks(blocks, chunk_chars=100, overlap_chars=10)
    assert [c.locator for c in chunks] == ["стр. 1", "стр. 2"]
    assert [c.seq for c in chunks] == [0, 1]


def test_unnamed_blocks_get_numbered_locators() -> None:
    chunks = make_chunks([Block(text="Просто текст без структуры.", locator="")], 100, 10)
    assert chunks[0].locator == "фрагмент 1"


def test_long_text_split_by_paragraphs() -> None:
    paragraphs = [f"Абзац номер {i} " + "слово " * 30 for i in range(6)]
    blocks = [Block(text="\n\n".join(paragraphs), locator="раздел «А»")]
    chunks = make_chunks(blocks, chunk_chars=400, overlap_chars=50)
    assert len(chunks) > 1
    assert all(len(c.text) <= 400 for c in chunks)
    assert all(c.locator == "раздел «А»" for c in chunks)
    # весь контент сохранился
    assert "Абзац номер 0" in chunks[0].text
    assert "Абзац номер 5" in chunks[-1].text


def test_giant_paragraph_split_with_overlap() -> None:
    text = "слово " * 300  # один абзац, нет границ \n\n
    chunks = make_chunks([Block(text=text, locator="")], chunk_chars=500, overlap_chars=100)
    assert len(chunks) >= 3
    # перекрытие: конец предыдущего встречается в начале следующего
    tail = chunks[0].text[-50:]
    assert tail.strip().split()[0] in chunks[1].text[:200]


def test_tiny_scraps_are_dropped() -> None:
    chunks = make_chunks([Block(text="ок", locator="")], 100, 10)
    assert chunks == []
