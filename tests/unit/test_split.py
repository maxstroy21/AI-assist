from sba.channels.telegram.split import cut_once, split_text


def test_short_text_single_part() -> None:
    assert split_text("привет", limit=100) == ["привет"]


def test_split_prefers_paragraph_boundary() -> None:
    text = "первый абзац\n\nвторой абзац"
    head, rest = cut_once(text, limit=20)
    assert head == "первый абзац"
    assert rest == "второй абзац"


def test_split_long_text_all_parts_within_limit() -> None:
    words = ("слово" + str(i) for i in range(2000))
    text = " ".join(words)
    parts = split_text(text, limit=300)
    assert all(len(p) <= 300 for p in parts)
    assert "".join(p.replace(" ", "") for p in parts) == text.replace(" ", "")


def test_unbreakable_text_hard_cut() -> None:
    text = "х" * 700
    parts = split_text(text, limit=300)
    assert [len(p) for p in parts] == [300, 300, 100]


def test_empty_text() -> None:
    assert split_text("") == []
