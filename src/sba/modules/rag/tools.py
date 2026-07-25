"""Инструмент поиска по личным документам (FR-2: «найди в моих документах…»).

Результат размечен как данные с источниками: модель обязана цитировать
файл и место — это проверяемо владельцем (защита от фабрикации, урок Sprint 2).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from sba.core.tools.spec import RiskLevel, ToolSpec
from sba.modules.rag.service import Passage, RAGService


class SearchDocumentsArgs(BaseModel):
    query: str = Field(
        description="Что искать в документах: тема, вопрос или ключевые слова"
    )


def format_passages(passages: list[Passage], snippet_chars: int) -> str:
    lines: list[str] = [
        "Найденные фрагменты документов (отвечай ТОЛЬКО по ним; в ответе "
        "укажи файл-источник и место):"
    ]
    for n, passage in enumerate(passages, start=1):
        snippet = " ".join(passage.text.split())
        if len(snippet) > snippet_chars:
            snippet = snippet[:snippet_chars] + "…"
        where = f", {passage.locator}" if passage.locator else ""
        lines.append(f"{n}. 📄 {passage.path}{where}\n«{snippet}»")
    return "\n\n".join(lines)


def build_rag_tools(
    service: RAGService,
    top_k: int,
    snippet_chars: int,
    configured: bool,
) -> list[ToolSpec]:
    async def search_documents(args: BaseModel) -> str:
        assert isinstance(args, SearchDocumentsArgs)
        if not configured:
            return (
                "Поиск по документам не настроен: список папок modules.rag.sources "
                "пуст. Попросите владельца добавить папки в config/local.yaml."
            )
        passages = await service.search(args.query, k=top_k)
        if not passages:
            return (
                f"По запросу {args.query!r} в проиндексированных документах ничего "
                "не найдено. Так и скажи пользователю — не выдумывай содержимое."
            )
        return format_passages(passages, snippet_chars)

    return [
        ToolSpec(
            name="search_documents",
            description=(
                "Поиск по содержимому документов (PDF/DOCX/XLSX/MD/TXT); "
                "возвращает фрагменты с указанием файла"
            ),
            args_schema=SearchDocumentsArgs,
            risk=RiskLevel.READ,
            module="rag",
            handler=search_documents,
        )
    ]
