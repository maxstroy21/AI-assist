"""Безопасность путей файловых инструментов (FR-6.5, FR-9.5).

Работа строго внутри whitelisted-корней из конфига (files.allowed_roots);
пути наружу отклоняются после resolve() — защита от .. и симлинков.
Используется и инструментами чтения (tools.py), и файловыми операциями (ops.py).
"""

from __future__ import annotations

from pathlib import Path

NO_ROOTS_HINT = (
    "Файловые инструменты не настроены: список разрешённых папок пуст. "
    "Попросите владельца добавить в config/local.yaml:\n"
    "files:\n  allowed_roots: ['C:\\Users\\имя\\Documents']"
)


class RootGuard:
    """Резолвит пути пользователя и модели строго внутри разрешённых корней."""

    def __init__(self, allowed_roots: list[Path] | list[str]) -> None:
        self._roots = [Path(r).expanduser().resolve() for r in allowed_roots]

    @property
    def roots(self) -> list[Path]:
        return list(self._roots)

    def summary(self) -> str:
        return ", ".join(str(r) for r in self._roots)

    def resolve(self, raw: str) -> Path:
        """Путь (абсолютный или относительный корня) → безопасный абсолютный.

        Путь может не существовать (цель перемещения) — важно лишь, что он
        внутри разрешённых корней. Иначе ValueError.
        """
        if not self._roots:
            raise ValueError(NO_ROOTS_HINT)
        cleaned = raw.strip().strip("'\"")
        # «Downloads» должно означать сам разрешённый корень с таким именем,
        # а не подпапку Downloads внутри корней
        for root in self._roots:
            if cleaned.rstrip("\\/").lower() in (root.name.lower(), str(root).lower()):
                return root
        path = Path(cleaned).expanduser()
        candidates = [path] if path.is_absolute() else [root / path for root in self._roots]
        for candidate in candidates:
            resolved = candidate.resolve()
            if any(resolved.is_relative_to(root) for root in self._roots):
                if resolved.exists() or candidate is candidates[-1]:
                    return resolved
        raise ValueError(
            f"путь {raw!r} вне разрешённых папок. Разрешены: {self.summary()}"
        )
