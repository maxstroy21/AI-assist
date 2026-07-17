"""Точка входа: python -m sba [--config-dir config]."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from sba.app import run_app
from sba.infra.config import ConfigError


def main() -> None:
    parser = argparse.ArgumentParser(prog="sba", description="Second Brain Agent")
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("config"),
        help="папка с default.yaml / local.yaml (по умолчанию: ./config)",
    )
    args = parser.parse_args()
    try:
        asyncio.run(run_app(args.config_dir))
    except ConfigError as exc:
        print(f"Ошибка конфигурации: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
