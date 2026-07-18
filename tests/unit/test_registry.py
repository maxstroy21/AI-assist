from pathlib import Path

import pytest
from pydantic import BaseModel

from sba.core.tools.registry import ConfirmationRequired, ToolRegistry
from sba.core.tools.spec import RiskLevel, ToolSpec
from sba.infra.audit import AuditLog
from sba.infra.db import Database
from sba.llm.gateway import ToolCall


class ProbeArgs(BaseModel):
    value: str


def make_spec(name: str = "probe", risk: RiskLevel = RiskLevel.READ, handler=None) -> ToolSpec:
    async def default_handler(args: BaseModel) -> str:
        assert isinstance(args, ProbeArgs)
        return f"результат:{args.value}"

    return ToolSpec(
        name=name,
        description="тестовый инструмент",
        args_schema=ProbeArgs,
        risk=risk,
        module="test",
        handler=handler or default_handler,
    )


@pytest.fixture
async def db(tmp_path: Path) -> Database:
    database = await Database.open(tmp_path / "test.db")
    yield database
    await database.close()


@pytest.fixture
def registry(db: Database) -> ToolRegistry:
    return ToolRegistry(AuditLog(db))


async def test_execute_and_audit(registry: ToolRegistry, db: Database) -> None:
    registry.register(make_spec())
    result = await registry.execute(ToolCall(id="1", name="probe", arguments={"value": "x"}))
    assert result.text == "результат:x"
    assert not result.error

    kinds = [r["kind"] for r in await db.fetch_all("SELECT kind FROM audit_log ORDER BY id")]
    assert kinds == ["tool_call", "tool_result"]


async def test_duplicate_registration_fails(registry: ToolRegistry) -> None:
    registry.register(make_spec())
    with pytest.raises(ValueError, match="уже зарегистрирован"):
        registry.register(make_spec())


async def test_unknown_tool_returns_error_result(registry: ToolRegistry) -> None:
    registry.register(make_spec())
    result = await registry.execute(ToolCall(id="1", name="ghost", arguments={}))
    assert result.error
    assert "не существует" in result.text
    assert "probe" in result.text  # подсказка со списком доступных


async def test_invalid_arguments_become_error_result(registry: ToolRegistry) -> None:
    registry.register(make_spec())
    result = await registry.execute(ToolCall(id="1", name="probe", arguments={"wrong": 1}))
    assert result.error
    assert "Ошибка аргументов" in result.text


async def test_handler_exception_does_not_propagate(registry: ToolRegistry) -> None:
    async def boom(args: BaseModel) -> str:
        raise RuntimeError("сломался")

    registry.register(make_spec(handler=boom))
    result = await registry.execute(ToolCall(id="1", name="probe", arguments={"value": "x"}))
    assert result.error
    assert "сломался" in result.text


async def test_destructive_requires_confirmation(registry: ToolRegistry, db: Database) -> None:
    executed = []

    async def dangerous(args: BaseModel) -> str:
        executed.append(True)
        return "удалено"

    registry.register(make_spec(risk=RiskLevel.DESTRUCTIVE, handler=dangerous))
    call = ToolCall(id="1", name="probe", arguments={"value": "x"})

    with pytest.raises(ConfirmationRequired):
        await registry.execute(call)
    assert executed == []

    result = await registry.execute(call, confirmed=True)
    assert result.text == "удалено"

    rows = await db.fetch_all("SELECT kind, confirmed FROM audit_log ORDER BY id")
    assert [r["kind"] for r in rows] == ["confirmation_requested", "tool_call", "tool_result"]
    assert rows[1]["confirmed"] == 1


async def test_hanging_tool_interrupted(
    registry: ToolRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio

    import sba.core.tools.registry as registry_module

    monkeypatch.setattr(registry_module, "TOOL_TIMEOUT_SECONDS", 0.05)

    async def hangs(args: BaseModel) -> str:
        await asyncio.sleep(10)
        return "никогда"

    registry.register(make_spec(handler=hangs))
    result = await registry.execute(ToolCall(id="1", name="probe", arguments={"value": "x"}))
    assert result.error
    assert "прерван" in result.text


async def test_huge_result_truncated(registry: ToolRegistry) -> None:
    async def huge(args: BaseModel) -> str:
        return "x" * 100_000

    registry.register(make_spec(handler=huge))
    result = await registry.execute(ToolCall(id="1", name="probe", arguments={"value": "x"}))
    assert len(result.text) < 7000
    assert "обрезан" in result.text
