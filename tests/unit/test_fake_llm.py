from sba.llm.gateway import ChatMessage, LLMGateway
from sba.llm.providers.fake import FakeLLM


async def test_fake_llm_replies_in_order_and_records_calls() -> None:
    llm = FakeLLM(replies=["первый", "второй"])

    r1 = await llm.chat("chat", [ChatMessage(role="user", content="а")])
    r2 = await llm.chat("extraction", [ChatMessage(role="user", content="б")])
    r3 = await llm.chat("chat", [ChatMessage(role="user", content="в")])

    assert (r1.text, r2.text, r3.text) == ("первый", "второй", "ok")
    assert [role for role, _ in llm.calls] == ["chat", "extraction", "chat"]


def test_fake_llm_satisfies_gateway_protocol() -> None:
    gateway: LLMGateway = FakeLLM()
    assert gateway is not None
