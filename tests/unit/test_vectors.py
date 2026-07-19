"""Обёртка Qdrant: ленивое создание коллекций, контроль размерности."""

import pytest

from sba.infra.vectors import VectorHit, VectorPoint, VectorStore, VectorStoreError

POINT = VectorPoint(
    id="9f2c6a1e-0000-4000-8000-000000000001",
    vector=[1.0, 0.0, 0.0],
    payload={"user_id": "owner"},
)


async def test_upsert_and_search() -> None:
    store = VectorStore.in_memory()
    await store.ensure_collection("t", dim=3)
    await store.upsert("t", [POINT])
    hits = await store.search("t", [1.0, 0.0, 0.0], k=3)
    assert isinstance(hits[0], VectorHit)
    assert hits[0].payload == {"user_id": "owner"}
    assert await store.count("t") == 1


async def test_search_missing_collection_returns_empty() -> None:
    store = VectorStore.in_memory()
    assert await store.search("nope", [1.0], k=3) == []
    assert await store.count("nope") == 0


async def test_dim_mismatch_is_explicit_error() -> None:
    store = VectorStore.in_memory()
    await store.ensure_collection("t", dim=3)
    with pytest.raises(VectorStoreError, match="переиндекс"):
        await store.ensure_collection("t", dim=5)


async def test_payload_filter() -> None:
    store = VectorStore.in_memory()
    await store.ensure_collection("t", dim=3)
    other = VectorPoint(
        id="9f2c6a1e-0000-4000-8000-000000000002",
        vector=[1.0, 0.0, 0.0],
        payload={"user_id": "чужой"},
    )
    await store.upsert("t", [POINT, other])
    hits = await store.search("t", [1.0, 0.0, 0.0], k=5, payload_filter={"user_id": "owner"})
    assert [h.payload["user_id"] for h in hits] == ["owner"]


async def test_delete_and_drop() -> None:
    store = VectorStore.in_memory()
    await store.ensure_collection("t", dim=3)
    await store.upsert("t", [POINT])
    await store.delete("t", [POINT.id])
    assert await store.count("t") == 0
    await store.drop_collection("t")
    assert await store.search("t", [1.0, 0.0, 0.0], k=1) == []
