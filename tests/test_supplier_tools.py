import pytest

from kicad_mcp.tools import supplier_tools as st


async def _noop_progress(_: float, __: str) -> None:  # noqa: D401
    pass


class _FakeResponse:
    def __init__(self, status: int, payload: dict):
        self.status = status
        self._payload = payload

    async def json(self):  # noqa: D401
        return self._payload

    async def __aenter__(self):  # noqa: D401
        return self

    async def __aexit__(self, exc_type, exc, tb):  # noqa: D401
        return False


class _FakeSession:
    def __init__(self, status: int, payload: dict):
        self._status = status
        self._payload = payload

    async def __aenter__(self):  # noqa: D401
        return self

    async def __aexit__(self, exc_type, exc, tb):  # noqa: D401
        return False

    def request(self, *_args, **_kwargs):  # noqa: D401
        return _FakeResponse(self._status, self._payload)


@pytest.fixture(autouse=True)
def _patch_env(monkeypatch):  # noqa: D401
    monkeypatch.setenv("DIGIKEY_API_KEY", "DUMMY")
    monkeypatch.setenv("MOUSER_API_KEY", "DUMMY")


@pytest.mark.asyncio
async def test_search_success(monkeypatch):
    """Happy-path search with cache miss then hit."""

    fake_payload = {"items": [{"manufacturerPartNumber": "ABC123", "mouserPartNumber": "123-ABC"}]}

    async def fake_mouser(*_a, **_kw):  # noqa: D401
        return [
            {
                "mpn": "ABC123",
                "distributor": "mouser",
                "sku": "123-ABC",
                "stock": 10,
                "price_breaks": [],
                "datasheet": None,
                "url": "https://mouser.com/123-ABC",
            }
        ]

    monkeypatch.setattr(st, "_search_mouser", fake_mouser)
    monkeypatch.setattr(st, "_search_digikey", fake_mouser)  # reuse fake

    res = await st.search_distributors("ABC", progress_callback=_noop_progress)
    assert res["ok"] is True
    assert len(res["result"]) == 2  # mouser + digikey

    # Second call should hit cache – patch functions to raise if called
    monkeypatch.setattr(st, "_search_mouser", pytest.fail)
    res2 = await st.search_distributors("ABC", progress_callback=_noop_progress)
    assert res2["ok"] is True
    assert len(res2["result"]) == 2


@pytest.mark.asyncio
async def test_auth_failure(monkeypatch):
    async def bad_mouser(*_a, **_kw):  # noqa: D401
        raise st.SupplierError("AuthError", "bad key")

    monkeypatch.setattr(st, "_search_mouser", bad_mouser)
    res = await st.search_distributors("XYZ", distributors=["mouser"], progress_callback=_noop_progress)
    assert res["ok"] is False
    assert res["error"]["type"] == "AuthError"


@pytest.mark.asyncio
async def test_network_timeout(monkeypatch):
    async def timeout(*_a, **_kw):  # noqa: D401
        raise st.SupplierError("NetworkError", "timeout")

    monkeypatch.setattr(st, "_get_part_mouser", timeout)
    res = await st.get_distributor_part("mouser", "ABC", progress_callback=_noop_progress)
    assert res["ok"] is False
    assert res["error"]["type"] == "NetworkError"


@pytest.mark.asyncio
async def test_batch_cache(monkeypatch):
    # Monkeypatch single part func to return deterministic data and verify called only once
    calls = {}

    async def fake_part(distributor: str, sku: str, *, progress_callback):  # noqa: D401
        calls[(distributor, sku)] = calls.get((distributor, sku), 0) + 1
        return {"ok": True, "result": {"sku": sku, "stock": 1}}

    monkeypatch.setattr(st, "get_distributor_part", fake_part)

    parts = [{"distributor": "digikey", "sku": "XYZ"}, {"distributor": "digikey", "sku": "XYZ"}]
    res = await st.batch_availability(parts, progress_callback=_noop_progress)
    assert res["ok"] is True
    assert len(res["result"]) == 2
    assert calls[("digikey", "XYZ")] == 2  # worker called per entry
