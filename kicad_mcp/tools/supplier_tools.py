"""Supplier lookup tools for KiCad-MCP.

This module adds three *asynchronous* MCP tools that integrate real–time
availability/price lookup from mainstream distributors (currently Mouser and
Digi-Key).  All tools follow the canonical MCP *envelope* structure::

    {
      "ok": bool,
      "result": …,        # present when ok is True
      "error": {
          "type": str,
          "message": str
      },
      "elapsed_s": float
    }

The real distributor APIs are **not** called during unit-tests – internal helper
coroutines such as :pyfunc:`_search_mouser` are designed to be monkey-patched.
"""
from __future__ import annotations

from pathlib import Path
import asyncio
import json
import os
import time
from typing import Callable, Dict, List, Any

import aiohttp  # Public dependency – declared in requirements.txt

# ---------------------------------------------------------------------------
# Constants & shared helpers
# ---------------------------------------------------------------------------

_ERROR_TYPES = {
    "NetworkError",
    "AuthError",
    "RateLimited",
    "NotFound",
    "ParseError",
}

_ResultEnvelope = Dict[str, object]

_PROGRESS_INTERVAL = 0.5  # seconds
_CACHE_PATH = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")) / "kicad_mcp_supplier.json"
_CACHE_TTL = 24 * 3600  # seconds


class SupplierError(Exception):
    """Raised by internal helpers to signal canonical *error type*."""

    def __init__(self, err_type: str, message: str):
        if err_type not in _ERROR_TYPES:
            err_type = "ParseError"
        self.err_type = err_type
        super().__init__(message)


# ---------------------------------------------------------------------------
# Cache helpers – extremely small & JSON-based
# ---------------------------------------------------------------------------

def _load_cache() -> Dict[str, Any]:  # noqa: D401
    if not _CACHE_PATH.exists():
        return {}
    try:
        with _CACHE_PATH.open("r", encoding="utf-8") as fp:
            return json.load(fp)
    except Exception:
        return {}


def _save_cache(cache: Dict[str, Any]) -> None:  # noqa: D401
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = _CACHE_PATH.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fp:
            json.dump(cache, fp)
        tmp.replace(_CACHE_PATH)
    except Exception:
        # Cache failures are silent – never break main workflow
        pass


def _cache_get(key: str) -> Any | None:  # noqa: D401
    cache = _load_cache()
    entry = cache.get(key)
    if not entry:
        return None
    if time.time() - entry["ts"] > _CACHE_TTL:
        return None
    return entry["data"]


def _cache_put(key: str, data: Any) -> None:  # noqa: D401
    cache = _load_cache()
    cache[key] = {"ts": time.time(), "data": data}
    _save_cache(cache)


# ---------------------------------------------------------------------------
# Envelope & progress helpers (copied from component_tools style)
# ---------------------------------------------------------------------------

def _envelope_ok(result, start: float) -> _ResultEnvelope:  # noqa: D401
    return {"ok": True, "result": result, "elapsed_s": time.perf_counter() - start}


def _envelope_err(err_type: str, message: str, start: float) -> _ResultEnvelope:  # noqa: D401
    if err_type not in _ERROR_TYPES:
        err_type = "ParseError"
    return {
        "ok": False,
        "error": {"type": err_type, "message": message},
        "elapsed_s": time.perf_counter() - start,
    }


async def _periodic_progress(
    cancel_event: asyncio.Event, progress_callback: Callable[[float, str], None], msg: str
) -> None:  # noqa: D401,E501
    pct = 0.0
    while not cancel_event.is_set():
        try:
            maybe_coro = progress_callback(pct, msg)
            if asyncio.iscoroutine(maybe_coro):
                await maybe_coro
        except Exception:
            pass
        pct = (pct + 5.0) % 100.0
        await asyncio.sleep(_PROGRESS_INTERVAL)


# ---------------------------------------------------------------------------
# HTTP helpers – shared aiohttp wrapper with timeout & retry
# ---------------------------------------------------------------------------

_DEFAULT_TIMEOUT = aiohttp.ClientTimeout(total=10.0)
_MAX_RETRIES = 2


async def _fetch_json(method: str, url: str, *, headers: Dict[str, str] | None = None, json_data: Any | None = None) -> Any:  # noqa: D401,E501
    for attempt in range(_MAX_RETRIES + 1):
        try:
            async with aiohttp.ClientSession(timeout=_DEFAULT_TIMEOUT) as sess:
                async with sess.request(method, url, headers=headers, json=json_data) as resp:
                    if resp.status in (401, 403):
                        raise SupplierError("AuthError", f"unauthorized: {resp.status}")
                    if resp.status == 429:
                        raise SupplierError("RateLimited", "rate limited")
                    if resp.status >= 500:
                        raise SupplierError("NetworkError", f"server error {resp.status}")
                    if resp.status == 404:
                        raise SupplierError("NotFound", "resource not found")
                    try:
                        return await resp.json()
                    except Exception:
                        raise SupplierError("ParseError", "invalid JSON response")
        except SupplierError:
            raise  # Bubble up without retry – already mapped
        except Exception as exc:
            if attempt >= _MAX_RETRIES:
                raise SupplierError("NetworkError", str(exc))
            # simple exponential back-off
            await asyncio.sleep(0.5 * (2 ** attempt))
    # Should never reach here
    raise SupplierError("NetworkError", "unreachable")


# ---------------------------------------------------------------------------
# Distributor-specific helpers (can be monkey-patched during tests)
# ---------------------------------------------------------------------------

async def _search_mouser(query: str, max_results: int) -> List[Dict[str, object]]:  # noqa: D401
    """Fetch *query* result list from Mouser API (simplified)."""
    api_key = os.getenv("MOUSER_API_KEY")
    if not api_key:
        raise SupplierError("AuthError", "MOUSER_API_KEY env var not set")

    # NOTE: Real Mouser API uses POST /api/v1/search/keyword with JSON payload.
    # We use GET for simplicity – this endpoint is fictional and only for demo.
    url = f"https://api.mouser.com/api/v1/search?apiKey={api_key}&q={query}&n={max_results}"
    data = await _fetch_json("GET", url)

    hits: List[Dict[str, object]] = []
    for item in data.get("items", [])[:max_results]:
        hits.append(
            {
                "mpn": item.get("manufacturerPartNumber"),
                "distributor": "mouser",
                "sku": item.get("mouserPartNumber"),
                "stock": item.get("availability"),
                "price_breaks": [
                    {"qty": p.get("quantity"), "unit_price_usd": p.get("priceUSD")}
                    for p in item.get("priceBreaks", [])
                ],
                "datasheet": item.get("datasheetUrl"),
                "url": item.get("productUrl"),
            }
        )
    return hits


async def _search_digikey(query: str, max_results: int) -> List[Dict[str, object]]:  # noqa: D401
    api_key = os.getenv("DIGIKEY_API_KEY")
    if not api_key:
        raise SupplierError("AuthError", "DIGIKEY_API_KEY env var not set")

    url = f"https://api.digikey.com/parts/search?apiKey={api_key}&keywords={query}&limit={max_results}"
    data = await _fetch_json("GET", url)

    hits: List[Dict[str, object]] = []
    for item in data.get("parts", [])[:max_results]:
        hits.append(
            {
                "mpn": item.get("manufacturerPartNumber"),
                "distributor": "digikey",
                "sku": item.get("digiKeyPartNumber"),
                "stock": item.get("quantityOnHand"),
                "price_breaks": [
                    {"qty": p["breakQty"], "unit_price_usd": p["unitPrice"],}
                    for p in item.get("standardPricing", [])
                ],
                "datasheet": item.get("datasheetUrl"),
                "url": item.get("productUrl"),
            }
        )
    return hits


async def _get_part_mouser(sku_or_mpn: str) -> Dict[str, object]:  # noqa: D401
    # Very thin wrapper around search – real API has dedicated endpoint.
    hits = await _search_mouser(sku_or_mpn, 1)
    if not hits:
        raise SupplierError("NotFound", sku_or_mpn)
    return hits[0]


async def _get_part_digikey(sku_or_mpn: str) -> Dict[str, object]:  # noqa: D401
    hits = await _search_digikey(sku_or_mpn, 1)
    if not hits:
        raise SupplierError("NotFound", sku_or_mpn)
    return hits[0]


# ---------------------------------------------------------------------------
# Public MCP tools
# ---------------------------------------------------------------------------

from mcp.server.fastmcp import FastMCP  # Imported late

_mcp_instance: FastMCP | None = None


async def search_distributors(  # noqa: D401
    query: str,
    distributors: List[str] | None = None,
    max_results: int = 20,
    *,
    progress_callback: Callable[[float, str], None],
) -> _ResultEnvelope:
    """Search Mouser and/or Digi-Key concurrently for *query*."""
    start = time.perf_counter()
    cancel = asyncio.Event()
    spinner = asyncio.create_task(_periodic_progress(cancel, progress_callback, "searching"))

    distributors = [d.lower() for d in (distributors or ["digikey", "mouser"])]
    # Use cache key based on query+distros to allow reuse across calls
    cache_key = f"search::{query}::{','.join(sorted(distributors))}::{max_results}"
    try:
        cached = _cache_get(cache_key)
        if cached is not None:
            cancel.set()
            await spinner
            return _envelope_ok(cached, start)

        tasks: List[asyncio.Future] = []
        if "mouser" in distributors:
            tasks.append(_search_mouser(query, max_results))
        if "digikey" in distributors:
            tasks.append(_search_digikey(query, max_results))
        if cached is not None:
            cancel.set()
            await spinner
            return _envelope_ok(cached, start)

        # Gather concurrently
        results: List[List[Dict[str, object]]] = await asyncio.gather(*tasks, return_exceptions=True)
        hits: List[Dict[str, object]] = []
        for res in results:
            if isinstance(res, Exception):
                raise res  # Will be handled below
            hits.extend(res)
        _cache_put(cache_key, hits)
        cancel.set()
        await spinner
        return _envelope_ok(hits, start)
    except SupplierError as se:
        cancel.set()
        await spinner
        return _envelope_err(se.err_type, str(se), start)
    except Exception as exc:  # pragma: no cover – unexpected
        cancel.set()
        await spinner
        return _envelope_err("NetworkError", str(exc), start)


async def get_distributor_part(  # noqa: D401
    distributor: str,
    sku: str,
    *,
    progress_callback: Callable[[float, str], None],
) -> _ResultEnvelope:
    """Return detailed information for *sku* from *distributor*."""
    start = time.perf_counter()
    cancel = asyncio.Event()
    spinner = asyncio.create_task(_periodic_progress(cancel, progress_callback, "fetching"))

    distributor = distributor.lower()
    cache_key = f"part::{distributor}::{sku}"
    try:
        cached = _cache_get(cache_key)
        if cached is not None:
            cancel.set()
            await spinner
            return _envelope_ok(cached, start)

        if distributor == "mouser":
            data = await _get_part_mouser(sku)
        elif distributor == "digikey":
            data = await _get_part_digikey(sku)
        else:
            raise SupplierError("ParseError", f"unsupported distributor: {distributor}")

        _cache_put(cache_key, data)
        cancel.set()
        await spinner
        return _envelope_ok(data, start)
    except SupplierError as se:
        cancel.set()
        await spinner
        return _envelope_err(se.err_type, str(se), start)
    except Exception as exc:
        cancel.set()
        await spinner
        return _envelope_err("NetworkError", str(exc), start)


async def batch_availability(  # noqa: D401
    parts: List[Dict[str, str]],
    *,
    progress_callback: Callable[[float, str], None],
) -> _ResultEnvelope:
    """Return *latest* stock/price info for each distributor *part* dict."""
    start = time.perf_counter()
    cancel = asyncio.Event()
    spinner = asyncio.create_task(_periodic_progress(cancel, progress_callback, "batch"))

    async def _worker(p: Dict[str, str]):
        res = await get_distributor_part(  # Re-use single-part routine
            p["distributor"], p["sku"], progress_callback=lambda *_: None
        )
        if res["ok"]:
            return res["result"]
        else:
            # Embed error in place so caller keeps index alignment
            return {"error": res["error"]}

    try:
        out = await asyncio.gather(*[_worker(p) for p in parts])
        cancel.set()
        await spinner
        return _envelope_ok(out, start)
    except Exception as exc:  # pragma: no cover
        cancel.set()
        await spinner
        return _envelope_err("NetworkError", str(exc), start)


# ---------------------------------------------------------------------------
# FastMCP registration helper – mirrors *component_tools* pattern
# ---------------------------------------------------------------------------

def register_supplier_tools(mcp: FastMCP) -> None:  # noqa: D401
    """Expose supplier-lookup tools to *FastMCP* instance."""
    global _mcp_instance
    _mcp_instance = mcp

    for impl in [search_distributors, get_distributor_part, batch_availability]:

        async def _stub(*args, __impl=impl, **kwargs):  # type: ignore[override]
            return await __impl(*args, **kwargs)

        _stub.__name__ = impl.__name__
        _stub.__doc__ = impl.__doc__
        mcp.tool()(_stub)
