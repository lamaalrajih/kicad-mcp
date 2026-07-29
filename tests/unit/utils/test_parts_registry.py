"""
Tests for the parts registry client (stubbed network — no real requests).
"""

import io
import json
import os
import tempfile
from contextlib import contextmanager

import pytest

from kicad_mcp.utils.parts_registry import (
    MAX_ASSET_BYTES,
    RegistryClient,
    RegistryError,
    asset_url_allowed,
    filename_for_asset,
)

INDEX = [
    {
        "id": "usb_c_16p",
        "name": "USB-C 16P",
        "family": "USB-C",
        "category": "connector",
        "manufacturer": "HRO",
        "keywords": ["usb", "type-c"],
        "page": "https://partreel.com/p/usb_c_16p/",
    },
    {
        "id": "jst_ph_4pin",
        "name": "JST PH 4-pin",
        "family": "JST PH",
        "category": "connector",
        "manufacturer": "JST",
        "keywords": ["jst", "ph"],
        "page": "https://partreel.com/p/jst_ph_4pin/",
    },
]

PART = {
    "id": "usb_c_16p",
    "license": "CC-BY-4.0",
    "files": {
        "footprint": "https://partreel.com/library/x/usb_c_16p/usb_c_16p.kicad_mod",
        "symbol": "https://partreel.com/library/x/usb_c_16p/usb_c_16p.kicad_sym",
    },
}


def make_opener(payloads: dict[str, bytes]):
    """Return an opener whose response body depends on the requested URL."""

    @contextmanager
    def opener(req, timeout=None):
        url = req.full_url
        for key, body in payloads.items():
            if key in url:
                yield io.BytesIO(body)
                return
        raise AssertionError(f"unexpected URL fetched: {url}")

    return opener


class TestSearch:
    def _client(self):
        return RegistryClient(
            "https://partreel.com/api/v1",
            opener=make_opener({"parts.json": json.dumps(INDEX).encode()}),
        )

    def test_token_and_match(self):
        hits = self._client().search("usb 16p")
        assert [p["id"] for p in hits] == ["usb_c_16p"]

    def test_keyword_match(self):
        hits = self._client().search("type-c")
        assert [p["id"] for p in hits] == ["usb_c_16p"]

    def test_no_match(self):
        assert self._client().search("nonexistent") == []

    def test_limit(self):
        assert len(self._client().search("connector", limit=1)) == 1

    def test_index_cached_single_fetch(self):
        calls = []

        @contextmanager
        def opener(req, timeout=None):
            calls.append(req.full_url)
            yield io.BytesIO(json.dumps(INDEX).encode())

        client = RegistryClient("https://partreel.com/api/v1", opener=opener)
        client.search("usb")
        client.search("jst")
        assert len(calls) == 1


class TestGetPart:
    def test_get_part(self):
        client = RegistryClient(
            "https://partreel.com/api/v1",
            opener=make_opener({"parts/usb_c_16p.json": json.dumps(PART).encode()}),
        )
        assert client.get_part("usb_c_16p")["id"] == "usb_c_16p"

    def test_invalid_part_id_rejected(self):
        client = RegistryClient("https://partreel.com/api/v1")
        with pytest.raises(RegistryError):
            client.get_part("../../etc/passwd")


class TestAssetSecurity:
    def test_registry_host_allowed(self):
        assert asset_url_allowed(
            "https://partreel.com/library/a.kicad_mod", "https://partreel.com/api/v1"
        )

    def test_subdomain_allowed(self):
        assert asset_url_allowed(
            "https://assets.partreel.com/a.glb", "https://partreel.com/api/v1"
        )

    def test_foreign_host_rejected(self):
        assert not asset_url_allowed(
            "https://evil.example.com/a.kicad_mod", "https://partreel.com/api/v1"
        )

    def test_lookalike_suffix_rejected(self):
        assert not asset_url_allowed(
            "https://notpartreel.com/a.kicad_mod", "https://partreel.com/api/v1"
        )

    def test_http_rejected(self):
        assert not asset_url_allowed(
            "http://partreel.com/a.kicad_mod", "https://partreel.com/api/v1"
        )

    def test_extra_hosts_env(self, monkeypatch):
        monkeypatch.setenv("PARTS_REGISTRY_ASSET_HOSTS", "mirror.example.org")
        assert asset_url_allowed(
            "https://mirror.example.org/a.kicad_mod", "https://partreel.com/api/v1"
        )

    def test_filename_from_remote_basename(self):
        assert (
            filename_for_asset(
                "https://partreel.com/lib/part1.kicad_mod", "part1", "footprint"
            )
            == "part1.kicad_mod"
        )

    def test_filename_bad_extension_replaced(self):
        assert (
            filename_for_asset(
                "https://partreel.com/lib/malware.exe", "part1", "footprint"
            )
            == "part1.kicad_mod"
        )

    def test_step_accepts_stp(self):
        assert (
            filename_for_asset("https://partreel.com/lib/m.stp", "part1", "step")
            == "m.stp"
        )

    def test_size_cap(self):
        big = b"x" * (MAX_ASSET_BYTES + 1)
        client = RegistryClient(
            "https://partreel.com/api/v1",
            opener=make_opener({"parts.json": big}),
        )
        with pytest.raises(RegistryError):
            client.index()


class TestDownload:
    def test_download_writes_file(self):
        client = RegistryClient(
            "https://partreel.com/api/v1",
            opener=make_opener({"usb_c_16p.kicad_mod": b"(footprint...)"}),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = client.download_asset(
                PART["files"]["footprint"], tmp, "usb_c_16p", "footprint"
            )
            assert os.path.basename(path) == "usb_c_16p.kicad_mod"
            with open(path, "rb") as f:
                assert f.read() == b"(footprint...)"

    def test_download_foreign_host_rejected(self):
        client = RegistryClient("https://partreel.com/api/v1")
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(RegistryError):
                client.download_asset(
                    "https://evil.example.com/a.kicad_mod", tmp, "p", "footprint"
                )
