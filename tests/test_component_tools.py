from pathlib import Path

import pytest

from kicad_mcp.tools.component_tools import (
    lookup_component_footprint,
    validate_footprint,
    get_footprint_info,
    search_component_libraries,
)


@pytest.fixture()
def footprint_tmp(tmp_path: Path) -> Path:
    """Create a temporary *.pretty* library with 2 footprints."""
    pretty = tmp_path / "Mock.pretty"
    pretty.mkdir()

    # Valid footprint with 2 pads on copper layers
    valid_fp = pretty / "R_0603.kicad_mod"
    valid_fp.write_text(
        """
(kicad_mod (version 20211014) (generator pcbnew)
  (footprint "R_0603" (layer F.Cu)
    (pad 1 smd rect (at -0.5 0) (size 1 1) (layers F.Cu B.Cu) (drill 0.3))
    (pad 2 smd rect (at 0.5 0) (size 1 1) (layers F.Cu B.Cu) (drill 0.3))
  )
)
""",
        encoding="utf-8",
    )

    # Invalid footprint – no pads
    invalid_fp = pretty / "Empty.kicad_mod"
    invalid_fp.write_text("(kicad_mod (footprint Empty))", encoding="utf-8")
    return pretty


async def _noop_progress(pct: float, msg: str) -> None:  # noqa: D401
    pass


@pytest.mark.asyncio
async def test_lookup_component_footprint(footprint_tmp: Path):
    res = await lookup_component_footprint(
        "R_*", libs=[str(footprint_tmp)], progress_callback=_noop_progress
    )
    assert res["ok"] is True
    hits = res["result"]
    assert len(hits) == 1
    assert hits[0]["lib"] == "Mock"


@pytest.mark.asyncio
async def test_validate_footprint_ok(footprint_tmp: Path):
    res = await validate_footprint(
        str(footprint_tmp), "R_0603", progress_callback=_noop_progress
    )
    assert res["ok"] is True
    assert res["result"]["valid"] is True


@pytest.mark.asyncio
async def test_validate_footprint_bad(footprint_tmp: Path):
    res = await validate_footprint(
        str(footprint_tmp), "Empty", progress_callback=_noop_progress
    )
    assert res["ok"] is True
    assert res["result"]["valid"] is False


@pytest.mark.asyncio
async def test_get_footprint_info(footprint_tmp: Path):
    res = await get_footprint_info(
        str(footprint_tmp), "R_0603", progress_callback=_noop_progress
    )
    assert res["ok"] is True
    info = res["result"]
    assert info["pin_count"] == 2
    assert len(info["pads"]) == 2
    assert info["bounding_box"]["units"] == "mm"


@pytest.mark.asyncio
async def test_search_component_libraries(footprint_tmp: Path, monkeypatch):
    monkeypatch.setenv("MCP_FOOTPRINT_PATHS", str(footprint_tmp))
    res = await search_component_libraries("*Mock*", progress_callback=_noop_progress)
    assert res["ok"] is True
    libs = res["result"]
    assert any("Mock.pretty" in p for p in libs)
