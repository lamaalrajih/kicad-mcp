"""Component–Footprint lookup & validation tools for KiCad-MCP.

All tools defined in this module follow the canonical *envelope* expected by the
MCP ecosystem::

    {
      "ok": bool,
      "result": …,        # present when ok is True
      "error": {          # present when ok is False
          "type": str,
          "message": str
      },
      "elapsed_s": float
    }

Implementation choices & notes
------------------------------
* *Pattern matching*: `lookup_component_footprint` and
  `search_component_libraries` treat *pattern* as a **Unix-style glob** (as used
  by :pymod:`fnmatch`). Users may supply ``*`` and ``?`` wild-cards.
* *Async design*: All potentially blocking filesystem work is executed inside
  :pyfunc:`asyncio.to_thread` keeping the FastAPI event-loop responsive.  Every
  tool emits ``progress_callback`` updates roughly every 0.5 s.
* *KiCad parsing*: A **very light-weight** text parser is implemented for
  ``.kicad_mod`` files so the tools work in CI where the native
  :pymod:`pcbnew` Python bindings are unavailable.  When ``pcbnew`` *is*
  importable at runtime we transparently prefer it as it offers richer and more
  robust parsing.
* *Error taxonomy*:  The tools raise only the error types requested by the
  specification – see :data:`_ERROR_TYPES`.
"""
from __future__ import annotations
from pathlib import Path

import asyncio
import fnmatch
import os
import re
import time

from typing import Callable, Dict, List, Tuple

try:
    import pcbnew  # type: ignore

    _PCBNEW_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover – pcbnew not present in CI
    _PCBNEW_AVAILABLE = False

# ---------------------------------------------------------------------------
# Toolbox helpers
# ---------------------------------------------------------------------------

_ERROR_TYPES = {
    "FileNotFound",
    "ParseError",
    "Timeout",
    "UnsupportedVersion",
    "InvalidFootprint",
    "LibraryNotAllowed",
}

_ResultEnvelope = Dict[str, object]

_PROGRESS_INTERVAL = 0.5  # seconds


async def _run_io(func, *args, **kwargs):
    """Convenience wrapper around *asyncio.to_thread*."""
    return await asyncio.to_thread(func, *args, **kwargs)


def _envelope_ok(result, start: float) -> _ResultEnvelope:  # noqa: D401
    """Return a *success* envelope."""
    return {
        "ok": True,
        "result": result,
        "elapsed_s": time.perf_counter() - start,
    }


def _envelope_err(err_type: str, message: str, start: float) -> _ResultEnvelope:  # noqa: D401
    """Return an *error* envelope using the canonical structure."""
    if err_type not in _ERROR_TYPES:
        err_type = "ParseError"  # fallback
    return {
        "ok": False,
        "error": {"type": err_type, "message": message},
        "elapsed_s": time.perf_counter() - start,
    }


async def _periodic_progress(cancel_event: asyncio.Event, progress_callback: Callable[[float, str], None], msg: str) -> None:  # noqa: D401,E501
    """Emit *msg* every ~0.5 s until *cancel_event* is set."""
    pct = 0.0
    while not cancel_event.is_set():
        try:
            progress_callback(pct, msg)
        except Exception:  # pragma: no cover – progress failures must not crash
            pass
        pct = (pct + 2.0) % 100  # simple spinner
        await asyncio.sleep(_PROGRESS_INTERVAL)


# ---------------------------------------------------------------------------
# Library discovery helpers
# ---------------------------------------------------------------------------

def _collect_library_paths() -> List[Path]:
    """Return all search roots for footprint libraries (.pretty dirs)."""
    paths: List[Path] = []
    # Project table – look for any *fp-lib-table* reachable from CWD.
    cwd_table = Path.cwd() / "fp-lib-table"
    if cwd_table.exists():
        paths += _parse_fp_lib_table(cwd_table)

    # Global fp-lib-table (KiCad 6):
    home = Path.home()
    for guess in [
        home / ".config" / "kicad" / "fp-lib-table",
        home / ".config" / "kicad/6.0" / "fp-lib-table",
    ]:
        if guess.exists():
            paths += _parse_fp_lib_table(guess)
            break

    # Environment variables
    env_kicad = os.environ.get("KICAD6_FOOTPRINT_DIR")
    if env_kicad:
        paths.append(Path(env_kicad))

    extra = os.environ.get("MCP_FOOTPRINT_PATHS")
    if extra:
        for p in extra.split(os.pathsep):
            paths.append(Path(p))

    # De-duplicate while keeping order
    seen = set()
    unique: List[Path] = []
    for p in paths:
        try:
            real = p.resolve()
        except Exception:
            real = p
        if real not in seen:
            unique.append(real)
            seen.add(real)
    return unique


def _parse_fp_lib_table(table_path: Path) -> List[Path]:
    """Parse a KiCad *fp-lib-table* returning library directories.*"""
    libs: List[Path] = []
    try:
        txt = table_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return libs

    for match in re.finditer(r"\(lib +\((?:[^()]|\([^)]*\))*\)\)", txt):
        block = match.group(0)
        path_m = re.search(r"\(uri +([^ )]+)\)", block)
        if path_m:
            uri = path_m.group(1).strip().strip('"')
            if uri.startswith("${KICAD6_3RD_PARTY}"):  # ignore for now
                continue
            libs.append(Path(uri))
    return libs


# ---------------------------------------------------------------------------
# Lightweight .kicad_mod parser (fallback when pcbnew unavailable)
# ---------------------------------------------------------------------------

_COORD_RE = re.compile(r"\(xy +([-+]?[0-9]*\.?[0-9]+) +([-+]?[0-9]*\.?[0-9]+)\)")
_PAD_RE = re.compile(r"\(pad +([^ ]+) +([^ ]+) +([^ ]+)")
_LAYER_SET_RE = re.compile(r"layers +([^\)]+)\)")
_DRILL_RE = re.compile(r"drill +([^ )]+)")


class _ModParseResult(Tuple):
    pin_count: int
    pads: List[Dict[str, object]]
    bbox: Dict[str, object]
    layers: List[str]


def _parse_kicad_mod(file_path: Path) -> _ModParseResult:
    """Very small text-based parser for *pcb footprint* files."""
    try:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        raise ValueError(f"cannot read footprint: {exc}")

    pads: List[Dict[str, object]] = []
    xs: List[float] = []
    ys: List[float] = []
    layers_set: set[str] = set()

    for pad_m in _PAD_RE.finditer(text):
        pad_num, pad_type, pad_shape = pad_m.groups()
        # Find position immediately after this match for nested search
        start = pad_m.end()
        end = text.find("(pad", start)  # naive – find next pad occurrence
        snippet = text[start:end] if end != -1 else text[start:]
        # Position (xy …)
        pos_m = _COORD_RE.search(snippet)
        x, y = (float(pos_m.group(1)), float(pos_m.group(2))) if pos_m else (0.0, 0.0)
        xs.append(x)
        ys.append(y)
        # Drill size if exists
        drill_m = _DRILL_RE.search(snippet)
        drill = float(drill_m.group(1)) if drill_m else None
        # Layer set
        layer_m = _LAYER_SET_RE.search(snippet)
        if layer_m:
            layers_local = [l.strip() for l in layer_m.group(1).split()]
            layers_set.update(layers_local)
        pads.append(
            {
                "number": pad_num,
                "type": pad_type,
                "shape": pad_shape,
                "drill": drill,
                "x": x,
                "y": y,
            }
        )

    if not pads:
        raise ValueError("footprint has no pads")

    if not layers_set.intersection({"F.Cu", "B.Cu"}):
        raise ValueError("no copper layers enabled")

    if xs and ys:
        w = max(xs) - min(xs)
        h = max(ys) - min(ys)
    else:
        raise ValueError("unable to derive bounding box")

    bbox = {"w": abs(w), "h": abs(h), "units": "mm"}
    return len(pads), pads, bbox, sorted(layers_set)


# ---------------------------------------------------------------------------
# Public MCP tools
# ---------------------------------------------------------------------------

from mcp.server.fastmcp import FastMCP  # imported late to avoid heavy deps

_mcp_instance: FastMCP | None = None  # filled by register_component_tools


async def lookup_component_footprint(  # noqa: D401
    query: str,
    libs: List[str] | None = None,
    *,
    progress_callback: Callable[[float, str], None],
) -> _ResultEnvelope:
    """Search KiCad footprint libraries for *query* (glob match).

    If *libs* is provided it must be a list of ``.pretty`` directories that
    should be searched *exclusively*.
    """
    start = time.perf_counter()
    cancel = asyncio.Event()
    spinner = asyncio.create_task(_periodic_progress(cancel, progress_callback, "searching"))

    try:
        roots = [Path(p) for p in libs] if libs else _collect_library_paths()
        hits: List[Dict[str, str]] = []

        async def scan_lib(root: Path):
            if not root.exists():
                return
            libname = root.stem
            for fp in root.glob("*.kicad_mod"):
                if fnmatch.fnmatch(fp.stem, query):
                    hits.append({"lib": libname, "path": str(fp.resolve())})

        await asyncio.gather(*[scan_lib(r) for r in roots])
        cancel.set()
        await spinner
        return _envelope_ok(hits, start)
    except Exception as exc:  # pragma: no cover
        cancel.set()
        await spinner
        return _envelope_err("ParseError", str(exc), start)


async def validate_footprint(  # noqa: D401
    lib_path: str,
    fp_name: str,
    *,
    progress_callback: Callable[[float, str], None],
) -> _ResultEnvelope:
    """Lightweight structural validation of *fp_name* inside *lib_path*."""
    start = time.perf_counter()
    cancel = asyncio.Event()
    spinner = asyncio.create_task(_periodic_progress(cancel, progress_callback, "validating"))

    try:
        lib_dir = Path(lib_path)
        if not lib_dir.exists():
            raise FileNotFoundError(lib_path)
        fp_file = lib_dir / (fp_name if fp_name.endswith(".kicad_mod") else f"{fp_name}.kicad_mod")
        if not fp_file.exists():
            raise FileNotFoundError(fp_file)

        try:
            if _PCBNEW_AVAILABLE:
                # pcbnew parsing – minimal to avoid heavy object traversal
                mod = await _run_io(pcbnew.FootprintLoad, str(lib_dir), fp_name)
                pin_count = mod.GetPadCount()
                if pin_count == 0:
                    raise ValueError("footprint has no pads")
                # Copper layer check
                any_copper = any(pad.IsOnLayer(pcbnew.F_Cu) or pad.IsOnLayer(pcbnew.B_Cu) for pad in mod.Pads())
                if not any_copper:
                    raise ValueError("no copper layers enabled")
                # BBox
                _ = mod.GetBoundingBox()
            else:
                # Fallback text parser
                _parse_kicad_mod(fp_file)
        except ValueError as ve:
            cancel.set()
            await spinner
            return _envelope_ok({"valid": False, "reason": str(ve)}, start)
        except Exception as exc:
            cancel.set()
            await spinner
            return _envelope_err("ParseError", str(exc), start)

        cancel.set()
        await spinner
        return _envelope_ok({"valid": True, "reason": None}, start)
    except FileNotFoundError as nf:
        cancel.set()
        await spinner
        return _envelope_err("FileNotFound", str(nf), start)
    except Exception as exc:  # pragma: no cover
        cancel.set()
        await spinner
        return _envelope_err("ParseError", str(exc), start)


async def get_footprint_info(  # noqa: D401
    lib_path: str,
    fp_name: str,
    *,
    progress_callback: Callable[[float, str], None],
) -> _ResultEnvelope:
    """Return rich metadata extracted from *fp_name* inside *lib_path*."""
    start = time.perf_counter()
    cancel = asyncio.Event()
    spinner = asyncio.create_task(_periodic_progress(cancel, progress_callback, "parsing"))

    try:
        lib_dir = Path(lib_path)
        fp_file = lib_dir / (fp_name if fp_name.endswith(".kicad_mod") else f"{fp_name}.kicad_mod")
        if not fp_file.exists():
            raise FileNotFoundError(fp_file)

        if _PCBNEW_AVAILABLE:
            # Run in thread – pcbnew is C++ binding but can block.
            mod = await _run_io(pcbnew.FootprintLoad, str(lib_dir), fp_name)
            pads = []
            for pad in mod.Pads():
                layers = [pcbnew.LayerName(layer) for layer in pad.GetLayerSet().Layers()]
                pads.append(
                    {
                        "number": pad.GetPadName(),
                        "shape": str(pad.GetShape()),
                        "drill": pad.GetDrillValue(),
                        "x": pad.GetPosition().x / 1e6,
                        "y": pad.GetPosition().y / 1e6,
                        "layers": layers,
                    }
                )
            bbox_kicad = mod.GetBoundingBox()
            bbox = {
                "w": bbox_kicad.GetWidth() / 1e6,
                "h": bbox_kicad.GetHeight() / 1e6,
                "units": "mm",
            }
            layer_set = list({l for p in pads for l in p["layers"]})
            pin_count = len(pads)
        else:
            pin_count, pads, bbox, layer_set = _parse_kicad_mod(fp_file)

        info = {
            "pin_count": pin_count,
            "pads": pads,
            "bounding_box": bbox,
            "layer_set": layer_set,
            "raw_source": fp_file.read_text(encoding="utf-8", errors="ignore"),
        }
        cancel.set()
        await spinner
        return _envelope_ok(info, start)
    except FileNotFoundError as nf:
        cancel.set()
        await spinner
        return _envelope_err("FileNotFound", str(nf), start)
    except ValueError as ve:
        cancel.set()
        await spinner
        return _envelope_err("InvalidFootprint", str(ve), start)
    except Exception as exc:  # pragma: no cover
        cancel.set()
        await spinner
        return _envelope_err("ParseError", str(exc), start)


async def search_component_libraries(  # noqa: D401
    pattern: str,
    *,
    user_only: bool = False,
    progress_callback: Callable[[float, str], None],
) -> _ResultEnvelope:
    """Return library directories with a nickname or path matching *pattern*."""
    start = time.perf_counter()
    cancel = asyncio.Event()
    spinner = asyncio.create_task(_periodic_progress(cancel, progress_callback, "scanning"))

    try:
        roots = _collect_library_paths()
        matches: List[str] = []
        for root in roots:
            if user_only:
                # Consider only libs located inside CWD (project local)
                try:
                    root.relative_to(Path.cwd())
                except ValueError:
                    continue
            nickname = root.stem
            if fnmatch.fnmatch(nickname, pattern) or fnmatch.fnmatch(str(root), pattern):
                matches.append(str(root.resolve()))
        cancel.set()
        await spinner
        return _envelope_ok(matches, start)
    except Exception as exc:  # pragma: no cover
        cancel.set()
        await spinner
        return _envelope_err("ParseError", str(exc), start)


# ---------------------------------------------------------------------------
# Registration helper – called by server
# ---------------------------------------------------------------------------

def register_component_tools(mcp: FastMCP) -> None:  # noqa: D401
    """Expose the four *component/footprint* tools to FastMCP."""
    global _mcp_instance
    _mcp_instance = mcp

    # We cannot simply decorate the *already-defined* functions after-the-fact,
    # FastMCP inspects the wrapper *function* object.  Therefore we wrap each
    # implementation in a thin stub that forwards the call.

    for impl in [
        lookup_component_footprint,
        validate_footprint,
        get_footprint_info,
        search_component_libraries,
    ]:

        async def _stub(*args, __impl=impl, **kwargs):  # type: ignore[override]
            return await __impl(*args, **kwargs)

        _stub.__name__ = impl.__name__  # ensure predictable tool id
        _stub.__doc__ = impl.__doc__
        mcp.tool()(_stub)

