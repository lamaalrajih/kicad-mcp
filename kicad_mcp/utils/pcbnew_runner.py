"""
Bridge to KiCad's bundled Python (pcbnew) and Specctra/Freerouting helpers.

The ``pcbnew`` SWIG module only imports under KiCad's own interpreter, so the
project's venv shells out to ``kicad_mcp/scripts/pcbnew_cli.py`` using that
interpreter. This module locates the interpreter, parses the KiCad netlist
(``.net``) into the JSON the backend expects, and wraps the subcommands.
"""
import json
import os
import shutil
import subprocess
import tempfile
from typing import Dict, Any, List, Optional

from kicad_mcp.config import KICAD_APP_PATH, system
from kicad_mcp.utils import sexpr

_SCRIPT = os.path.join(os.path.dirname(__file__), "..", "scripts", "pcbnew_cli.py")
_SCRIPT = os.path.abspath(_SCRIPT)


class PcbnewError(Exception):
    pass


def find_kicad_python() -> Optional[str]:
    """Locate the KiCad-bundled Python interpreter that can import pcbnew."""
    env = os.environ.get("KICAD_PYTHON")
    if env and os.path.exists(env):
        return env
    candidates: List[str] = []
    if system == "Darwin":
        fw = os.path.join(KICAD_APP_PATH, "Contents", "Frameworks",
                          "Python.framework", "Versions")
        if os.path.isdir(fw):
            for ver in sorted(os.listdir(fw), reverse=True):
                candidates.append(os.path.join(fw, ver, "bin", "python3"))
    elif system == "Windows":
        candidates.append(os.path.join(KICAD_APP_PATH, "bin", "python.exe"))
    else:  # Linux: pcbnew usually importable from system python3
        candidates.append(shutil.which("python3") or "")
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None


def _run_backend(subcmd: str, extra_args: List[str]) -> Dict[str, Any]:
    py = find_kicad_python()
    if not py:
        raise PcbnewError(
            "Could not find KiCad's bundled Python (needed for pcbnew). "
            "Set the KICAD_PYTHON environment variable to its path."
        )
    cmd = [py, _SCRIPT, subcmd] + extra_args
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    out = (proc.stdout or "").strip()
    # pcbnew prints a benign wxApp assert to stderr; ignore unless no stdout.
    if not out:
        raise PcbnewError(
            f"pcbnew backend produced no output (rc={proc.returncode}). "
            f"stderr: {proc.stderr.strip()[-400:]}"
        )
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        raise PcbnewError(f"Unexpected backend output: {out[:400]}")


# ---- netlist (.net) -> JSON ------------------------------------------------

def parse_kicad_netlist(net_path: str) -> Dict[str, Any]:
    """Parse a KiCad ``.net`` file into ``{components, nodes, nets}``."""
    root = sexpr.parse(open(net_path, "r", encoding="utf-8").read())

    components = []
    comps_node = sexpr.find(root, "components")
    if comps_node:
        for comp in sexpr.find_all(comps_node, "comp"):
            ref = _val(sexpr.find(comp, "ref"))
            value = _val(sexpr.find(comp, "value"))
            fp = _val(sexpr.find(comp, "footprint"))
            components.append({"ref": ref, "value": value, "footprint": fp})

    nodes = []
    names = []
    nets_node = sexpr.find(root, "nets")
    if nets_node:
        for net in sexpr.find_all(nets_node, "net"):
            name = _val(sexpr.find(net, "name"))
            if name:
                names.append(name)
            for node in sexpr.find_all(net, "node"):
                nodes.append({
                    "ref": _val(sexpr.find(node, "ref")),
                    "pin": _val(sexpr.find(node, "pin")),
                    "net": name,
                })
    return {"components": components, "nodes": nodes, "nets": names}


def _val(node) -> str:
    """Return the first argument of an s-expr node as a plain string."""
    if node and len(node) > 1:
        return str(node[1])
    return ""


# ---- public operations -----------------------------------------------------

def export_netlist(schematic_path: str, out_path: str) -> None:
    """Export a KiCad netlist from a schematic using kicad-cli."""
    from kicad_mcp.utils.kicad_cli import get_cli_manager  # lazy import
    cli = get_cli_manager().find_kicad_cli()
    if not cli:
        raise PcbnewError("kicad-cli not found")
    proc = subprocess.run(
        [cli, "sch", "export", "netlist", "--output", out_path, schematic_path],
        capture_output=True, text=True, timeout=120,
    )
    if not os.path.exists(out_path):
        raise PcbnewError(f"netlist export failed: {proc.stderr.strip()[-300:]}")


def update_pcb_from_netlist(board_path: str, netlist_json: Dict[str, Any],
                            footprint_dirs: Optional[List[str]] = None) -> Dict[str, Any]:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump(netlist_json, tf)
        json_path = tf.name
    try:
        args = ["--board", board_path, "--input", json_path]
        for d in (footprint_dirs or []):
            args += ["--footprint-dir", d]
        return _run_backend("update-from-netlist", args)
    finally:
        os.unlink(json_path)


def export_dsn(board_path: str, dsn_path: str) -> Dict[str, Any]:
    return _run_backend("export-dsn", ["--board", board_path, "--output", dsn_path])


def import_ses(board_path: str, ses_path: str) -> Dict[str, Any]:
    return _run_backend("import-ses", ["--board", board_path, "--input", ses_path])
