"""
Schematic authoring helpers for KiCad.

Builds and edits ``.kicad_sch`` files programmatically (and creates the
accompanying ``.kicad_pro`` / ``.kicad_pcb``) using the small S-expression
reader/writer in :mod:`kicad_mcp.utils.sexpr`. Symbol definitions are loaded
from the KiCad standard symbol libraries and embedded into the schematic's
``lib_symbols`` block, exactly as the KiCad editor does.

This is intentionally headless: it does not require a running KiCad instance,
only the installed symbol libraries (for looking up symbols) and, optionally,
a template board (for a valid empty PCB).
"""
import os
import copy
import glob
import math
import uuid as _uuid
from typing import Dict, List, Any, Tuple, Optional

from kicad_mcp.config import KICAD_APP_PATH, system
from kicad_mcp.utils import sexpr
from kicad_mcp.utils.sexpr import Sym


# --------------------------------------------------------------------------
# Locating KiCad shared resources
# --------------------------------------------------------------------------

def _shared_support_dir() -> Optional[str]:
    """Return KiCad's SharedSupport-equivalent resource directory."""
    if system == "Darwin":
        return os.path.join(KICAD_APP_PATH, "Contents", "SharedSupport")
    if system == "Windows":
        return os.path.join(KICAD_APP_PATH, "share", "kicad")
    # Linux and unknown
    for cand in ("/usr/share/kicad", "/usr/local/share/kicad", KICAD_APP_PATH):
        if cand and os.path.isdir(cand):
            return cand
    return None


def symbol_library_dirs() -> List[str]:
    """Directories that may contain ``.kicad_sym`` libraries.

    An explicit override may be supplied via the ``KICAD_SYMBOL_DIR``
    environment variable (useful on unusual installs).
    """
    dirs: List[str] = []
    env = os.environ.get("KICAD_SYMBOL_DIR")
    if env:
        dirs.extend(p for p in env.split(os.pathsep) if p)
    shared = _shared_support_dir()
    if shared:
        dirs.append(os.path.join(shared, "symbols"))
    return [d for d in dirs if os.path.isdir(d)]


def _template_pcb_path() -> Optional[str]:
    """Find any installed template board to derive a valid layer stackup."""
    shared = _shared_support_dir()
    if not shared:
        return None
    pattern = os.path.join(shared, "template", "**", "*.kicad_pcb")
    matches = sorted(glob.glob(pattern, recursive=True))
    return matches[0] if matches else None


# --------------------------------------------------------------------------
# Symbol loading
# --------------------------------------------------------------------------

class SymbolNotFound(Exception):
    pass


def load_symbol(lib_id: str) -> Tuple[list, List[Dict[str, Any]]]:
    """Load a symbol definition by ``lib_id`` (e.g. ``"Device:R"``).

    Returns ``(symbol_node, pins)`` where ``symbol_node`` is a deep-copied
    S-expression tree renamed to ``lib_id`` (ready to drop into
    ``lib_symbols``) and ``pins`` is a list of
    ``{"number", "name", "x", "y", "angle", "length"}`` in symbol-local
    coordinates.
    """
    if ":" not in lib_id:
        raise ValueError(f"lib_id must be 'Library:Symbol', got {lib_id!r}")
    lib_name, sym_name = lib_id.split(":", 1)

    lib_file = None
    for d in symbol_library_dirs():
        cand = os.path.join(d, f"{lib_name}.kicad_sym")
        if os.path.isfile(cand):
            lib_file = cand
            break
    if not lib_file:
        raise SymbolNotFound(
            f"Symbol library '{lib_name}' not found in {symbol_library_dirs()}"
        )

    with open(lib_file, "r", encoding="utf-8") as f:
        root = sexpr.parse(f.read())

    target = None
    for child in sexpr.find_all(root, "symbol"):
        if len(child) >= 2 and str(child[1]) == sym_name:
            target = child
            break
    if target is None:
        raise SymbolNotFound(f"Symbol '{sym_name}' not found in {lib_file}")

    node = copy.deepcopy(target)
    # Rename the top-level symbol to the fully-qualified lib_id.
    node[1] = lib_id

    pins = _collect_pins(node)
    return node, pins


def _collect_pins(symbol_node: list) -> List[Dict[str, Any]]:
    """Recursively gather pin definitions from a symbol (and its units)."""
    pins: List[Dict[str, Any]] = []

    def walk(node: list):
        for child in node:
            if not isinstance(child, list) or not child:
                continue
            head = str(child[0])
            if head == "pin":
                at = sexpr.find(child, "at")
                length_node = sexpr.find(child, "length")
                number_node = sexpr.find(child, "number")
                name_node = sexpr.find(child, "name")
                if at is None or number_node is None:
                    continue
                x = float(at[1])
                y = float(at[2])
                angle = float(at[3]) if len(at) > 3 else 0.0
                length = float(length_node[1]) if length_node else 0.0
                pins.append({
                    "number": str(number_node[1]),
                    "name": str(name_node[1]) if name_node else "",
                    "x": x,
                    "y": y,
                    "angle": angle,
                    "length": length,
                })
            elif head == "symbol":
                walk(child)  # descend into unit sub-symbols

    walk(symbol_node)
    return pins


# --------------------------------------------------------------------------
# Coordinate transforms
# --------------------------------------------------------------------------

def transform_pin(inst_x: float, inst_y: float, inst_rot: float,
                  px: float, py: float) -> Tuple[float, float]:
    """Map a symbol-local pin coordinate to schematic coordinates.

    KiCad stores symbol geometry with the Y axis pointing up, while the
    schematic canvas has Y pointing down; placement therefore flips Y and
    then rotates by the instance angle. Verified for rotation 0 (the common
    case); 90/180/270 use the standard rotation matrix.
    """
    # Y flip from library (Y-up) to schematic (Y-down)
    x, y = px, -py
    r = math.radians(inst_rot % 360)
    c, s = math.cos(r), math.sin(r)
    # Counter-clockwise rotation as seen on a Y-down canvas.
    xr = x * c + y * s
    yr = -x * s + y * c
    ax = round(inst_x + xr, 4)
    ay = round(inst_y + yr, 4)
    # normalize -0.0
    return (ax + 0.0, ay + 0.0)


# --------------------------------------------------------------------------
# Schematic construction / IO
# --------------------------------------------------------------------------

def new_uuid() -> str:
    return str(_uuid.uuid4())


def new_schematic(root_uuid: str) -> list:
    """Build a minimal but valid empty schematic tree."""
    return [
        Sym("kicad_sch"),
        [Sym("version"), Sym("20250114")],
        [Sym("generator"), "eeschema"],
        [Sym("generator_version"), "10.0"],
        [Sym("uuid"), root_uuid],
        [Sym("paper"), "A4"],
        [Sym("lib_symbols")],
        [Sym("sheet_instances"), [Sym("path"), "/", [Sym("page"), "1"]]],
    ]


def load_schematic(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return sexpr.parse(f.read())


def save_schematic(path: str, root: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(sexpr.dumps(root))
        f.write("\n")


def schematic_root_uuid(root: list) -> str:
    node = sexpr.find(root, "uuid")
    return str(node[1]) if node else ""


def ensure_lib_symbol(root: list, lib_id: str) -> List[Dict[str, Any]]:
    """Ensure ``lib_id`` is present in ``lib_symbols``; return its pins."""
    lib_symbols = sexpr.find(root, "lib_symbols")
    if lib_symbols is None:
        lib_symbols = [Sym("lib_symbols")]
        # insert after uuid/paper, before sheet_instances
        root.insert(len(root) - 1, lib_symbols)

    # already present?
    for child in sexpr.find_all(lib_symbols, "symbol"):
        if len(child) >= 2 and str(child[1]) == lib_id:
            return _collect_pins(child)

    node, pins = load_symbol(lib_id)
    lib_symbols.append(node)
    return pins


def add_component(root: list, lib_id: str, reference: str, value: str,
                  x: float, y: float, rotation: float = 0.0,
                  footprint: str = "") -> Dict[str, Any]:
    """Add a symbol instance to the schematic.

    Returns a dict describing the placed component, including the absolute
    schematic coordinates of every pin (keyed by pin number).
    """
    pins = ensure_lib_symbol(root, lib_id)
    comp_uuid = new_uuid()
    root_uuid = schematic_root_uuid(root)

    def prop(name, val, px, py, hide=False):
        p = [
            Sym("property"), name, val,
            [Sym("at"), Sym(str(px)), Sym(str(py)), Sym("0")],
        ]
        eff = [Sym("effects"), [Sym("font"), [Sym("size"), Sym("1.27"), Sym("1.27")]]]
        if hide:
            p.append([Sym("hide"), Sym("yes")])
        p.append(eff)
        return p

    symbol = [
        Sym("symbol"),
        [Sym("lib_id"), lib_id],
        [Sym("at"), Sym(str(x)), Sym(str(y)), Sym(str(int(rotation)))],
        [Sym("unit"), Sym("1")],
        [Sym("exclude_from_sim"), Sym("no")],
        [Sym("in_bom"), Sym("yes")],
        [Sym("on_board"), Sym("yes")],
        [Sym("dnp"), Sym("no")],
        [Sym("uuid"), comp_uuid],
        prop("Reference", reference, round(x + 2.54, 3), round(y - 1.27, 3)),
        prop("Value", value, round(x + 2.54, 3), round(y + 1.27, 3)),
        prop("Footprint", footprint, round(x - 1.778, 3), y, hide=True),
        prop("Datasheet", "~", x, y, hide=True),
    ]

    # pin uuids
    for pin in pins:
        symbol.append([Sym("pin"), str(pin["number"]), [Sym("uuid"), new_uuid()]])

    symbol.append([
        Sym("instances"),
        [Sym("project"), "",
         [Sym("path"), f"/{root_uuid}",
          [Sym("reference"), reference],
          [Sym("unit"), Sym("1")]]],
    ])

    # place before sheet_instances
    si_index = next(
        (i for i, c in enumerate(root)
         if isinstance(c, list) and c and str(c[0]) == "sheet_instances"),
        len(root),
    )
    root.insert(si_index, symbol)

    abs_pins = {
        pin["number"]: transform_pin(x, y, rotation, pin["x"], pin["y"])
        for pin in pins
    }
    return {
        "reference": reference,
        "lib_id": lib_id,
        "value": value,
        "position": [x, y],
        "rotation": rotation,
        "uuid": comp_uuid,
        "pins": {num: {"x": p[0], "y": p[1]} for num, p in abs_pins.items()},
    }


def add_wire(root: list, x1: float, y1: float, x2: float, y2: float) -> str:
    wire_uuid = new_uuid()
    wire = [
        Sym("wire"),
        [Sym("pts"),
         [Sym("xy"), Sym(str(x1)), Sym(str(y1))],
         [Sym("xy"), Sym(str(x2)), Sym(str(y2))]],
        [Sym("stroke"), [Sym("width"), Sym("0")], [Sym("type"), Sym("default")]],
        [Sym("uuid"), wire_uuid],
    ]
    si_index = next(
        (i for i, c in enumerate(root)
         if isinstance(c, list) and c and str(c[0]) == "sheet_instances"),
        len(root),
    )
    root.insert(si_index, wire)
    return wire_uuid


def add_label(root: list, text: str, x: float, y: float, rotation: float = 0.0) -> str:
    label_uuid = new_uuid()
    label = [
        Sym("label"), text,
        [Sym("at"), Sym(str(x)), Sym(str(y)), Sym(str(int(rotation)))],
        [Sym("effects"),
         [Sym("font"), [Sym("size"), Sym("1.27"), Sym("1.27")]],
         [Sym("justify"), Sym("left"), Sym("bottom")]],
        [Sym("uuid"), label_uuid],
    ]
    si_index = next(
        (i for i, c in enumerate(root)
         if isinstance(c, list) and c and str(c[0]) == "sheet_instances"),
        len(root),
    )
    root.insert(si_index, label)
    return label_uuid


# --------------------------------------------------------------------------
# Project / board creation
# --------------------------------------------------------------------------

def _minimal_pcb(root_uuid: str) -> str:
    """Return text for an empty but valid board (layers from a template)."""
    tmpl = _template_pcb_path()
    if tmpl:
        board = sexpr.parse(open(tmpl, "r", encoding="utf-8").read())
        keep_heads = {"version", "generator", "generator_version",
                      "general", "paper", "layers", "setup"}
        new_board = [Sym("kicad_pcb")]
        for child in board[1:]:
            if isinstance(child, list) and child and str(child[0]) in keep_heads:
                new_board.append(child)
        new_board.append([Sym("net"), Sym("0"), ""])
        return sexpr.dumps(new_board) + "\n"
    # Fallback: no template available.
    raise RuntimeError("No KiCad board template found to derive a valid PCB")


def create_project(project_dir: str, name: str, with_pcb: bool = True) -> Dict[str, Any]:
    """Create a new KiCad project (``.kicad_pro`` + ``.kicad_sch`` [+ ``.kicad_pcb``])."""
    import json

    os.makedirs(project_dir, exist_ok=True)
    pro_path = os.path.join(project_dir, f"{name}.kicad_pro")
    sch_path = os.path.join(project_dir, f"{name}.kicad_sch")
    pcb_path = os.path.join(project_dir, f"{name}.kicad_pcb")

    root_uuid = new_uuid()

    # Schematic
    save_schematic(sch_path, new_schematic(root_uuid))

    # Project file (JSON)
    pro = {
        "board": {"design_settings": {}, "layer_presets": [], "viewports": []},
        "boards": [],
        "meta": {"filename": f"{name}.kicad_pro", "version": 3},
        "net_settings": {"classes": [{"name": "Default"}]},
        "pcbnew": {"page_layout_descr_file": ""},
        "schematic": {"legacy_lib_dir": "", "legacy_lib_list": []},
        "sheets": [[root_uuid, "Root"]],
        "text_variables": {},
    }
    with open(pro_path, "w", encoding="utf-8") as f:
        json.dump(pro, f, indent=2)

    result = {
        "name": name,
        "directory": project_dir,
        "project_file": pro_path,
        "schematic_file": sch_path,
    }

    if with_pcb:
        with open(pcb_path, "w", encoding="utf-8") as f:
            f.write(_minimal_pcb(root_uuid))
        result["pcb_file"] = pcb_path

    return result
