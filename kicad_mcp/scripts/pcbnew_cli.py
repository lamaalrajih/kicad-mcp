#!/usr/bin/env python3
"""
Headless pcbnew backend, executed by KiCad's *bundled* Python interpreter.

This script is intentionally standalone (no kicad_mcp imports) so it can run
under ``/Applications/KiCad/.../python3`` where the ``pcbnew`` SWIG module is
importable. The project's venv talks to it via subprocess (see
:mod:`kicad_mcp.utils.pcbnew_runner`) and exchanges JSON.

Subcommands:
  update-from-netlist  --board PCB --input JSON [--footprint-dir DIR ...]
  export-dsn           --board PCB --output DSN
  import-ses           --board PCB --input SES

Every subcommand prints a single JSON object to stdout describing the result.
"""
import argparse
import json
import os
import sys

import pcbnew  # provided by KiCad's bundled Python


def _out(obj):
    sys.stdout.write(json.dumps(obj))
    sys.stdout.flush()


def _default_footprint_dirs():
    dirs = []
    # macOS bundle
    for base in (
        "/Applications/KiCad/KiCad.app/Contents/SharedSupport/footprints",
        "/usr/share/kicad/footprints",
        "/usr/local/share/kicad/footprints",
    ):
        if os.path.isdir(base):
            dirs.append(base)
    env = os.environ.get("KICAD_FOOTPRINT_DIR")
    if env:
        dirs = [p for p in env.split(os.pathsep) if p] + dirs
    return dirs


def _resolve_pretty(nickname, extra_dirs):
    for base in list(extra_dirs) + _default_footprint_dirs():
        cand = os.path.join(base, f"{nickname}.pretty")
        if os.path.isdir(cand):
            return cand
    return None


def cmd_update_from_netlist(args):
    with open(args.input) as f:
        data = json.load(f)

    board_path = args.board
    if os.path.exists(board_path):
        board = pcbnew.LoadBoard(board_path)
    else:
        board = pcbnew.NewBoard(board_path)

    extra_dirs = args.footprint_dir or []

    # Index existing footprints by reference (so re-running updates in place).
    existing = {fp.GetReference(): fp for fp in board.GetFootprints()}

    placed, updated, errors = [], [], []
    # simple auto-placement grid for newly added footprints
    grid_x, grid_y, step, per_row = 50.0, 50.0, 12.0, 6
    idx = 0

    for comp in data["components"]:
        ref = comp["ref"]
        fpid = comp.get("footprint") or ""
        if ":" not in fpid:
            errors.append(f"{ref}: no footprint assigned")
            continue
        nick, name = fpid.split(":", 1)
        if ref in existing:
            updated.append(ref)
            continue
        pretty = _resolve_pretty(nick, extra_dirs)
        if not pretty:
            errors.append(f"{ref}: footprint library '{nick}' not found")
            continue
        fp = pcbnew.FootprintLoad(pretty, name)
        if fp is None:
            errors.append(f"{ref}: footprint '{name}' not found in {nick}")
            continue
        fp.SetReference(ref)
        fp.SetValue(comp.get("value", ""))
        px = grid_x + (idx % per_row) * step
        py = grid_y + (idx // per_row) * step
        fp.SetPosition(pcbnew.VECTOR2I(pcbnew.FromMM(px), pcbnew.FromMM(py)))
        board.Add(fp)
        existing[ref] = fp
        placed.append({"ref": ref, "footprint": fpid, "x": px, "y": py})
        idx += 1

    # Create nets and assign to pads.
    netmap = {}
    for netname in data.get("nets", []):
        if not netname:
            continue
        ni = board.FindNet(netname)
        if ni is None:
            ni = pcbnew.NETINFO_ITEM(board, netname)
            board.Add(ni)
        netmap[netname] = ni

    assigned = 0
    for node in data.get("nodes", []):
        fp = existing.get(node["ref"])
        if fp is None:
            continue
        net = netmap.get(node["net"])
        if net is None:
            continue
        for pad in fp.Pads():
            if pad.GetNumber() == str(node["pin"]):
                pad.SetNet(net)
                assigned += 1
                break

    board.BuildListOfNets()
    pcbnew.SaveBoard(board_path, board)
    _out({
        "success": True,
        "board": board_path,
        "placed": placed,
        "updated_existing": updated,
        "pads_assigned": assigned,
        "nets": [n for n in netmap.keys()],
        "errors": errors,
    })


def cmd_add_outline(args):
    board = pcbnew.LoadBoard(args.board)
    rect = pcbnew.PCB_SHAPE(board)
    rect.SetShape(pcbnew.SHAPE_T_RECT)
    rect.SetStart(pcbnew.VECTOR2I(pcbnew.FromMM(args.x), pcbnew.FromMM(args.y)))
    rect.SetEnd(pcbnew.VECTOR2I(pcbnew.FromMM(args.x + args.width),
                                pcbnew.FromMM(args.y + args.height)))
    rect.SetLayer(pcbnew.Edge_Cuts)
    rect.SetWidth(pcbnew.FromMM(0.1))
    board.Add(rect)
    pcbnew.SaveBoard(args.board, board)
    _out({"success": True, "board": args.board,
          "outline_mm": {"x": args.x, "y": args.y,
                         "w": args.width, "h": args.height}})


def cmd_export_dsn(args):
    board = pcbnew.LoadBoard(args.board)
    ok = pcbnew.ExportSpecctraDSN(board, args.output)
    _out({"success": bool(ok), "board": args.board, "dsn": args.output,
          "exists": os.path.exists(args.output)})


def cmd_import_ses(args):
    board = pcbnew.LoadBoard(args.board)
    ok = pcbnew.ImportSpecctraSES(board, args.input)
    if ok:
        pcbnew.SaveBoard(args.board, board)
    _out({"success": bool(ok), "board": args.board, "ses": args.input})


def main():
    p = argparse.ArgumentParser(description="Headless pcbnew backend")
    sub = p.add_subparsers(dest="cmd", required=True)

    u = sub.add_parser("update-from-netlist")
    u.add_argument("--board", required=True)
    u.add_argument("--input", required=True)
    u.add_argument("--footprint-dir", action="append")
    u.set_defaults(func=cmd_update_from_netlist)

    o = sub.add_parser("add-outline")
    o.add_argument("--board", required=True)
    o.add_argument("--x", type=float, required=True)
    o.add_argument("--y", type=float, required=True)
    o.add_argument("--width", type=float, required=True)
    o.add_argument("--height", type=float, required=True)
    o.set_defaults(func=cmd_add_outline)

    e = sub.add_parser("export-dsn")
    e.add_argument("--board", required=True)
    e.add_argument("--output", required=True)
    e.set_defaults(func=cmd_export_dsn)

    i = sub.add_parser("import-ses")
    i.add_argument("--board", required=True)
    i.add_argument("--input", required=True)
    i.set_defaults(func=cmd_import_ses)

    args = p.parse_args()
    try:
        args.func(args)
    except Exception as e:  # emit structured error
        _out({"success": False, "error": f"{type(e).__name__}: {e}"})
        sys.exit(1)


if __name__ == "__main__":
    main()
