"""
Tests for the S-expression reader/writer and schematic authoring helpers.
"""
import os
import tempfile

import pytest

from kicad_mcp.utils import sexpr
from kicad_mcp.utils.sexpr import Sym
from kicad_mcp.utils import sexpr_author as author


class TestSexpr:
    def test_roundtrip_simple(self):
        text = '(kicad_sch (version 20250114) (paper "A4"))'
        tree = sexpr.parse(text)
        assert str(tree[0]) == "kicad_sch"
        # re-parsing the serialized form yields an equivalent tree
        again = sexpr.parse(sexpr.dumps(tree))
        assert sexpr.dumps(again) == sexpr.dumps(tree)

    def test_quoted_string_preserved(self):
        tree = sexpr.parse('(label "VOUT net")')
        assert tree[1] == "VOUT net"
        assert '"VOUT net"' in sexpr.dumps(tree)

    def test_escape_quotes(self):
        tree = sexpr.parse(r'(text "he said \"hi\"")')
        assert tree[1] == 'he said "hi"'

    def test_find_helpers(self):
        tree = sexpr.parse('(root (a 1) (b 2) (b 3))')
        assert sexpr.find(tree, "a")[1] == Sym("1")
        assert len(sexpr.find_all(tree, "b")) == 2


class TestTransform:
    def test_rotation_zero_flips_y(self):
        # library Y-up -> schematic Y-down: (0, 3.81) maps to Y - 3.81
        x, y = author.transform_pin(127.0, 63.5, 0, 0.0, 3.81)
        assert x == pytest.approx(127.0)
        assert y == pytest.approx(59.69)

    def test_rotation_180(self):
        x, y = author.transform_pin(100.0, 100.0, 180, 0.0, 3.81)
        assert x == pytest.approx(100.0)
        assert y == pytest.approx(103.81)


class TestSchematicBuild:
    def test_new_schematic_is_parseable(self):
        root = author.new_schematic(author.new_uuid())
        assert str(root[0]) == "kicad_sch"
        assert sexpr.find(root, "lib_symbols") is not None
        # serialize + reparse
        again = sexpr.parse(sexpr.dumps(root))
        assert str(again[0]) == "kicad_sch"

    def test_add_wire_and_label_without_libs(self):
        root = author.new_schematic(author.new_uuid())
        author.add_wire(root, 0.0, 0.0, 10.0, 0.0)
        author.add_label(root, "NET1", 0.0, 0.0)
        assert sexpr.find(root, "wire") is not None
        assert sexpr.find(root, "label") is not None
        # wires/labels inserted before sheet_instances
        heads = [str(c[0]) for c in root if isinstance(c, list)]
        assert heads.index("wire") < heads.index("sheet_instances")


# The following require the installed KiCad symbol libraries.
_HAS_SYMBOLS = len(author.symbol_library_dirs()) > 0


@pytest.mark.skipif(not _HAS_SYMBOLS, reason="KiCad symbol libraries not installed")
class TestWithSymbols:
    def test_load_resistor_symbol(self):
        node, pins = author.load_symbol("Device:R")
        assert node[1] == "Device:R"
        numbers = {p["number"] for p in pins}
        assert numbers == {"1", "2"}

    def test_add_component_reports_pins(self):
        root = author.new_schematic(author.new_uuid())
        comp = author.add_component(root, "Device:R", "R1", "10k", 127.0, 63.5)
        assert set(comp["pins"].keys()) == {"1", "2"}
        assert comp["pins"]["1"]["y"] == pytest.approx(59.69)
        assert comp["pins"]["2"]["y"] == pytest.approx(67.31)

    def test_create_project_files(self):
        with tempfile.TemporaryDirectory() as d:
            info = author.create_project(d, "proj", with_pcb=True)
            assert os.path.isfile(info["project_file"])
            assert os.path.isfile(info["schematic_file"])
            assert os.path.isfile(info["pcb_file"])

    def test_missing_symbol_raises(self):
        with pytest.raises(author.SymbolNotFound):
            author.load_symbol("Device:NoSuchSymbolXYZ")
