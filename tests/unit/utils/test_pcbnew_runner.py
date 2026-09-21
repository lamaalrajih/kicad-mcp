"""
Tests for the pcbnew bridge's pure-Python parts (netlist parsing).

The pcbnew-backed operations require KiCad's bundled interpreter and are not
exercised here; this covers the netlist (.net) -> JSON conversion.
"""
import os
import tempfile

from kicad_mcp.utils import pcbnew_runner as pr


SAMPLE_NET = '''(export (version "E")
  (components
    (comp (ref "R1") (value "1k") (footprint "Resistor_SMD:R_0603_1608Metric"))
    (comp (ref "C1") (value "100nF") (footprint "Capacitor_SMD:C_0603_1608Metric")))
  (nets
    (net (code "1") (name "/VIN")
      (node (ref "R1") (pin "1")))
    (net (code "2") (name "/VOUT")
      (node (ref "R1") (pin "2"))
      (node (ref "C1") (pin "1")))
    (net (code "3") (name "/GND")
      (node (ref "C1") (pin "2")))))
'''


def test_parse_kicad_netlist():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "n.net")
        with open(p, "w") as f:
            f.write(SAMPLE_NET)
        data = pr.parse_kicad_netlist(p)

    refs = {c["ref"] for c in data["components"]}
    assert refs == {"R1", "C1"}
    fps = {c["ref"]: c["footprint"] for c in data["components"]}
    assert fps["R1"] == "Resistor_SMD:R_0603_1608Metric"

    assert set(data["nets"]) == {"/VIN", "/VOUT", "/GND"}
    # VOUT ties R1.2 and C1.1
    vout = {(n["ref"], n["pin"]) for n in data["nodes"] if n["net"] == "/VOUT"}
    assert vout == {("R1", "2"), ("C1", "1")}


def test_find_kicad_python_returns_str_or_none():
    result = pr.find_kicad_python()
    assert result is None or isinstance(result, str)
