"""
Common circuit templates, built with the schematic authoring primitives.

Each template adds parts and net labels to an existing schematic. Components
are connected by placing identically-named net labels on their pins (a label
is a connection point in KiCad), so no manual wiring math is needed.
"""
from typing import Dict, Any, List, Callable

from kicad_mcp.utils import sexpr_author as author

R_FP = "Resistor_SMD:R_0603_1608Metric"
C_FP = "Capacitor_SMD:C_0603_1608Metric"
LED_FP = "LED_SMD:LED_0603_1608Metric"


def _label_pin(root, comp, pin, net):
    p = comp["pins"][pin]
    author.add_label(root, net, p["x"], p["y"])


def _voltage_divider(root, params):
    r1 = params.get("r1", "10k")
    r2 = params.get("r2", "10k")
    vin, vout, gnd = params.get("vin", "VIN"), params.get("vout", "VOUT"), params.get("gnd", "GND")
    c1 = author.add_component(root, "Device:R", "R1", r1, 120.0, 60.0, footprint=R_FP)
    c2 = author.add_component(root, "Device:R", "R2", r2, 120.0, 80.0, footprint=R_FP)
    _label_pin(root, c1, "1", vin)
    _label_pin(root, c1, "2", vout)
    _label_pin(root, c2, "1", vout)
    _label_pin(root, c2, "2", gnd)
    return {"components": ["R1", "R2"], "nets": [vin, vout, gnd]}


def _rc_lowpass(root, params):
    r = params.get("r", "1k")
    c = params.get("c", "100nF")
    vin, vout, gnd = params.get("vin", "VIN"), params.get("vout", "VOUT"), params.get("gnd", "GND")
    c1 = author.add_component(root, "Device:R", "R1", r, 120.0, 60.0, footprint=R_FP)
    c2 = author.add_component(root, "Device:C", "C1", c, 135.0, 72.0, footprint=C_FP)
    _label_pin(root, c1, "1", vin)
    _label_pin(root, c1, "2", vout)
    _label_pin(root, c2, "1", vout)
    _label_pin(root, c2, "2", gnd)
    return {"components": ["R1", "C1"], "nets": [vin, vout, gnd]}


def _led_indicator(root, params):
    r = params.get("r", "330")
    vcc, gnd = params.get("vcc", "VCC"), params.get("gnd", "GND")
    c1 = author.add_component(root, "Device:R", "R1", r, 120.0, 60.0, footprint=R_FP)
    d1 = author.add_component(root, "Device:LED", "D1", params.get("color", "RED"),
                              120.0, 80.0, footprint=LED_FP)
    _label_pin(root, c1, "1", vcc)
    _label_pin(root, c1, "2", "LED_A")
    _label_pin(root, d1, "1", "LED_A")
    _label_pin(root, d1, "2", gnd)
    return {"components": ["R1", "D1"], "nets": [vcc, "LED_A", gnd]}


def _decoupling(root, params):
    count = int(params.get("count", 2))
    value = params.get("value", "100nF")
    vcc, gnd = params.get("vcc", "VCC"), params.get("gnd", "GND")
    refs = []
    x = 110.0
    for i in range(count):
        ref = f"C{i + 1}"
        c = author.add_component(root, "Device:C", ref, value, x, 70.0, footprint=C_FP)
        _label_pin(root, c, "1", vcc)
        _label_pin(root, c, "2", gnd)
        refs.append(ref)
        x += 12.0
    return {"components": refs, "nets": [vcc, gnd]}


TEMPLATES: Dict[str, Callable] = {
    "voltage_divider": _voltage_divider,
    "rc_lowpass": _rc_lowpass,
    "led_indicator": _led_indicator,
    "decoupling": _decoupling,
}

TEMPLATE_HELP = {
    "voltage_divider": "Two resistors R1/R2 in series (params: r1, r2, vin, vout, gnd)",
    "rc_lowpass": "RC low-pass filter R1 + C1 (params: r, c, vin, vout, gnd)",
    "led_indicator": "LED with series resistor (params: r, color, vcc, gnd)",
    "decoupling": "N decoupling caps across VCC/GND (params: count, value, vcc, gnd)",
}


def apply_template(schematic_path: str, template: str,
                   params: Dict[str, Any] = None) -> Dict[str, Any]:
    if template not in TEMPLATES:
        raise ValueError(
            f"Unknown template '{template}'. Available: {sorted(TEMPLATES)}"
        )
    root = author.load_schematic(schematic_path)
    summary = TEMPLATES[template](root, params or {})
    author.save_schematic(schematic_path, root)
    summary["template"] = template
    summary["schematic_path"] = schematic_path
    return summary
