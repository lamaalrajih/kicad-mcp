"""
Utility functions for working with KiCad component values and properties.
"""

import re
from typing import Any


def extract_voltage_from_regulator(value: str) -> str:
    """Extract output voltage from a voltage regulator part number or description.

    Args:
        value: Regulator part number or description

    Returns:
        Extracted voltage as a string or "unknown" if not found
    """
    # Common patterns:
    # 78xx/79xx series: 7805 = 5V, 7812 = 12V
    # LDOs often have voltage in the part number, like LM1117-3.3

    # 78xx/79xx series
    match = re.search(r"78(\d\d)|79(\d\d)", value, re.IGNORECASE)
    if match:
        group = match.group(1) or match.group(2)
        # Convert code to voltage (e.g., 05 -> 5V, 12 -> 12V)
        try:
            voltage = int(group)
            # For 78xx series, voltage code is directly in volts
            if voltage < 50:  # Sanity check to prevent weird values
                return f"{voltage}V"
        except ValueError:
            pass

    # Look for common voltage indicators in the string
    voltage_patterns = [
        r"(\d+\.?\d*)V",  # 3.3V, 5V, etc.
        r"-(\d+\.?\d*)V",  # -5V, -12V, etc. (for negative regulators)
        r"(\d+\.?\d*)[_-]?V",  # 3.3_V, 5-V, etc.
        r"[_-](\d+\.?\d*)",  # LM1117-3.3, LD1117-3.3, etc.
    ]

    for pattern in voltage_patterns:
        match = re.search(pattern, value, re.IGNORECASE)
        if match:
            try:
                voltage = float(match.group(1))
                if 0 < voltage < 50:  # Sanity check
                    # Format as integer if it's a whole number
                    if voltage.is_integer():
                        return f"{int(voltage)}V"
                    else:
                        return f"{voltage}V"
            except ValueError:
                pass

    # Check for common fixed voltage regulators
    regulators = {
        "LM7805": "5V",
        "LM7809": "9V",
        "LM7812": "12V",
        "LM7905": "-5V",
        "LM7912": "-12V",
        "LM1117-3.3": "3.3V",
        "LM1117-5": "5V",
        "LM317": "Adjustable",
        "LM337": "Adjustable (Negative)",
        "AP1117-3.3": "3.3V",
        "AMS1117-3.3": "3.3V",
        "L7805": "5V",
        "L7812": "12V",
        "MCP1700-3.3": "3.3V",
        "MCP1700-5.0": "5V",
    }

    for reg, volt in regulators.items():
        if re.search(re.escape(reg), value, re.IGNORECASE):
            return volt

    return "unknown"


def extract_frequency_from_value(value: str) -> str:
    """Extract frequency information from a component value or description.

    Args:
        value: Component value or description (e.g., "16MHz", "Crystal 8MHz")

    Returns:
        Frequency as a string or "unknown" if not found
    """
    # Common frequency patterns with various units
    frequency_patterns = [
        r"(\d+\.?\d*)[\s-]*([kKmMgG]?)[hH][zZ]",  # 16MHz, 32.768 kHz, etc.
        r"(\d+\.?\d*)[\s-]*([kKmMgG])",  # 16M, 32.768k, etc.
    ]

    for pattern in frequency_patterns:
        match = re.search(pattern, value, re.IGNORECASE)
        if match:
            try:
                freq = float(match.group(1))
                unit = match.group(2).upper() if match.group(2) else ""

                # Make sure the frequency is in a reasonable range
                if freq > 0:
                    # Format the output
                    if unit == "K":
                        if freq >= 1000:
                            return f"{freq / 1000:.3f}MHz"
                        else:
                            return f"{freq:.3f}kHz"
                    elif unit == "M":
                        if freq >= 1000:
                            return f"{freq / 1000:.3f}GHz"
                        else:
                            return f"{freq:.3f}MHz"
                    elif unit == "G":
                        return f"{freq:.3f}GHz"
                    else:  # No unit, need to determine based on value
                        if freq < 1000:
                            return f"{freq:.3f}Hz"
                        elif freq < 1000000:
                            return f"{freq / 1000:.3f}kHz"
                        elif freq < 1000000000:
                            return f"{freq / 1000000:.3f}MHz"
                        else:
                            return f"{freq / 1000000000:.3f}GHz"
            except ValueError:
                pass

    # Check for common crystal frequencies
    if "32.768" in value or "32768" in value:
        return "32.768kHz"  # Common RTC crystal
    elif "16M" in value or "16MHZ" in value.upper():
        return "16MHz"  # Common MCU crystal
    elif "8M" in value or "8MHZ" in value.upper():
        return "8MHz"
    elif "20M" in value or "20MHZ" in value.upper():
        return "20MHz"
    elif "27M" in value or "27MHZ" in value.upper():
        return "27MHz"
    elif "25M" in value or "25MHZ" in value.upper():
        return "25MHz"

    return "unknown"


def extract_resistance_value(value: str) -> tuple[float | None, str | None]:
    """Extract resistance value and unit from component value.

    Args:
        value: Resistance value (e.g., "10k", "4.7k", "100")

    Returns:
        Tuple of (numeric value, unit) or (None, None) if parsing fails
    """
    # Common resistance patterns
    # 10k, 4.7k, 100R, 1M, 10, etc.
    match = re.search(r"(\d+\.?\d*)([kKmMrRΩ]?)", value)
    if match:
        try:
            resistance = float(match.group(1))
            unit = match.group(2).upper() if match.group(2) else "Ω"

            # Normalize unit
            if unit == "R" or unit == "":
                unit = "Ω"

            return resistance, unit
        except ValueError:
            pass

    # Handle special case like "4k7" (means 4.7k)
    match = re.search(r"(\d+)[kKmM](\d+)", value)
    if match:
        try:
            value1 = int(match.group(1))
            value2 = int(match.group(2))
            resistance = float(f"{value1}.{value2}")
            unit = "k" if "k" in value.lower() else "M" if "m" in value.lower() else "Ω"

            return resistance, unit
        except ValueError:
            pass

    return None, None


def extract_capacitance_value(value: str) -> tuple[float | None, str | None]:
    """Extract capacitance value and unit from component value.

    Args:
        value: Capacitance value (e.g., "10uF", "4.7nF", "100pF")

    Returns:
        Tuple of (numeric value, unit) or (None, None) if parsing fails
    """
    # Common capacitance patterns
    # 10uF, 4.7nF, 100pF, etc.
    match = re.search(r"(\d+\.?\d*)([pPnNuUμF]+)", value)
    if match:
        try:
            capacitance = float(match.group(1))
            unit = match.group(2).lower()

            # Normalize unit
            if "p" in unit or "pf" in unit:
                unit = "pF"
            elif "n" in unit or "nf" in unit:
                unit = "nF"
            elif "u" in unit or "μ" in unit or "uf" in unit or "μf" in unit:
                unit = "μF"
            else:
                unit = "F"

            return capacitance, unit
        except ValueError:
            pass

    # Handle special case like "4n7" (means 4.7nF)
    match = re.search(r"(\d+)[pPnNuUμ](\d+)", value)
    if match:
        try:
            value1 = int(match.group(1))
            value2 = int(match.group(2))
            capacitance = float(f"{value1}.{value2}")

            if "p" in value.lower():
                unit = "pF"
            elif "n" in value.lower():
                unit = "nF"
            elif "u" in value.lower() or "μ" in value:
                unit = "μF"
            else:
                unit = "F"

            return capacitance, unit
        except ValueError:
            pass

    return None, None


def extract_inductance_value(value: str) -> tuple[float | None, str | None]:
    """Extract inductance value and unit from component value.

    Args:
        value: Inductance value (e.g., "10uH", "4.7nH", "100mH")

    Returns:
        Tuple of (numeric value, unit) or (None, None) if parsing fails
    """
    # Common inductance patterns
    # 10uH, 4.7nH, 100mH, etc.
    match = re.search(r"(\d+\.?\d*)([pPnNuUμmM][hH])", value)
    if match:
        try:
            inductance = float(match.group(1))
            unit = match.group(2).lower()

            # Normalize unit
            if "p" in unit:
                unit = "pH"
            elif "n" in unit:
                unit = "nH"
            elif "u" in unit or "μ" in unit:
                unit = "μH"
            elif "m" in unit:
                unit = "mH"
            else:
                unit = "H"

            return inductance, unit
        except ValueError:
            pass

    # Handle special case like "4u7" (means 4.7uH)
    match = re.search(r"(\d+)[pPnNuUμmM](\d+)[hH]", value)
    if match:
        try:
            value1 = int(match.group(1))
            value2 = int(match.group(2))
            inductance = float(f"{value1}.{value2}")

            if "p" in value.lower():
                unit = "pH"
            elif "n" in value.lower():
                unit = "nH"
            elif "u" in value.lower() or "μ" in value:
                unit = "μH"
            elif "m" in value.lower():
                unit = "mH"
            else:
                unit = "H"

            return inductance, unit
        except ValueError:
            pass

    return None, None


def format_resistance(resistance: float, unit: str) -> str:
    """Format resistance value with appropriate unit.

    Args:
        resistance: Resistance value
        unit: Unit string (Ω, k, M)

    Returns:
        Formatted resistance string
    """
    if unit == "Ω":
        return f"{resistance:.0f}Ω" if resistance.is_integer() else f"{resistance}Ω"
    elif unit == "k":
        return f"{resistance:.0f}kΩ" if resistance.is_integer() else f"{resistance}kΩ"
    elif unit == "M":
        return f"{resistance:.0f}MΩ" if resistance.is_integer() else f"{resistance}MΩ"
    else:
        return f"{resistance}{unit}"


def format_capacitance(capacitance: float, unit: str) -> str:
    """Format capacitance value with appropriate unit.

    Args:
        capacitance: Capacitance value
        unit: Unit string (pF, nF, μF, F)

    Returns:
        Formatted capacitance string
    """
    if capacitance.is_integer():
        return f"{int(capacitance)}{unit}"
    else:
        return f"{capacitance}{unit}"


def format_inductance(inductance: float, unit: str) -> str:
    """Format inductance value with appropriate unit.

    Args:
        inductance: Inductance value
        unit: Unit string (pH, nH, μH, mH, H)

    Returns:
        Formatted inductance string
    """
    if inductance.is_integer():
        return f"{int(inductance)}{unit}"
    else:
        return f"{inductance}{unit}"


def normalize_component_value(value: str, component_type: str) -> str:
    """Normalize a component value string based on component type.

    Args:
        value: Raw component value string
        component_type: Type of component (R, C, L, etc.)

    Returns:
        Normalized value string
    """
    if component_type == "R":
        resistance, unit = extract_resistance_value(value)
        if resistance is not None and unit is not None:
            return format_resistance(resistance, unit)
    elif component_type == "C":
        capacitance, unit = extract_capacitance_value(value)
        if capacitance is not None and unit is not None:
            return format_capacitance(capacitance, unit)
    elif component_type == "L":
        inductance, unit = extract_inductance_value(value)
        if inductance is not None and unit is not None:
            return format_inductance(inductance, unit)

    # For other component types or if parsing fails, return the original value
    return value


def get_component_type_from_reference(reference: str) -> str:
    """Determine component type from reference designator.

    Args:
        reference: Component reference (e.g., R1, C2, U3)

    Returns:
        Component type letter (R, C, L, Q, etc.)
    """
    # Extract the alphabetic prefix (component type)
    match = re.match(r"^([A-Za-z_]+)", reference)
    if match:
        return match.group(1)
    return ""


def is_power_component(component: dict[str, Any]) -> bool:
    """Check if a component is likely a power-related component.

    Args:
        component: Component information dictionary

    Returns:
        True if the component is power-related, False otherwise
    """
    ref = component.get("reference", "")
    value = component.get("value", "").upper()
    lib_id = component.get("lib_id", "").upper()

    # Check reference designator
    if ref.startswith(("VR", "PS", "REG")):
        return True

    # Check for power-related terms in value or library ID
    power_terms = ["VCC", "VDD", "GND", "POWER", "PWR", "SUPPLY", "REGULATOR", "LDO"]
    if any(term in value or term in lib_id for term in power_terms):
        return True

    # Check for regulator part numbers
    regulator_patterns = [
        r"78\d\d",  # 7805, 7812, etc.
        r"79\d\d",  # 7905, 7912, etc.
        r"LM\d{3}",  # LM317, LM337, etc.
        r"LM\d{4}",  # LM1117, etc.
        r"AMS\d{4}",  # AMS1117, etc.
        r"MCP\d{4}",  # MCP1700, etc.
    ]

    return any(re.search(pattern, value, re.IGNORECASE) for pattern in regulator_patterns)
