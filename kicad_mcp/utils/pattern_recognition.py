"""
Circuit pattern recognition functions for KiCad schematics.
"""

import re
from typing import Dict, List, Any
from kicad_mcp.utils.component_utils import extract_voltage_from_regulator, extract_frequency_from_value

def identify_power_supplies(components: Dict[str, Any], nets: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Identify power supply circuits in the schematic.
    
    Args:
        components: Dictionary of components from netlist
        nets: Dictionary of nets from netlist
        
    Returns:
        List of identified power supply circuits
    """
    power_supplies = []
    
    # Look for voltage regulators (Linear)
    regulator_patterns = {
        "78xx": r"78\d\d|LM78\d\d|MC78\d\d",  # 7805, 7812, etc.
        "79xx": r"79\d\d|LM79\d\d|MC79\d\d",  # 7905, 7912, etc.
        "LDO": r"LM\d{3}|LD\d{3}|AMS\d{4}|LT\d{4}|TLV\d{3}|AP\d{4}|MIC\d{4}|NCP\d{3}|LP\d{4}|L\d{2}|TPS\d{5}"
    }
    
    for ref, component in components.items():
        # Check for voltage regulators by part value or lib_id
        component_value = component.get('value', '').upper()
        component_lib = component.get('lib_id', '').upper()
        
        for reg_type, pattern in regulator_patterns.items():
            if re.search(pattern, component_value, re.IGNORECASE) or re.search(pattern, component_lib, re.IGNORECASE):
                # Found a regulator, look for associated components
                power_supplies.append({
                    "type": "linear_regulator",
                    "subtype": reg_type,
                    "main_component": ref,
                    "value": component_value,
                    "input_voltage": "unknown",  # Would need more analysis to determine
                    "output_voltage": extract_voltage_from_regulator(component_value),
                    "associated_components": []  # Would need connection analysis to find these
                })
    
    # Look for switching regulators
    switching_patterns = {
        "buck": r"LM\d{4}|TPS\d{4}|MP\d{4}|RT\d{4}|LT\d{4}|MC\d{4}|NCP\d{4}|TL\d{4}|LTC\d{4}",
        "boost": r"MC\d{4}|LT\d{4}|TPS\d{4}|MAX\d{4}|NCP\d{4}|LTC\d{4}",
        "buck_boost": r"LTC\d{4}|LM\d{4}|TPS\d{4}|MAX\d{4}"
    }
    
    for ref, component in components.items():
        component_value = component.get('value', '').upper()
        component_lib = component.get('lib_id', '').upper()
        
        # Check for inductor (key component in switching supplies)
        if ref.startswith('L') or 'Inductor' in component_lib:
            # Look for nearby ICs that might be switching controllers
            for ic_ref, ic_component in components.items():
                if ic_ref.startswith('U') or ic_ref.startswith('IC'):
                    ic_value = ic_component.get('value', '').upper()
                    ic_lib = ic_component.get('lib_id', '').upper()
                    
                    for converter_type, pattern in switching_patterns.items():
                        if re.search(pattern, ic_value, re.IGNORECASE) or re.search(pattern, ic_lib, re.IGNORECASE):
                            power_supplies.append({
                                "type": "switching_regulator",
                                "subtype": converter_type,
                                "main_component": ic_ref,
                                "inductor": ref,
                                "value": ic_value
                            })
    
    return power_supplies


def identify_amplifiers(components: Dict[str, Any], nets: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Identify amplifier circuits in the schematic.
    
    Args:
        components: Dictionary of components from netlist
        nets: Dictionary of nets from netlist
        
    Returns:
        List of identified amplifier circuits
    """
    amplifiers = []
    
    # Look for op-amps
    opamp_patterns = [
        r"LM\d{3}|TL\d{3}|NE\d{3}|LF\d{3}|OP\d{2}|MCP\d{3}|AD\d{3}|LT\d{4}|OPA\d{3}",
        r"Opamp|Op-Amp|OpAmp|Operational Amplifier"
    ]
    
    for ref, component in components.items():
        component_value = component.get('value', '').upper()
        component_lib = component.get('lib_id', '').upper()
        
        # Check for op-amps
        for pattern in opamp_patterns:
            if re.search(pattern, component_value, re.IGNORECASE) or re.search(pattern, component_lib, re.IGNORECASE):
                # Common op-amps
                if re.search(r"LM358|LM324|TL072|TL082|NE5532|LF353|MCP6002|AD8620|OPA2134", component_value, re.IGNORECASE):
                    amplifiers.append({
                        "type": "operational_amplifier",
                        "subtype": "general_purpose",
                        "component": ref,
                        "value": component_value
                    })
                # Audio op-amps
                elif re.search(r"NE5534|OPA134|OPA1612|OPA1652|LM4562|LME49720|LME49860|TL071|TL072", component_value, re.IGNORECASE):
                    amplifiers.append({
                        "type": "operational_amplifier",
                        "subtype": "audio",
                        "component": ref,
                        "value": component_value
                    })
                # Instrumentation amplifiers
                elif re.search(r"INA\d{3}|AD620|AD8221|AD8429|LT1167", component_value, re.IGNORECASE):
                    amplifiers.append({
                        "type": "operational_amplifier",
                        "subtype": "instrumentation",
                        "component": ref,
                        "value": component_value
                    })
                else:
                    amplifiers.append({
                        "type": "operational_amplifier",
                        "subtype": "unknown",
                        "component": ref,
                        "value": component_value
                    })
    
    # Look for transistor amplifiers
    transistor_refs = [ref for ref in components.keys() if ref.startswith('Q')]
    
    for ref in transistor_refs:
        component = components[ref]
        component_lib = component.get('lib_id', '').upper()
        
        # Check if it's a BJT or FET
        if 'BJT' in component_lib or 'NPN' in component_lib or 'PNP' in component_lib:
            # Look for resistors connected to transistor (biasing network)
            has_biasing = False
            for net_name, pins in nets.items():
                # Check if this net connects to our transistor
                if any(pin.get('component') == ref for pin in pins):
                    # Check if the net also connects to resistors
                    if any(pin.get('component', '').startswith('R') for pin in pins):
                        has_biasing = True
                        break
            
            if has_biasing:
                amplifiers.append({
                    "type": "transistor_amplifier",
                    "subtype": "BJT",
                    "component": ref,
                    "value": component.get('value', '')
                })
        
        elif 'FET' in component_lib or 'MOSFET' in component_lib or 'JFET' in component_lib:
            # Similar check for FET amplifiers
            has_biasing = False
            for net_name, pins in nets.items():
                if any(pin.get('component') == ref for pin in pins):
                    if any(pin.get('component', '').startswith('R') for pin in pins):
                        has_biasing = True
                        break
            
            if has_biasing:
                amplifiers.append({
                    "type": "transistor_amplifier",
                    "subtype": "FET",
                    "component": ref,
                    "value": component.get('value', '')
                })
    
    # Look for audio amplifier ICs
    audio_amp_patterns = [
        r"LM386|LM383|LM380|LM1875|LM3886|TDA\d{4}|TPA\d{4}|SSM\d{4}|PAM\d{4}|TAS\d{4}"
    ]
    
    for ref, component in components.items():
        component_value = component.get('value', '').upper()
        component_lib = component.get('lib_id', '').upper()
        
        for pattern in audio_amp_patterns:
            if re.search(pattern, component_value, re.IGNORECASE) or re.search(pattern, component_lib, re.IGNORECASE):
                amplifiers.append({
                    "type": "audio_amplifier_ic",
                    "component": ref,
                    "value": component_value
                })
    
    return amplifiers


def identify_filters(components: Dict[str, Any], nets: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Identify filter circuits in the schematic.
    
    Args:
        components: Dictionary of components from netlist
        nets: Dictionary of nets from netlist
        
    Returns:
        List of identified filter circuits
    """
    filters = []
    
    # Look for RC low-pass filters
    # These typically have a resistor followed by a capacitor to ground
    resistor_refs = [ref for ref in components.keys() if ref.startswith('R')]
    capacitor_refs = [ref for ref in components.keys() if ref.startswith('C')]
    
    for r_ref in resistor_refs:
        r_nets = []
        # Find which nets this resistor is connected to
        for net_name, pins in nets.items():
            if any(pin.get('component') == r_ref for pin in pins):
                r_nets.append(net_name)
        
        # For each net, check if there's a capacitor connected to it
        for net_name in r_nets:
            # Find capacitors connected to this net
            connected_caps = []
            for pin in nets.get(net_name, []):
                comp = pin.get('component')
                if comp and comp.startswith('C'):
                    connected_caps.append(comp)
            
            if connected_caps:
                # Check if the other side of the capacitor goes to ground
                for c_ref in connected_caps:
                    c_is_to_ground = False
                    for gnd_name in ['GND', 'AGND', 'DGND', 'VSS']:
                        for pin in nets.get(gnd_name, []):
                            if pin.get('component') == c_ref:
                                c_is_to_ground = True
                                break
                        if c_is_to_ground:
                            break
                    
                    if c_is_to_ground:
                        filters.append({
                            "type": "passive_filter",
                            "subtype": "rc_low_pass",
                            "components": [r_ref, c_ref]
                        })
    
    # Look for active filters (op-amp with feedback RC components)
    opamp_refs = []
    
    for ref, component in components.items():
        component_value = component.get('value', '').upper()
        component_lib = component.get('lib_id', '').upper()
        
        if re.search(r"LM\d{3}|TL\d{3}|NE\d{3}|LF\d{3}|OP\d{2}|MCP\d{3}|AD\d{3}|LT\d{4}|OPA\d{3}", 
                     component_value, re.IGNORECASE) or "OP_AMP" in component_lib:
            opamp_refs.append(ref)
    
    for op_ref in opamp_refs:
        # Find op-amp output
        # In a full implementation, we'd know which pin is the output
        # For simplicity, we'll look for feedback components
        has_feedback_r = False
        has_feedback_c = False
        
        for net_name, pins in nets.items():
            # If this net connects to our op-amp
            if any(pin.get('component') == op_ref for pin in pins):
                # Check if it also connects to resistors and capacitors
                connects_to_r = any(pin.get('component', '').startswith('R') for pin in pins)
                connects_to_c = any(pin.get('component', '').startswith('C') for pin in pins)
                
                if connects_to_r:
                    has_feedback_r = True
                if connects_to_c:
                    has_feedback_c = True
        
        if has_feedback_r and has_feedback_c:
            filters.append({
                "type": "active_filter",
                "main_component": op_ref,
                "value": components[op_ref].get('value', '')
            })
    
    # Look for crystal filters or ceramic filters
    for ref, component in components.items():
        component_value = component.get('value', '').upper()
        component_lib = component.get('lib_id', '').upper()
        
        if ref.startswith('Y') or ref.startswith('X') or "CRYSTAL" in component_lib or "XTAL" in component_lib:
            filters.append({
                "type": "crystal_filter",
                "component": ref,
                "value": component_value
            })
        
        if "FILTER" in component_lib or "MURATA" in component_lib or "CERAMIC_FILTER" in component_lib:
            filters.append({
                "type": "ceramic_filter",
                "component": ref,
                "value": component_value
            })
    
    return filters


def identify_oscillators(components: Dict[str, Any], nets: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Identify oscillator circuits in the schematic.
    
    Args:
        components: Dictionary of components from netlist
        nets: Dictionary of nets from netlist
        
    Returns:
        List of identified oscillator circuits
    """
    oscillators = []
    
    # Look for crystal oscillators
    for ref, component in components.items():
        component_value = component.get('value', '').upper()
        component_lib = component.get('lib_id', '').upper()
        
        # Crystals
        if ref.startswith('Y') or ref.startswith('X') or "CRYSTAL" in component_lib or "XTAL" in component_lib:
            # Check if the crystal has load capacitors
            has_load_caps = False
            crystal_nets = []
            
            for net_name, pins in nets.items():
                if any(pin.get('component') == ref for pin in pins):
                    crystal_nets.append(net_name)
            
            # Look for capacitors connected to the crystal nets
            for net_name in crystal_nets:
                for pin in nets.get(net_name, []):
                    comp = pin.get('component')
                    if comp and comp.startswith('C'):
                        has_load_caps = True
                        break
                if has_load_caps:
                    break
            
            oscillators.append({
                "type": "crystal_oscillator",
                "component": ref,
                "value": component_value,
                "frequency": extract_frequency_from_value(component_value),
                "has_load_capacitors": has_load_caps
            })
        
        # Oscillator ICs
        if "OSC" in component_lib or "OSCILLATOR" in component_lib or re.search(r"OSC|OSCILLATOR", component_value, re.IGNORECASE):
            oscillators.append({
                "type": "oscillator_ic",
                "component": ref,
                "value": component_value,
                "frequency": extract_frequency_from_value(component_value)
            })
        
        # RC oscillators (555 timer, etc)
        if re.search(r"NE555|LM555|ICM7555|TLC555", component_value, re.IGNORECASE) or "555" in component_lib:
            oscillators.append({
                "type": "rc_oscillator",
                "subtype": "555_timer",
                "component": ref,
                "value": component_value
            })
    
    return oscillators


def identify_digital_interfaces(components: Dict[str, Any], nets: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Identify digital interface circuits in the schematic.
    
    Args:
        components: Dictionary of components from netlist
        nets: Dictionary of nets from netlist
        
    Returns:
        List of identified digital interface circuits
    """
    interfaces = []
    
    # I2C interface detection
    i2c_signals = {"SCL", "SDA", "I2C_SCL", "I2C_SDA"}
    has_i2c = False
    
    for net_name in nets.keys():
        if any(signal in net_name.upper() for signal in i2c_signals):
            has_i2c = True
            break
    
    if has_i2c:
        interfaces.append({
            "type": "i2c_interface",
            "signals_found": [net for net in nets.keys() if any(signal in net.upper() for signal in i2c_signals)]
        })
    
    # SPI interface detection
    spi_signals = {"MOSI", "MISO", "SCK", "SS", "SPI_MOSI", "SPI_MISO", "SPI_SCK", "SPI_CS"}
    has_spi = False
    
    for net_name in nets.keys():
        if any(signal in net_name.upper() for signal in spi_signals):
            has_spi = True
            break
    
    if has_spi:
        interfaces.append({
            "type": "spi_interface",
            "signals_found": [net for net in nets.keys() if any(signal in net.upper() for signal in spi_signals)]
        })
    
    # UART interface detection
    uart_signals = {"TX", "RX", "TXD", "RXD", "UART_TX", "UART_RX"}
    has_uart = False
    
    for net_name in nets.keys():
        if any(signal in net_name.upper() for signal in uart_signals):
            has_uart = True
            break
    
    if has_uart:
        interfaces.append({
            "type": "uart_interface",
            "signals_found": [net for net in nets.keys() if any(signal in net.upper() for signal in uart_signals)]
        })
    
    # USB interface detection
    usb_signals = {"USB_D+", "USB_D-", "USB_DP", "USB_DM", "D+", "D-", "DP", "DM", "VBUS"}
    has_usb = False
    
    for net_name in nets.keys():
        if any(signal in net_name.upper() for signal in usb_signals):
            has_usb = True
            break
    
    # Also check for USB interface ICs
    for ref, component in components.items():
        component_value = component.get('value', '').upper()
        if re.search(r"FT232|CH340|CP210|MCP2200|TUSB|FT231|FT201", component_value, re.IGNORECASE):
            has_usb = True
            break
    
    if has_usb:
        interfaces.append({
            "type": "usb_interface",
            "signals_found": [net for net in nets.keys() if any(signal in net.upper() for signal in usb_signals)]
        })
    
    # Ethernet interface detection
    ethernet_signals = {"TX+", "TX-", "RX+", "RX-", "MDI", "MDIO", "ETH"}
    has_ethernet = False
    
    for net_name in nets.keys():
        if any(signal in net_name.upper() for signal in ethernet_signals):
            has_ethernet = True
            break
    
    # Also check for Ethernet PHY ICs
    for ref, component in components.items():
        component_value = component.get('value', '').upper()
        if re.search(r"W5500|ENC28J60|LAN87|KSZ80|DP83|RTL8|AX88", component_value, re.IGNORECASE):
            has_ethernet = True
            break
    
    if has_ethernet:
        interfaces.append({
            "type": "ethernet_interface",
            "signals_found": [net for net in nets.keys() if any(signal in net.upper() for signal in ethernet_signals)]
        })
    
    return interfaces


def identify_sensor_interfaces(components: Dict[str, Any], nets: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Identify sensor interface circuits in the schematic.
    
    Args:
        components: Dictionary of components from netlist
        nets: Dictionary of nets from netlist
        
    Returns:
        List of identified sensor interface circuits
    """
    sensor_interfaces = []
    
    # Common sensor IC patterns
    sensor_patterns = {
        "temperature": r"LM35|DS18B20|DHT11|DHT22|BME280|BMP280|TMP\d+|MCP9808|MAX31855|MAX6675|SI7021|HTU21|SHT[0123]\d|PCT2075",
        "humidity": r"DHT11|DHT22|BME280|SI7021|HTU21|SHT[0123]\d|HDC1080",
        "pressure": r"BMP\d+|BME280|LPS\d+|MS5611|DPS310|MPL3115|SPL06",
        "accelerometer": r"ADXL\d+|LIS3DH|MMA\d+|MPU\d+|LSM\d+|BMI\d+|BMA\d+|KX\d+",
        "gyroscope": r"L3G\d+|MPU\d+|BMI\d+|LSM\d+|ICM\d+",
        "magnetometer": r"HMC\d+|QMC\d+|LSM\d+|MMC\d+|RM\d+",
        "proximity": r"APDS9960|VL53L0X|VL6180|GP2Y|VCNL4040|VCNL4010",
        "light": r"BH1750|TSL\d+|MAX4\d+|VEML\d+|APDS9960|LTR329|OPT\d+",
        "air_quality": r"CCS811|BME680|SGP\d+|SEN\d+|MQ\d+|MiCS",
        "current": r"ACS\d+|INA\d+|MAX\d+|ZXCT\d+",
        "voltage": r"INA\d+|MCP\d+|ADS\d+",
        "ADC": r"ADS\d+|MCP33\d+|MCP32\d+|LTC\d+|NAU7802|HX711",
        "GPS": r"NEO-[67]M|L80|MTK\d+|SIM\d+|SAM-M8Q|MAX-M8"
    }
    
    for ref, component in components.items():
        component_value = component.get('value', '').upper()
        component_lib = component.get('lib_id', '').upper()
        
        for sensor_type, pattern in sensor_patterns.items():
            if re.search(pattern, component_value, re.IGNORECASE) or re.search(pattern, component_lib, re.IGNORECASE):
                # Identify specific sensors
                
                # Temperature sensors
                if sensor_type == "temperature":
                    if re.search(r"DS18B20", component_value, re.IGNORECASE):
                        sensor_interfaces.append({
                            "type": "temperature_sensor",
                            "model": "DS18B20",
                            "component": ref,
                            "interface": "1-Wire",
                            "range": "-55°C to +125°C"
                        })
                    elif re.search(r"BME280|BMP280", component_value, re.IGNORECASE):
                        sensor_interfaces.append({
                            "type": "multi_sensor",
                            "model": component_value,
                            "component": ref,
                            "measures": ["temperature", "pressure", "humidity" if "BME" in component_value else "pressure"],
                            "interface": "I2C/SPI"
                        })
                    elif re.search(r"LM35", component_value, re.IGNORECASE):
                        sensor_interfaces.append({
                            "type": "temperature_sensor",
                            "model": "LM35",
                            "component": ref,
                            "interface": "Analog",
                            "range": "0°C to +100°C"
                        })
                    else:
                        sensor_interfaces.append({
                            "type": "temperature_sensor",
                            "model": component_value,
                            "component": ref
                        })
                
                # Motion sensors (accelerometer, gyroscope, etc.)
                elif sensor_type in ["accelerometer", "gyroscope"]:
                    if re.search(r"MPU6050", component_value, re.IGNORECASE):
                        sensor_interfaces.append({
                            "type": "motion_sensor",
                            "model": "MPU6050",
                            "component": ref,
                            "measures": ["accelerometer", "gyroscope"],
                            "interface": "I2C"
                        })
                    elif re.search(r"MPU9250", component_value, re.IGNORECASE):
                        sensor_interfaces.append({
                            "type": "motion_sensor",
                            "model": "MPU9250",
                            "component": ref,
                            "measures": ["accelerometer", "gyroscope", "magnetometer"],
                            "interface": "I2C/SPI"
                        })
                    elif re.search(r"LSM6DS3", component_value, re.IGNORECASE):
                        sensor_interfaces.append({
                            "type": "motion_sensor",
                            "model": "LSM6DS3",
                            "component": ref,
                            "measures": ["accelerometer", "gyroscope"],
                            "interface": "I2C/SPI"
                        })
                    else:
                        sensor_interfaces.append({
                            "type": "motion_sensor",
                            "model": component_value,
                            "component": ref,
                            "measures": [sensor_type]
                        })
                
                # Light and proximity sensors
                elif sensor_type in ["light", "proximity"]:
                    if re.search(r"APDS9960", component_value, re.IGNORECASE):
                        sensor_interfaces.append({
                            "type": "optical_sensor",
                            "model": "APDS9960",
                            "component": ref,
                            "measures": ["proximity", "light", "gesture", "color"],
                            "interface": "I2C"
                        })
                    elif re.search(r"VL53L0X", component_value, re.IGNORECASE):
                        sensor_interfaces.append({
                            "type": "optical_sensor",
                            "model": "VL53L0X",
                            "component": ref,
                            "measures": ["time-of-flight distance"],
                            "interface": "I2C",
                            "range": "Up to 2m"
                        })
                    elif re.search(r"BH1750", component_value, re.IGNORECASE):
                        sensor_interfaces.append({
                            "type": "optical_sensor",
                            "model": "BH1750",
                            "component": ref,
                            "measures": ["ambient light"],
                            "interface": "I2C"
                        })
                    else:
                        sensor_interfaces.append({
                            "type": "optical_sensor",
                            "model": component_value,
                            "component": ref,
                            "measures": [sensor_type]
                        })
                
                # ADCs (often used for sensor interfaces)
                elif sensor_type == "ADC":
                    if re.search(r"ADS1115", component_value, re.IGNORECASE):
                        sensor_interfaces.append({
                            "type": "analog_interface",
                            "model": "ADS1115",
                            "component": ref,
                            "resolution": "16-bit",
                            "channels": 4,
                            "interface": "I2C"
                        })
                    elif re.search(r"HX711", component_value, re.IGNORECASE):
                        sensor_interfaces.append({
                            "type": "analog_interface",
                            "model": "HX711",
                            "component": ref,
                            "resolution": "24-bit",
                            "common_usage": "Load cell/strain gauge",
                            "interface": "Digital"
                        })
                    else:
                        sensor_interfaces.append({
                            "type": "analog_interface",
                            "model": component_value,
                            "component": ref
                        })
                
                # Other types of sensors
                else:
                    sensor_interfaces.append({
                        "type": f"{sensor_type}_sensor",
                        "model": component_value,
                        "component": ref
                    })
                
                # Once identified a component as a specific sensor, no need to check other types
                break
    
    # Look for common analog sensors
    # These often don't have specific ICs but have designators like "RT" for thermistors
    thermistor_refs = [ref for ref in components.keys() if ref.startswith('RT') or ref.startswith('TH')]
    for ref in thermistor_refs:
        component = components[ref]
        sensor_interfaces.append({
            "type": "temperature_sensor",
            "subtype": "thermistor",
            "component": ref,
            "value": component.get('value', ''),
            "interface": "Analog"
        })
    
    # Look for photodiodes, photoresistors (LDRs)
    photosensor_refs = [ref for ref in components.keys() if ref.startswith('PD') or ref.startswith('LDR')]
    for ref in photosensor_refs:
        component = components[ref]
        sensor_interfaces.append({
            "type": "optical_sensor",
            "subtype": "photosensor",
            "component": ref,
            "value": component.get('value', ''),
            "interface": "Analog"
        })
    
    # Look for potentiometers (often used for manual sensing/control)
    pot_refs = [ref for ref in components.keys() if ref.startswith('RV') or ref.startswith('POT')]
    for ref in pot_refs:
        component = components[ref]
        sensor_interfaces.append({
            "type": "position_sensor",
            "subtype": "potentiometer",
            "component": ref,
            "value": component.get('value', ''),
            "interface": "Analog"
        })
    
    return sensor_interfaces


def identify_microcontrollers(components: Dict[str, Any], nets: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Identify microcontroller circuits in the schematic.

    Args:
        components: Dictionary of components from netlist
        nets: Dictionary of nets from netlist

    Returns:
        List of identified microcontroller circuits
    """
    microcontrollers = []
    
    # Common microcontroller families
    mcu_patterns = {
        "AVR": r"ATMEGA\d+|ATTINY\d+|AT90\w+",
        "STM32": r"STM32\w+",
        "PIC": r"PIC\d+\w+",
        "ESP": r"ESP32|ESP8266",
        "Arduino": r"ARDUINO",
        "MSP430": r"MSP430\w+",
        "RP2040": r"RP2040|PICO",
        "NXP": r"LPC\d+|IMXRT\d+|MK\d+",
        "SAM": r"SAMD\d+|SAM\w+",
        "ARM Cortex": r"CORTEX|ARM",
        "8051": r"8051|AT89"
    }
    
    for ref, component in components.items():
        component_value = component.get('value', '').upper()
        component_lib = component.get('lib_id', '').upper()
        
        for family, pattern in mcu_patterns.items():
            if re.search(pattern, component_value, re.IGNORECASE) or re.search(pattern, component_lib, re.IGNORECASE):
                # Identify specific models
                identified = False
                
                # ATmega328P (Arduino Uno/Nano)
                if re.search(r"ATMEGA328P|ATMEGA328", component_value, re.IGNORECASE):
                    microcontrollers.append({
                        "type": "microcontroller",
                        "family": "AVR",
                        "model": "ATmega328P",
                        "component": ref,
                        "common_usage": "Arduino Uno/Nano compatible"
                    })
                    identified = True
                
                # ATmega32U4 (Arduino Leonardo/Micro)
                elif re.search(r"ATMEGA32U4", component_value, re.IGNORECASE):
                    microcontrollers.append({
                        "type": "microcontroller",
                        "family": "AVR",
                        "model": "ATmega32U4",
                        "component": ref,
                        "common_usage": "Arduino Leonardo/Micro compatible"
                    })
                    identified = True
                
                # ESP32
                elif re.search(r"ESP32", component_value, re.IGNORECASE):
                    microcontrollers.append({
                        "type": "microcontroller",
                        "family": "ESP",
                        "model": "ESP32",
                        "component": ref,
                        "features": "Wi-Fi & Bluetooth"
                    })
                    identified = True
                
                # ESP8266
                elif re.search(r"ESP8266", component_value, re.IGNORECASE):
                    microcontrollers.append({
                        "type": "microcontroller",
                        "family": "ESP",
                        "model": "ESP8266",
                        "component": ref,
                        "features": "Wi-Fi"
                    })
                    identified = True
                
                # STM32 series
                elif re.search(r"STM32F\d+", component_value, re.IGNORECASE):
                    model = re.search(r"(STM32F\d+)", component_value, re.IGNORECASE).group(1)
                    microcontrollers.append({
                        "type": "microcontroller",
                        "family": "STM32",
                        "model": model.upper(),
                        "component": ref,
                        "features": "ARM Cortex-M"
                    })
                    identified = True
                
                # Raspberry Pi Pico (RP2040)
                elif re.search(r"RP2040|PICO", component_value, re.IGNORECASE):
                    microcontrollers.append({
                        "type": "microcontroller",
                        "family": "RP2040",
                        "model": "RP2040",
                        "component": ref,
                        "common_usage": "Raspberry Pi Pico"
                    })
                    identified = True
                
                # PIC microcontrollers
                elif re.search(r"PIC\d+", component_value, re.IGNORECASE):
                    model = re.search(r"(PIC\d+\w+)", component_value, re.IGNORECASE)
                    if model:
                        microcontrollers.append({
                            "type": "microcontroller",
                            "family": "PIC",
                            "model": model.group(1).upper(),
                            "component": ref
                        })
                        identified = True
                
                # MSP430 series
                elif re.search(r"MSP430\w+", component_value, re.IGNORECASE):
                    model = re.search(r"(MSP430\w+)", component_value, re.IGNORECASE)
                    if model:
                        microcontrollers.append({
                            "type": "microcontroller",
                            "family": "MSP430",
                            "model": model.group(1).upper(),
                            "component": ref,
                            "features": "Ultra-low power"
                        })
                        identified = True
                
                # If not identified specifically but matches a family
                if not identified:
                    microcontrollers.append({
                        "type": "microcontroller",
                        "family": family,
                        "component": ref,
                        "value": component_value
                    })
                
                # Once identified a component as a microcontroller, no need to check other families
                break
    
    # Look for microcontroller development boards
    dev_board_patterns = {
        "Arduino": r"ARDUINO|UNO|NANO|MEGA|LEONARDO|DUE",
        "ESP32 Dev Board": r"ESP32-DEVKIT|NODEMCU-32S|ESP-WROOM-32",
        "ESP8266 Dev Board": r"NODEMCU|WEMOS|D1_MINI|ESP-01",
        "STM32 Dev Board": r"NUCLEO|DISCOVERY|BLUEPILL",
        "Raspberry Pi": r"RASPBERRY|RPI|RPICO|PICO"
    }
    
    for ref, component in components.items():
        component_value = component.get('value', '').upper()
        component_lib = component.get('lib_id', '').upper()
        
        for board_type, pattern in dev_board_patterns.items():
            if re.search(pattern, component_value, re.IGNORECASE) or re.search(pattern, component_lib, re.IGNORECASE):
                microcontrollers.append({
                    "type": "development_board",
                    "board_type": board_type,
                    "component": ref,
                    "value": component_value
                })
                break
    
    return microcontrollers
