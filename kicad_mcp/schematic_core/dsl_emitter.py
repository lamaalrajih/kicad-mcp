"""
DSL Emitter for Unified Schematic Core v0.3

This module generates token-efficient Domain Specific Language (DSL) output
from normalized schematic data models. The DSL is optimized for LLM consumption
with net-centric connectivity and adaptive component formatting.

Key Features:
- Net-centric: All connectivity defined in NETS section only
- Compressed passives: Simple R/C/L components inline only, no COMP blocks
- Expanded actives: Complex components get full DEF blocks with pin listings
- Inline pin hints: Format as U1.22(PA9_TX) for named pins
- Global net summaries: Truncate large nets to first 10 connections
- Inter-page links: Show LINKS line for nets spanning multiple pages
"""

from typing import List, Dict, Set, Tuple
try:
    from .models import Component, Net, Pin
except ImportError:
    from models import Component, Net, Pin


def emit_page_dsl(
    components: List[Component],
    nets: List[Net],
    net_page_map: Dict[str, Set[str]]
) -> str:
    """
    Generate DSL for a single schematic page.

    Args:
        components: List of components on this page
        nets: List of nets with pins on this page
        net_page_map: Dict mapping net names to set of page names (the Atlas)

    Returns:
        Formatted DSL string for the page

    Format:
        # PAGE: <page_name>

        # COMPONENTS
        <complex component blocks>

        # NETS
        <net blocks>
    """
    if not components:
        return "# No components on this page\n"

    # Get page name from first component
    page_name = components[0].page if components else "Unknown"

    # Sort components alphabetically by refdes
    sorted_components = sorted(components, key=lambda c: c.refdes)

    # Sort nets alphabetically by name
    sorted_nets = sorted(nets, key=lambda n: n.name)

    # Build output
    lines = [f"# PAGE: {page_name}", ""]

    # COMPONENTS section - only complex components get blocks
    lines.append("# COMPONENTS")
    complex_components = [c for c in sorted_components if c.is_complex()]

    if not complex_components:
        lines.append("(All components are simple passives - see NETS section)")
    else:
        for comp in complex_components:
            block = _format_component_block(comp)
            lines.append(block)

    lines.append("")

    # NETS section
    lines.append("# NETS")
    for net in sorted_nets:
        # Use net_page_map to determine if net is inter-page
        net_pages = net_page_map.get(net.name, set())
        block = _format_net_block(net, net_pages, sorted_components)
        lines.append(block)

    return "\n".join(lines)


def emit_context_dsl(
    primary_components: List[Component],
    neighbor_components: List[Component],
    nets: List[Net]
) -> str:
    """
    Generate DSL for a context bubble (1-hop traversal from primary components).

    This output includes:
    - Full COMP blocks for primary components
    - CONTEXT_NEIGHBORS section with simplified summaries for neighbor components
    - Nets showing connectivity between primary and neighbors

    Args:
        primary_components: Components explicitly requested for context
        neighbor_components: Components found in 1-hop traversal
        nets: Nets connecting primary and neighbor components

    Returns:
        Formatted DSL string for the context bubble

    Format:
        # CONTEXT: <refdes1>, <refdes2>, ...

        # COMPONENTS
        <complex component blocks for primary>

        # CONTEXT_NEIGHBORS
        <simplified component summaries>

        # NETS
        <net blocks>
    """
    if not primary_components:
        return "# No components in context\n"

    # Sort components and nets
    sorted_primary = sorted(primary_components, key=lambda c: c.refdes)
    sorted_neighbors = sorted(neighbor_components, key=lambda c: c.refdes)
    sorted_nets = sorted(nets, key=lambda n: n.name)

    # Build output
    primary_refdes = ", ".join(c.refdes for c in sorted_primary)
    lines = [f"# CONTEXT: {primary_refdes}", ""]

    # COMPONENTS section - primary components only
    lines.append("# COMPONENTS")
    for comp in sorted_primary:
        if comp.is_complex():
            block = _format_component_block(comp)
            lines.append(block)
        else:
            # Simple primary components still get listed (but inline in nets)
            pass

    lines.append("")

    # CONTEXT_NEIGHBORS section - simplified summaries
    if sorted_neighbors:
        lines.append("# CONTEXT_NEIGHBORS")
        for comp in sorted_neighbors:
            # Format: U2 (LM358) - Dual Op-Amp
            summary = _format_neighbor_summary(comp)
            lines.append(summary)
        lines.append("")

    # NETS section
    lines.append("# NETS")
    all_components = sorted_primary + sorted_neighbors
    for net in sorted_nets:
        # For context, we don't have full net_page_map, so just use net.pages
        block = _format_net_block(net, net.pages, all_components)
        lines.append(block)

    return "\n".join(lines)


def _format_component_block(component: Component) -> str:
    """
    Format a complex component as a DEF block.

    Format:
        DEF <type> <description>
        COMP <refdes> (<value>)
          MPN: <mpn>
          FP: <footprint>
          PINS:
            <pin_designator>: <pin_name>
            ...

    Args:
        component: Component to format

    Returns:
        Formatted component block as multi-line string
    """
    lines = []

    # DEF line
    comp_type = component.derived_type()
    if component.description:
        lines.append(f"DEF {comp_type} {component.description}")
    else:
        lines.append(f"DEF {comp_type}")

    # COMP line
    lines.append(f"COMP {component.refdes} ({component.value})")

    # MPN line (omit if empty)
    if component.mpn:
        lines.append(f"  MPN: {component.mpn}")

    # FP line (omit if empty)
    if component.footprint:
        lines.append(f"  FP: {component.footprint}")

    # PINS section
    if component.pins:
        lines.append("  PINS:")
        # Sort pins alphabetically by designator (natural sort for numbers)
        sorted_pins = sorted(component.pins, key=lambda p: _natural_sort_key(p.designator))
        for pin in sorted_pins:
            if pin.name:
                lines.append(f"    {pin.designator}: {pin.name}")
            else:
                lines.append(f"    {pin.designator}:")

    return "\n".join(lines)


def _format_net_block(
    net: Net,
    net_pages: Set[str],
    components: List[Component]
) -> str:
    """
    Format a net block with connectivity information.

    Format (standard):
        NET <net_name>
          CON: <comp.pin>, <comp.pin>, ...

    Format (inter-page):
        NET <net_name>
          LINKS: <page1>, <page2>
          CON: <comp.pin>, ...

    Format (global/summarized):
        NET <net_name>
          LINKS: ALL_PAGES
          CON: <comp.pin>, ... (+ N others)

    Args:
        net: Net to format
        net_pages: Set of pages where this net appears
        components: List of components (for pin lookup)

    Returns:
        Formatted net block as multi-line string
    """
    lines = []

    # NET line
    lines.append(f"NET {net.name}")

    # LINKS line (for inter-page or global nets)
    is_global = net.is_global()
    is_inter_page = len(net_pages) > 1

    if is_global:
        lines.append("  LINKS: ALL_PAGES")
    elif is_inter_page:
        # Sort pages alphabetically
        sorted_pages = sorted(net_pages)
        pages_str = ", ".join(sorted_pages)
        lines.append(f"  LINKS: {pages_str}")

    # CON line - format pin references
    pin_refs = []
    for refdes, pin_designator in net.members:
        pin_ref = _format_pin_reference(refdes, pin_designator, components)
        pin_refs.append(pin_ref)

    # Sort pin references alphabetically
    pin_refs.sort()

    # Truncate for global nets (show first 10 + count)
    if is_global and len(pin_refs) > 10:
        shown_refs = pin_refs[:10]
        others_count = len(pin_refs) - 10
        con_str = ", ".join(shown_refs) + f" (+ {others_count} others)"
    else:
        con_str = ", ".join(pin_refs)

    lines.append(f"  CON: {con_str}")

    return "\n".join(lines)


def _format_pin_reference(
    refdes: str,
    pin_designator: str,
    components: List[Component]
) -> str:
    """
    Format a pin reference for inclusion in a net connection list.

    Rules:
    - Simple pin (no name or simple name): "R1.1"
    - Complex pin with semantic name: "U1.22(PA9_TX)"

    Args:
        refdes: Component reference designator
        pin_designator: Pin number/identifier
        components: List of components to lookup pin details

    Returns:
        Formatted pin reference string
    """
    # Find the component
    component = None
    for comp in components:
        if comp.refdes == refdes:
            component = comp
            break

    # If component not found or is simple, just return refdes.pin
    if not component:
        return f"{refdes}.{pin_designator}"

    # Find the pin
    pin = None
    for p in component.pins:
        if p.designator == pin_designator:
            pin = p
            break

    # If pin not found or has no name, simple format
    if not pin or not pin.name:
        return f"{refdes}.{pin_designator}"

    # Check if pin name is "simple" (just numeric or A/K)
    simple_names = {"1", "2", "3", "4", "A", "K"}
    if pin.name in simple_names:
        return f"{refdes}.{pin_designator}"

    # Complex pin with semantic name - include it in parentheses
    return f"{refdes}.{pin_designator}({pin.name})"


def _format_neighbor_summary(component: Component) -> str:
    """
    Format a neighbor component as a simplified one-line summary.

    Format: <refdes> (<value>) - <description>

    Args:
        component: Neighbor component to summarize

    Returns:
        One-line summary string
    """
    if component.description:
        return f"{component.refdes} ({component.value}) - {component.description}"
    else:
        return f"{component.refdes} ({component.value})"


def _natural_sort_key(text: str) -> Tuple:
    """
    Generate a sort key for natural sorting (e.g., "1", "2", "10" instead of "1", "10", "2").

    This is used for sorting pin designators that may contain numbers.

    Args:
        text: Text to generate sort key for

    Returns:
        Tuple that can be used for sorting
    """
    import re

    def atoi(text):
        return int(text) if text.isdigit() else text

    return tuple(atoi(c) for c in re.split(r'(\d+)', text))
