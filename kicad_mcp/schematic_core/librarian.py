"""
Librarian - State Manager and Navigation Layer for Unified Schematic Core

This module implements the "Atlas" pattern for schematic navigation and context
extraction. It orchestrates data fetching, caching, and high-level queries for
the DSL emitter.

Key Responsibilities:
- State management with "Nuke and Rebuild" pattern
- Building the Atlas (net-to-pages mapping)
- Providing navigation queries (index, page, context)
- 1-hop traversal for context bubbles
"""

from typing import List, Dict, Set, Optional

try:
    from .interfaces import SchematicProvider
    from .models import Component, Net, Pin
    from . import dsl_emitter
except ImportError:
    from interfaces import SchematicProvider
    from models import Component, Net, Pin
    import dsl_emitter


class Librarian:
    """
    Central state manager and navigation layer for schematic data.

    The Librarian acts as the orchestrator between data providers and DSL
    emitters. It implements a caching strategy where data is fetched once
    and cached until explicitly marked dirty.

    The Atlas is a key data structure mapping net names to the set of pages
    where they appear, enabling efficient inter-page signal detection.

    Attributes:
        provider: SchematicProvider implementation for data fetching
        dirty: Flag indicating if cached data needs refresh
        components: Cached list of all components
        nets: Cached list of all nets
        net_page_map: The Atlas - maps net names to sets of page names
    """

    def __init__(self, provider: SchematicProvider):
        """
        Initialize the Librarian with a data provider.

        Args:
            provider: Implementation of SchematicProvider interface
        """
        self.provider = provider
        self.dirty = True
        self.components: List[Component] = []
        self.nets: List[Net] = []
        self.net_page_map: Dict[str, Set[str]] = {}  # The Atlas

    def refresh(self) -> None:
        """
        Nuke and rebuild all state if dirty flag is set.

        This implements the "Nuke and Rebuild" pattern - no incremental updates,
        just full refresh when needed. If data is already fresh (dirty=False),
        this is a no-op.

        The refresh process:
        1. Fetch raw data from provider
        2. Get normalized components and nets
        3. Build the Atlas (net-to-pages mapping)
        4. Clear dirty flag

        Raises:
            Exception: Propagates any exceptions from the provider
        """
        if not self.dirty:
            return

        # Fetch fresh data from provider
        self.provider.fetch_raw_data()
        self.components = self.provider.get_components()
        self.nets = self.provider.get_nets()

        # Build the Atlas - map each net name to its set of pages
        self.net_page_map = {}
        for net in self.nets:
            self.net_page_map[net.name] = net.pages

        self.dirty = False

    def get_index(self) -> str:
        """
        Generate high-level schematic overview with page list and inter-page signals.

        The index provides a navigation overview showing:
        - All pages with component and net counts
        - Inter-page signals (nets spanning multiple pages)
        - Global nets (power/ground rails)

        Returns:
            Formatted index string with page summary and inter-page signals

        Format:
            # SCHEMATIC INDEX

            ## Pages
            - Main_Sheet (25 components, 40 nets)
            - Power_Module (8 components, 15 nets)

            ## Inter-Page Signals
            - UART_TX: Main_Sheet ↔ Connector_Page
            - GND: ALL_PAGES (Power Rail)
        """
        self.refresh()

        # Handle empty schematic
        if not self.components and not self.nets:
            return "# SCHEMATIC INDEX\n\n(Empty schematic - no components or nets)\n"

        lines = ["# SCHEMATIC INDEX", ""]

        # Build page statistics
        # Group components by page
        page_component_counts: Dict[str, int] = {}
        for comp in self.components:
            page_component_counts[comp.page] = page_component_counts.get(comp.page, 0) + 1

        # Group nets by page (count nets that appear on each page)
        page_net_counts: Dict[str, int] = {}
        for net in self.nets:
            for page in net.pages:
                page_net_counts[page] = page_net_counts.get(page, 0) + 1

        # Get all unique pages (from both components and nets)
        all_pages = set(page_component_counts.keys()) | set(page_net_counts.keys())

        # Pages section
        lines.append("## Pages")
        if not all_pages:
            lines.append("(No pages found)")
        else:
            # Sort pages alphabetically
            for page in sorted(all_pages):
                comp_count = page_component_counts.get(page, 0)
                net_count = page_net_counts.get(page, 0)
                lines.append(f"- {page} ({comp_count} components, {net_count} nets)")

        lines.append("")

        # Inter-page signals section
        lines.append("## Inter-Page Signals")

        # Find all inter-page nets
        inter_page_nets = [net for net in self.nets if net.is_inter_page()]

        if not inter_page_nets:
            lines.append("(No inter-page signals)")
        else:
            # Sort by net name
            inter_page_nets.sort(key=lambda n: n.name)

            for net in inter_page_nets:
                if net.is_global():
                    # Global nets show as ALL_PAGES
                    # Classify as Power Rail or Ground based on name
                    if 'GND' in net.name.upper() or 'VSS' in net.name.upper():
                        lines.append(f"- {net.name}: ALL_PAGES (Ground)")
                    else:
                        lines.append(f"- {net.name}: ALL_PAGES (Power Rail)")
                else:
                    # Regular inter-page nets show page connections
                    pages = sorted(net.pages)
                    pages_str = " ↔ ".join(pages)
                    lines.append(f"- {net.name}: {pages_str}")

        return "\n".join(lines)

    def get_page(self, page_name: str) -> str:
        """
        Generate DSL representation of a single schematic page.

        Args:
            page_name: Name of the page to retrieve

        Returns:
            Formatted DSL string for the specified page

        Notes:
            - If page doesn't exist, returns a message indicating that
            - Only includes nets that have at least one connection on this page
            - Uses dsl_emitter.emit_page_dsl() for actual formatting
        """
        self.refresh()

        # Filter components for this page
        page_components = [c for c in self.components if c.page == page_name]

        # Check if page exists
        if not page_components:
            # Check if page exists in nets
            page_exists = any(page_name in net.pages for net in self.nets)
            if not page_exists:
                return f"# PAGE: {page_name}\n\n(Page not found in schematic)\n"
            else:
                # Page exists but has no components (only nets)
                page_components = []

        # Filter nets to include only those with pins on this page
        # A net appears on this page if any of its member components are on this page
        page_component_refdes = {c.refdes for c in page_components}
        page_nets = []
        for net in self.nets:
            # Check if any net member is on this page
            has_connection_on_page = any(
                refdes in page_component_refdes
                for refdes, pin_designator in net.members
            )
            if has_connection_on_page:
                page_nets.append(net)

        # Use DSL emitter to format the page
        return dsl_emitter.emit_page_dsl(page_components, page_nets, self.net_page_map)

    def get_context(self, refdes_list: List[str]) -> str:
        """
        Generate DSL for context bubble around specific components (1-hop traversal).

        This performs a 1-hop traversal from the primary components:
        1. Get primary components (those in refdes_list)
        2. Find all nets connected to primary component pins
        3. Find all neighbor components connected to those nets
        4. Classify neighbors as passive (inline) or active (summary)
        5. Handle global nets by potentially truncating connections

        Args:
            refdes_list: List of component reference designators to build context around

        Returns:
            Formatted DSL string showing context bubble with primary components,
            neighbors, and connecting nets

        Format:
            # CONTEXT: U1, R5

            # COMPONENTS
            <complex primary component blocks>

            # CONTEXT_NEIGHBORS
            <simplified neighbor summaries>

            # NETS
            <net blocks showing connectivity>
        """
        self.refresh()

        # Handle empty input
        if not refdes_list:
            return "# CONTEXT: (empty)\n\n(No components specified for context)\n"

        # Step 1: Get primary components
        primary_components = [c for c in self.components if c.refdes in refdes_list]

        # Handle case where none of the requested components exist
        if not primary_components:
            missing = ", ".join(refdes_list)
            return f"# CONTEXT: {missing}\n\n(Components not found in schematic)\n"

        # Step 2: Find all nets connected to primary components
        primary_refdes_set = {c.refdes for c in primary_components}
        connected_nets = []

        for net in self.nets:
            # Check if any member of this net is a primary component
            has_primary = any(
                refdes in primary_refdes_set
                for refdes, pin_designator in net.members
            )
            if has_primary:
                connected_nets.append(net)

        # Step 3: Find all neighbor components on those nets
        neighbor_refdes_set = set()

        for net in connected_nets:
            for refdes, pin_designator in net.members:
                # Add to neighbors if not a primary component
                if refdes not in primary_refdes_set:
                    neighbor_refdes_set.add(refdes)

        # Get full neighbor component objects
        all_neighbors = [c for c in self.components if c.refdes in neighbor_refdes_set]

        # Step 4: Classify neighbors - only active (non-passive) go in CONTEXT_NEIGHBORS
        # Passive components will appear inline in NET lines only
        neighbor_components = [c for c in all_neighbors if not c.is_passive()]

        # Step 5: Handle global nets
        # For context DSL, we may want to filter global net members to only show
        # connections relevant to the context (primary + neighbors)
        context_refdes = primary_refdes_set | neighbor_refdes_set
        context_nets = []

        for net in connected_nets:
            # Filter net members to only those in our context
            filtered_members = [
                (refdes, pin_designator)
                for refdes, pin_designator in net.members
                if refdes in context_refdes
            ]

            # Create a filtered version of the net for context
            # We need to preserve the original net structure but with filtered members
            # Create a new Net object with filtered data
            context_net = Net(
                name=net.name,
                pages=net.pages,
                members=filtered_members
            )
            context_nets.append(context_net)

        # Use DSL emitter to format the context
        return dsl_emitter.emit_context_dsl(
            primary_components,
            neighbor_components,
            context_nets
        )

    def mark_dirty(self) -> None:
        """
        Mark the cached data as dirty, forcing refresh on next query.

        This should be called if the underlying schematic data has changed
        and needs to be re-fetched.
        """
        self.dirty = True

    def get_all_pages(self) -> List[str]:
        """
        Get list of all unique page names in the schematic.

        Returns:
            Sorted list of page names

        Notes:
            This is a helper method for external tools that need to
            enumerate available pages.
        """
        self.refresh()

        pages = set()

        # Get pages from components
        for comp in self.components:
            pages.add(comp.page)

        # Get pages from nets
        for net in self.nets:
            pages.update(net.pages)

        return sorted(pages)

    def get_component(self, refdes: str) -> Optional[Component]:
        """
        Get a specific component by reference designator.

        Args:
            refdes: Component reference designator (e.g., "U1", "R5")

        Returns:
            Component object if found, None otherwise

        Notes:
            This is a helper method for external tools that need to
            query individual components.
        """
        self.refresh()

        for comp in self.components:
            if comp.refdes == refdes:
                return comp

        return None

    def get_net(self, net_name: str) -> Optional[Net]:
        """
        Get a specific net by name.

        Args:
            net_name: Net name (e.g., "GND", "UART_TX")

        Returns:
            Net object if found, None otherwise

        Notes:
            This is a helper method for external tools that need to
            query individual nets.
        """
        self.refresh()

        for net in self.nets:
            if net.name == net_name:
                return net

        return None

    def get_stats(self) -> Dict[str, int]:
        """
        Get basic statistics about the schematic.

        Returns:
            Dictionary with count statistics:
            - total_components: Total number of components
            - total_nets: Total number of nets
            - total_pages: Number of unique pages
            - inter_page_nets: Number of nets spanning multiple pages
            - global_nets: Number of global nets (power/ground rails)

        Notes:
            Useful for debugging and reporting.
        """
        self.refresh()

        stats = {
            'total_components': len(self.components),
            'total_nets': len(self.nets),
            'total_pages': len(self.get_all_pages()),
            'inter_page_nets': sum(1 for net in self.nets if net.is_inter_page()),
            'global_nets': sum(1 for net in self.nets if net.is_global()),
        }

        return stats
