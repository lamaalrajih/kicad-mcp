"""
Abstract provider interface for the Unified Schematic DSL Generator.

This module defines the contract that all EDA tool adapters must implement
to provide schematic data to the core library. This abstraction allows the
core logic to remain tool-agnostic.
"""

from abc import ABC, abstractmethod
from typing import List

try:
    from .models import Component, Net
except ImportError:
    from models import Component, Net


class SchematicProvider(ABC):
    """
    Abstract base class defining the contract for EDA tool adapters.

    All schematic data providers (Altium, KiCad, etc.) must implement this
    interface to work with the core library. This enables the hexagonal
    architecture where the core logic depends only on abstractions, not
    concrete implementations.

    Implementation Notes:
        - fetch_raw_data() must be idempotent (safe to call multiple times)
        - get_components() and get_nets() can be called multiple times after
          fetch_raw_data() without re-fetching
        - Providers must handle missing/malformed data gracefully using defaults
        - All methods should raise appropriate exceptions for fatal errors
    """

    @abstractmethod
    def fetch_raw_data(self) -> None:
        """
        Fetch data from the source (API call, file read, database query, etc.).

        This method updates the internal state of the provider but returns
        nothing. It should be called before get_components() or get_nets().

        The implementation must be idempotent - calling it multiple times
        should be safe and produce the same result (fresh data fetch).

        Raises:
            Exception: Implementation-specific exceptions for connection
                      failures, file not found, parsing errors, etc.
        """
        pass

    @abstractmethod
    def get_components(self) -> List[Component]:
        """
        Return normalized Component objects from the fetched data.

        This method should be called after fetch_raw_data(). It transforms
        the raw tool-specific data into the unified Component data model.

        Returns:
            List of Component objects with all fields populated. Empty list
            if no components are available.

        Implementation Requirements:
            - All required Component fields must be populated (use empty
              strings or appropriate defaults for missing data)
            - Pins list must be populated with connectivity information
            - Location tuple should use (0, 0) if position unknown
            - Properties dict should contain any additional metadata

        Raises:
            Exception: If called before fetch_raw_data() or if data
                      transformation fails critically
        """
        pass

    @abstractmethod
    def get_nets(self) -> List[Net]:
        """
        Return normalized Net objects from the fetched data.

        This method should be called after fetch_raw_data(). It builds the
        net list by analyzing component pin connectivity and transforming
        it into the unified Net data model.

        Returns:
            List of Net objects with members and pages populated. Empty list
            if no nets are available.

        Implementation Requirements:
            - Each net must have a name (use auto-generated names like
              "Net_U1_5" for unnamed nets if needed)
            - Members list must contain (refdes, pin_designator) tuples
            - Pages set must contain all page names where net appears
            - Nets with no connections should typically be excluded

        Raises:
            Exception: If called before fetch_raw_data() or if data
                      transformation fails critically
        """
        pass
