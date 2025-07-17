"""
Centralized KiCad CLI detection and management.

Provides a single source of truth for locating KiCad CLI across platforms
with caching and configuration support.
"""

import logging
import os
import platform
import shutil
import subprocess

from ..config import TIMEOUT_CONSTANTS

logger = logging.getLogger(__name__)


class KiCadCLIError(Exception):
    """Raised when KiCad CLI operations fail."""

    pass


class KiCadCLIManager:
    """
    Manages KiCad CLI detection and validation across platforms.

    Provides caching and fallback mechanisms for reliable CLI access.
    """

    def __init__(self):
        """Initialize the CLI manager."""
        self._cached_cli_path: str | None = None
        self._cache_validated = False
        self._system = platform.system()

    def find_kicad_cli(self, force_refresh: bool = False) -> str | None:
        """
        Find the KiCad CLI executable path.

        Args:
            force_refresh: Force re-detection even if cached

        Returns:
            Path to kicad-cli executable or None if not found
        """
        # Return cached path if available and valid
        if self._cached_cli_path and not force_refresh and self._cache_validated:
            return self._cached_cli_path

        # Try to find CLI
        cli_path = self._detect_cli_path()

        if cli_path:
            # Validate the found CLI
            if self._validate_cli_path(cli_path):
                self._cached_cli_path = cli_path
                self._cache_validated = True
                logger.info(f"Found KiCad CLI at: {cli_path}")
                return cli_path
            else:
                logger.warning(f"Found KiCad CLI at {cli_path} but validation failed")

        # Clear cache if detection failed
        self._cached_cli_path = None
        self._cache_validated = False
        logger.warning("KiCad CLI not found on this system")
        return None

    def get_cli_path(self, required: bool = True) -> str:
        """
        Get KiCad CLI path, raising exception if not found and required.

        Args:
            required: Whether to raise exception if CLI not found

        Returns:
            Path to kicad-cli executable

        Raises:
            KiCadCLIError: If CLI not found and required=True
        """
        cli_path = self.find_kicad_cli()

        if cli_path is None and required:
            raise KiCadCLIError(
                "KiCad CLI not found. Please install KiCad or set KICAD_CLI_PATH environment variable."
            )

        return cli_path

    def is_available(self) -> bool:
        """Check if KiCad CLI is available."""
        return self.find_kicad_cli() is not None

    def get_version(self) -> str | None:
        """
        Get KiCad CLI version string.

        Returns:
            Version string or None if CLI not available
        """
        cli_path = self.find_kicad_cli()
        if not cli_path:
            return None

        try:
            result = subprocess.run(  # nosec B603 - CLI path is validated
                [cli_path, "--version"],
                capture_output=True,
                text=True,
                timeout=TIMEOUT_CONSTANTS["kicad_cli_version_check"],
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.SubprocessError, OSError) as e:
            logger.warning(f"Failed to get KiCad CLI version: {e}")

        return None

    def _detect_cli_path(self) -> str | None:
        """
        Detect KiCad CLI path using platform-specific strategies.

        Returns:
            Path to CLI executable or None if not found
        """
        # Check environment variable first
        env_path = os.environ.get("KICAD_CLI_PATH")
        if env_path and os.path.isfile(env_path) and os.access(env_path, os.X_OK):
            logger.info(f"Using KiCad CLI from environment: {env_path}")
            return env_path

        # Try system PATH
        cli_name = self._get_cli_executable_name()
        system_path = shutil.which(cli_name)
        if system_path:
            logger.info(f"Found KiCad CLI in system PATH: {system_path}")
            return system_path

        # Try platform-specific common locations
        common_paths = self._get_common_installation_paths()
        for path in common_paths:
            if os.path.isfile(path) and os.access(path, os.X_OK):
                logger.info(f"Found KiCad CLI at common location: {path}")
                return path

        return None

    def _get_cli_executable_name(self) -> str:
        """Get the CLI executable name for current platform."""
        if self._system == "Windows":
            return "kicad-cli.exe"
        return "kicad-cli"

    def _get_common_installation_paths(self) -> list[str]:
        """Get list of common installation paths for current platform."""
        paths = []

        if self._system == "Darwin":  # macOS
            paths.extend(
                [
                    "/Applications/KiCad/KiCad.app/Contents/MacOS/kicad-cli",
                    "/Applications/KiCad/kicad-cli",
                    "/usr/local/bin/kicad-cli",
                    "/opt/homebrew/bin/kicad-cli",
                ]
            )
        elif self._system == "Windows":
            paths.extend(
                [
                    r"C:\Program Files\KiCad\bin\kicad-cli.exe",
                    r"C:\Program Files (x86)\KiCad\bin\kicad-cli.exe",
                    r"C:\KiCad\bin\kicad-cli.exe",
                ]
            )
        else:  # Linux and other Unix-like systems
            paths.extend(
                [
                    "/usr/bin/kicad-cli",
                    "/usr/local/bin/kicad-cli",
                    "/opt/kicad/bin/kicad-cli",
                    "/snap/kicad/current/usr/bin/kicad-cli",
                ]
            )

        return paths

    def _validate_cli_path(self, cli_path: str) -> bool:
        """
        Validate that a CLI path is working.

        Args:
            cli_path: Path to validate

        Returns:
            True if CLI is working
        """
        try:
            result = subprocess.run(  # nosec B603 - CLI path is validated
                [cli_path, "--version"],
                capture_output=True,
                text=True,
                timeout=TIMEOUT_CONSTANTS["kicad_cli_version_check"],
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, OSError, FileNotFoundError):
            return False


# Global CLI manager instance
_cli_manager = None


def get_cli_manager() -> KiCadCLIManager:
    """Get the global KiCad CLI manager instance."""
    global _cli_manager
    if _cli_manager is None:
        _cli_manager = KiCadCLIManager()
    return _cli_manager


def find_kicad_cli(force_refresh: bool = False) -> str | None:
    """Convenience function to find KiCad CLI path."""
    return get_cli_manager().find_kicad_cli(force_refresh)


def get_kicad_cli_path(required: bool = True) -> str:
    """Convenience function to get KiCad CLI path."""
    return get_cli_manager().get_cli_path(required)


def is_kicad_cli_available() -> bool:
    """Convenience function to check if KiCad CLI is available."""
    return get_cli_manager().is_available()


def get_kicad_version() -> str | None:
    """Convenience function to get KiCad CLI version."""
    return get_cli_manager().get_version()
