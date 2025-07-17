"""
Path validation utility for KiCad MCP.

Provides secure path validation to prevent path traversal attacks
and ensure file operations are restricted to safe directories.
"""

import os
import pathlib

from kicad_mcp.config import KICAD_EXTENSIONS


class PathValidationError(Exception):
    """Raised when path validation fails."""

    pass


class PathValidator:
    """
    Validates file paths for security and correctness.

    Prevents path traversal attacks and ensures files are within
    trusted directories with valid KiCad extensions.
    """

    def __init__(self, trusted_roots: set[str] | None = None):
        """
        Initialize path validator.

        Args:
            trusted_roots: Set of trusted root directories. If None,
                          uses current working directory.
        """
        self.trusted_roots = trusted_roots or {os.getcwd()}
        # Normalize trusted roots to absolute paths
        self.trusted_roots = {
            os.path.realpath(os.path.expanduser(root)) for root in self.trusted_roots
        }

    def add_trusted_root(self, root_path: str) -> None:
        """
        Add a trusted root directory.

        Args:
            root_path: Path to add as trusted root
        """
        normalized_root = os.path.realpath(os.path.expanduser(root_path))
        self.trusted_roots.add(normalized_root)

    def validate_path(self, file_path: str, must_exist: bool = False) -> str:
        """
        Validate a file path for security and correctness.

        Args:
            file_path: Path to validate
            must_exist: Whether the file must exist

        Returns:
            Normalized absolute path

        Raises:
            PathValidationError: If path validation fails
        """
        if not file_path or not isinstance(file_path, str):
            raise PathValidationError("Path must be a non-empty string")

        try:
            # Expand user home directory and resolve symbolic links
            normalized_path = os.path.realpath(os.path.expanduser(file_path))
        except (OSError, ValueError) as e:
            raise PathValidationError(f"Invalid path: {e}") from e

        # Check if path is within trusted roots
        if not self._is_within_trusted_roots(normalized_path):
            raise PathValidationError(f"Path '{file_path}' is outside trusted directories")

        # Check if file exists when required
        if must_exist and not os.path.exists(normalized_path):
            raise PathValidationError(f"Path does not exist: {file_path}")

        return normalized_path

    def validate_kicad_file(self, file_path: str, file_type: str, must_exist: bool = True) -> str:
        """
        Validate a KiCad file path with extension checking.

        Args:
            file_path: Path to validate
            file_type: Expected KiCad file type ('project', 'schematic', 'pcb', etc.)
            must_exist: Whether the file must exist

        Returns:
            Normalized absolute path

        Raises:
            PathValidationError: If path validation fails
        """
        # First validate the basic path
        normalized_path = self.validate_path(file_path, must_exist)

        # Check file extension
        if file_type not in KICAD_EXTENSIONS:
            raise PathValidationError(f"Unknown KiCad file type: {file_type}")

        expected_extension = KICAD_EXTENSIONS[file_type]
        if not normalized_path.endswith(expected_extension):
            raise PathValidationError(
                f"File must have {expected_extension} extension, got: {file_path}"
            )

        return normalized_path

    def validate_directory(self, dir_path: str, must_exist: bool = True) -> str:
        """
        Validate a directory path.

        Args:
            dir_path: Directory path to validate
            must_exist: Whether the directory must exist

        Returns:
            Normalized absolute directory path

        Raises:
            PathValidationError: If validation fails
        """
        normalized_path = self.validate_path(dir_path, must_exist)

        if must_exist and not os.path.isdir(normalized_path):
            raise PathValidationError(f"Path is not a directory: {dir_path}")

        return normalized_path

    def validate_project_directory(self, project_path: str) -> str:
        """
        Validate and return the directory containing a KiCad project file.

        Args:
            project_path: Path to .kicad_pro file

        Returns:
            Normalized absolute directory path

        Raises:
            PathValidationError: If validation fails
        """
        validated_project = self.validate_kicad_file(project_path, "project", must_exist=True)
        return os.path.dirname(validated_project)

    def create_safe_temp_path(self, base_name: str, extension: str = "") -> str:
        """
        Create a safe temporary file path within trusted directories.

        Args:
            base_name: Base name for the temporary file
            extension: File extension (including dot)

        Returns:
            Safe temporary file path
        """
        import tempfile

        # Use the first trusted root as temp directory base
        temp_root = next(iter(self.trusted_roots))

        # Create temp directory if it doesn't exist
        temp_dir = os.path.join(temp_root, "temp")
        os.makedirs(temp_dir, exist_ok=True)

        # Generate unique temp file path
        temp_fd, temp_path = tempfile.mkstemp(
            suffix=extension, prefix=f"{base_name}_", dir=temp_dir
        )
        os.close(temp_fd)  # Close the file descriptor, we just need the path

        return temp_path

    def _is_within_trusted_roots(self, path: str) -> bool:
        """
        Check if a path is within any trusted root directory.

        Args:
            path: Normalized absolute path to check

        Returns:
            True if path is within trusted roots
        """
        for root in self.trusted_roots:
            try:
                # Check if path is within this root
                pathlib.Path(root).resolve()
                pathlib.Path(path).resolve().relative_to(pathlib.Path(root).resolve())
                return True
            except ValueError:
                # Path is not relative to this root
                continue
        return False


# Global default validator instance
_default_validator = None


def get_default_validator() -> PathValidator:
    """Get the default global path validator instance."""
    global _default_validator
    if _default_validator is None:
        _default_validator = PathValidator()
    return _default_validator


def validate_path(file_path: str, must_exist: bool = False) -> str:
    """Convenience function using default validator."""
    return get_default_validator().validate_path(file_path, must_exist)


def validate_kicad_file(file_path: str, file_type: str, must_exist: bool = True) -> str:
    """Convenience function using default validator."""
    return get_default_validator().validate_kicad_file(file_path, file_type, must_exist)


def validate_directory(dir_path: str, must_exist: bool = True) -> str:
    """Convenience function using default validator."""
    return get_default_validator().validate_directory(dir_path, must_exist)
