"""
Secure subprocess execution utilities for KiCad MCP.

Provides safe subprocess execution with input validation,
timeout enforcement, and security controls.
"""

import asyncio
import logging
import os
import subprocess  # nosec B404 - subprocess usage is secured with validation

from ..config import TIMEOUT_CONSTANTS
from .kicad_cli import get_kicad_cli_path
from .path_validator import PathValidator, get_default_validator

logger = logging.getLogger(__name__)


class SecureSubprocessError(Exception):
    """Raised when secure subprocess operations fail."""

    pass


class SecureSubprocessRunner:
    """
    Secure subprocess runner with validation and safety controls.

    Provides methods for safely executing KiCad CLI commands and other
    subprocess operations with proper input validation and security controls.
    """

    def __init__(self, path_validator: PathValidator | None = None):
        """
        Initialize secure subprocess runner.

        Args:
            path_validator: Path validator to use (defaults to global instance)
        """
        self.path_validator = path_validator or get_default_validator()
        self.default_timeout = TIMEOUT_CONSTANTS["subprocess_default"]

    def run_kicad_command(
        self,
        command_args: list[str],
        input_files: list[str] | None = None,
        output_files: list[str] | None = None,
        working_dir: str | None = None,
        timeout: float | None = None,
        capture_output: bool = True,
    ) -> subprocess.CompletedProcess:
        """
        Run a KiCad CLI command with security validation.

        Args:
            command_args: Command arguments (excluding the kicad-cli executable)
            input_files: List of input file paths to validate
            output_files: List of output file paths to validate
            working_dir: Working directory for command execution
            timeout: Command timeout in seconds
            capture_output: Whether to capture stdout/stderr

        Returns:
            CompletedProcess result

        Raises:
            SecureSubprocessError: If validation fails or command fails
            KiCadCLIError: If KiCad CLI not found
            PathValidationError: If path validation fails
        """
        # Get and validate KiCad CLI path
        kicad_cli = get_kicad_cli_path(required=True)

        # Validate input files
        if input_files:
            for file_path in input_files:
                self.path_validator.validate_path(file_path, must_exist=True)

        # Validate output file directories
        if output_files:
            for file_path in output_files:
                output_dir = os.path.dirname(file_path)
                if output_dir:  # Only validate if there's a directory component
                    self.path_validator.validate_directory(output_dir, must_exist=True)

        # Validate working directory
        if working_dir:
            working_dir = self.path_validator.validate_directory(working_dir, must_exist=True)

        # Construct full command
        full_command = [kicad_cli] + command_args

        # Log command for debugging (sanitized)
        logger.debug(f"Executing KiCad command: {' '.join(full_command)}")

        try:
            return self._run_subprocess(
                full_command,
                working_dir=working_dir,
                timeout=timeout or self.default_timeout,
                capture_output=capture_output,
            )
        except subprocess.SubprocessError as e:
            raise SecureSubprocessError(f"KiCad command failed: {e}") from e

    async def run_kicad_command_async(
        self,
        command_args: list[str],
        input_files: list[str] | None = None,
        output_files: list[str] | None = None,
        working_dir: str | None = None,
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess:
        """
        Async version of run_kicad_command.

        Args:
            command_args: Command arguments (excluding the kicad-cli executable)
            input_files: List of input file paths to validate
            output_files: List of output file paths to validate
            working_dir: Working directory for command execution
            timeout: Command timeout in seconds

        Returns:
            CompletedProcess result
        """
        # Run in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.run_kicad_command,
            command_args,
            input_files,
            output_files,
            working_dir,
            timeout,
            True,  # capture_output
        )

    def run_safe_command(
        self,
        command: list[str],
        working_dir: str | None = None,
        timeout: float | None = None,
        allowed_commands: list[str] | None = None,
        capture_output: bool = True,
    ) -> subprocess.CompletedProcess:
        """
        Run a general command with security validation.

        Args:
            command: Full command list including executable
            working_dir: Working directory for command execution
            timeout: Command timeout in seconds
            allowed_commands: List of allowed executables (whitelist)
            capture_output: Whether to capture stdout/stderr

        Returns:
            CompletedProcess result

        Raises:
            SecureSubprocessError: If validation fails or command fails
        """
        if not command:
            raise SecureSubprocessError("Command cannot be empty")

        executable = command[0]

        # Validate executable against whitelist if provided
        if allowed_commands and executable not in allowed_commands:
            raise SecureSubprocessError(f"Command '{executable}' not in allowed list")

        # Validate working directory
        if working_dir:
            working_dir = self.path_validator.validate_directory(working_dir, must_exist=True)

        # Log command for debugging (sanitized)
        logger.debug(f"Executing safe command: {' '.join(command)}")

        try:
            return self._run_subprocess(
                command,
                working_dir=working_dir,
                timeout=timeout or self.default_timeout,
                capture_output=capture_output,
            )
        except subprocess.SubprocessError as e:
            raise SecureSubprocessError(f"Command failed: {e}") from e

    def create_temp_file(
        self, suffix: str = "", prefix: str = "kicad_mcp_", content: str | None = None
    ) -> str:
        """
        Create a temporary file within validated directories.

        Args:
            suffix: File suffix/extension
            prefix: File prefix
            content: Optional content to write to file

        Returns:
            Path to created temporary file
        """
        temp_path = self.path_validator.create_safe_temp_path(prefix.rstrip("_"), suffix)

        if content is not None:
            with open(temp_path, "w", encoding="utf-8") as f:
                f.write(content)

        return temp_path

    def _run_subprocess(
        self,
        command: list[str],
        working_dir: str | None = None,
        timeout: float = TIMEOUT_CONSTANTS["subprocess_default"],
        capture_output: bool = True,
    ) -> subprocess.CompletedProcess:
        """
        Internal subprocess runner with consistent settings.

        Args:
            command: Command to execute
            working_dir: Working directory
            timeout: Timeout in seconds
            capture_output: Whether to capture output

        Returns:
            CompletedProcess result

        Raises:
            subprocess.SubprocessError: If command fails
        """
        kwargs = {
            "timeout": timeout,
            "cwd": working_dir,
            "text": True,
        }

        if capture_output:
            kwargs.update(
                {
                    "capture_output": True,
                    "check": False,  # Don't raise on non-zero exit code
                }
            )

        return subprocess.run(command, **kwargs)  # nosec B603 - input is validated


# Global secure subprocess runner instance
_subprocess_runner = None


def get_subprocess_runner() -> SecureSubprocessRunner:
    """Get the global secure subprocess runner instance."""
    global _subprocess_runner
    if _subprocess_runner is None:
        _subprocess_runner = SecureSubprocessRunner()
    return _subprocess_runner


def run_kicad_command(
    command_args: list[str],
    input_files: list[str] | None = None,
    output_files: list[str] | None = None,
    working_dir: str | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess:
    """Convenience function to run KiCad command."""
    return get_subprocess_runner().run_kicad_command(
        command_args, input_files, output_files, working_dir, timeout
    )


async def run_kicad_command_async(
    command_args: list[str],
    input_files: list[str] | None = None,
    output_files: list[str] | None = None,
    working_dir: str | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess:
    """Convenience function to run KiCad command asynchronously."""
    return await get_subprocess_runner().run_kicad_command_async(
        command_args, input_files, output_files, working_dir, timeout
    )


def create_temp_file(
    suffix: str = "", prefix: str = "kicad_mcp_", content: str | None = None
) -> str:
    """Convenience function to create temporary file."""
    return get_subprocess_runner().create_temp_file(suffix, prefix, content)
