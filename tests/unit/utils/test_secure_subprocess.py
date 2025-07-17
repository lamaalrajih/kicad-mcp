"""
Tests for secure subprocess utility.
"""

import os
import subprocess
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from kicad_mcp.utils.kicad_cli import KiCadCLIError
from kicad_mcp.utils.path_validator import PathValidationError, PathValidator
from kicad_mcp.utils.secure_subprocess import (
    SecureSubprocessError,
    SecureSubprocessRunner,
    create_temp_file,
    run_kicad_command,
)


def _kicad_cli_available():
    """Check if KiCad CLI is available."""
    try:
        from kicad_mcp.utils.kicad_cli import get_kicad_cli_path

        get_kicad_cli_path()
        return True
    except Exception:
        return False


class TestSecureSubprocessRunner:
    """Test cases for SecureSubprocessRunner class."""

    def test_init_with_default_validator(self):
        """Test initialization with default path validator."""
        runner = SecureSubprocessRunner()
        assert runner.path_validator is not None
        assert runner.default_timeout == 30.0

    def test_init_with_custom_validator(self):
        """Test initialization with custom path validator."""
        validator = PathValidator(trusted_roots={"/tmp"})
        runner = SecureSubprocessRunner(path_validator=validator)
        assert runner.path_validator is validator

    @patch("kicad_mcp.utils.secure_subprocess.get_kicad_cli_path")
    @patch.object(SecureSubprocessRunner, "_run_subprocess")
    def test_run_kicad_command_success(self, mock_run_subprocess, mock_get_cli):
        """Test successful KiCad command execution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup mocks
            mock_get_cli.return_value = "/usr/bin/kicad-cli"
            mock_result = MagicMock(returncode=0, stdout="Success")
            mock_run_subprocess.return_value = mock_result

            # Create test files
            input_file = os.path.join(temp_dir, "input.kicad_sch")
            output_file = os.path.join(temp_dir, "output.svg")

            with open(input_file, "w") as f:
                f.write("test schematic")

            # Create runner with temp directory as trusted root
            validator = PathValidator(trusted_roots={temp_dir})
            runner = SecureSubprocessRunner(path_validator=validator)

            # Run command
            result = runner.run_kicad_command(
                ["sch", "export", "svg", input_file, "-o", output_file],
                input_files=[input_file],
                output_files=[output_file],
            )

            assert result is mock_result
            mock_run_subprocess.assert_called_once()

    @patch("kicad_mcp.utils.secure_subprocess.get_kicad_cli_path")
    def test_run_kicad_command_cli_not_found(self, mock_get_cli):
        """Test KiCad command when CLI not found."""
        mock_get_cli.side_effect = KiCadCLIError("CLI not found")

        runner = SecureSubprocessRunner()

        with pytest.raises(KiCadCLIError):
            runner.run_kicad_command(["--version"])

    @pytest.mark.skipif(not _kicad_cli_available(), reason="KiCad CLI not available")
    def test_run_kicad_command_invalid_input_file(self):
        """Test KiCad command with invalid input file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            validator = PathValidator(trusted_roots={temp_dir})
            runner = SecureSubprocessRunner(path_validator=validator)

            # Try to use file outside trusted directory
            with pytest.raises(PathValidationError):
                runner.run_kicad_command(
                    ["sch", "export", "svg", "/etc/passwd"], input_files=["/etc/passwd"]
                )

    @pytest.mark.skipif(not _kicad_cli_available(), reason="KiCad CLI not available")
    def test_run_kicad_command_invalid_output_directory(self):
        """Test KiCad command with invalid output directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            validator = PathValidator(trusted_roots={temp_dir})
            runner = SecureSubprocessRunner(path_validator=validator)

            # Try to output to directory outside trusted roots
            with pytest.raises(PathValidationError):
                runner.run_kicad_command(
                    ["sch", "export", "svg", "input.sch", "-o", "/etc/output.svg"],
                    output_files=["/etc/output.svg"],
                )

    @patch("kicad_mcp.utils.secure_subprocess.get_kicad_cli_path")
    @patch.object(SecureSubprocessRunner, "_run_subprocess")
    def test_run_kicad_command_with_working_dir(self, mock_run_subprocess, mock_get_cli):
        """Test KiCad command with working directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_get_cli.return_value = "/usr/bin/kicad-cli"
            mock_result = MagicMock(returncode=0)
            mock_run_subprocess.return_value = mock_result

            validator = PathValidator(trusted_roots={temp_dir})
            runner = SecureSubprocessRunner(path_validator=validator)

            runner.run_kicad_command(["--version"], working_dir=temp_dir)

            # Verify working directory was passed
            mock_run_subprocess.assert_called_once()
            call_args = mock_run_subprocess.call_args
            assert call_args[1]["working_dir"] == os.path.realpath(temp_dir)

    @patch("kicad_mcp.utils.secure_subprocess.get_kicad_cli_path")
    @patch.object(SecureSubprocessRunner, "_run_subprocess")
    def test_run_kicad_command_subprocess_error(self, mock_run_subprocess, mock_get_cli):
        """Test KiCad command with subprocess error."""
        mock_get_cli.return_value = "/usr/bin/kicad-cli"
        mock_run_subprocess.side_effect = subprocess.SubprocessError("Command failed")

        runner = SecureSubprocessRunner()

        with pytest.raises(SecureSubprocessError, match="KiCad command failed"):
            runner.run_kicad_command(["--version"])

    @pytest.mark.asyncio
    @patch("kicad_mcp.utils.secure_subprocess.get_kicad_cli_path")
    @patch.object(SecureSubprocessRunner, "run_kicad_command")
    async def test_run_kicad_command_async(self, mock_run_command, mock_get_cli):
        """Test async KiCad command execution."""
        mock_get_cli.return_value = "/usr/bin/kicad-cli"
        mock_result = MagicMock(returncode=0)
        mock_run_command.return_value = mock_result

        runner = SecureSubprocessRunner()

        # Run the async function in a synchronous context for testing
        result = await runner.run_kicad_command_async(["--version"])
        assert result is mock_result

    def test_run_safe_command_success(self):
        """Test successful safe command execution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            validator = PathValidator(trusted_roots={temp_dir})
            runner = SecureSubprocessRunner(path_validator=validator)

            with patch.object(runner, "_run_subprocess") as mock_run:
                mock_result = MagicMock(returncode=0)
                mock_run.return_value = mock_result

                result = runner.run_safe_command(["echo", "test"], allowed_commands=["echo"])

                assert result is mock_result

    def test_run_safe_command_empty_command(self):
        """Test safe command with empty command list."""
        runner = SecureSubprocessRunner()

        with pytest.raises(SecureSubprocessError, match="Command cannot be empty"):
            runner.run_safe_command([])

    def test_run_safe_command_not_in_whitelist(self):
        """Test safe command not in whitelist."""
        runner = SecureSubprocessRunner()

        with pytest.raises(SecureSubprocessError, match="not in allowed list"):
            runner.run_safe_command(["rm", "-rf", "/"], allowed_commands=["echo", "ls"])

    def test_run_safe_command_invalid_working_dir(self):
        """Test safe command with invalid working directory."""
        runner = SecureSubprocessRunner()

        with pytest.raises(PathValidationError):
            runner.run_safe_command(["echo", "test"], working_dir="/nonexistent/directory")

    def test_create_temp_file_without_content(self):
        """Test temporary file creation without content."""
        with tempfile.TemporaryDirectory() as temp_dir:
            validator = PathValidator(trusted_roots={temp_dir})
            runner = SecureSubprocessRunner(path_validator=validator)

            temp_path = runner.create_temp_file(suffix=".txt", prefix="test_")

            assert os.path.exists(temp_path)
            assert temp_path.endswith(".txt")
            assert "test_" in os.path.basename(temp_path)

            # Cleanup
            os.unlink(temp_path)

    def test_create_temp_file_with_content(self):
        """Test temporary file creation with content."""
        with tempfile.TemporaryDirectory() as temp_dir:
            validator = PathValidator(trusted_roots={temp_dir})
            runner = SecureSubprocessRunner(path_validator=validator)

            content = "test content"
            temp_path = runner.create_temp_file(content=content)

            assert os.path.exists(temp_path)

            with open(temp_path) as f:
                assert f.read() == content

            # Cleanup
            os.unlink(temp_path)

    def test_run_subprocess_with_capture_output(self):
        """Test subprocess execution with output capture."""
        runner = SecureSubprocessRunner()

        result = runner._run_subprocess(["echo", "test"], capture_output=True, timeout=5.0)

        assert result.returncode == 0
        assert "test" in result.stdout

    def test_run_subprocess_without_capture_output(self):
        """Test subprocess execution without output capture."""
        runner = SecureSubprocessRunner()

        result = runner._run_subprocess(["echo", "test"], capture_output=False, timeout=5.0)

        assert result.returncode == 0
        # stdout/stderr should be None when not captured
        assert result.stdout is None
        assert result.stderr is None

    def test_run_subprocess_timeout(self):
        """Test subprocess timeout."""
        runner = SecureSubprocessRunner()

        with pytest.raises(subprocess.TimeoutExpired):
            runner._run_subprocess(["sleep", "10"], timeout=0.1)


class TestConvenienceFunctions:
    """Test convenience functions."""

    @patch.object(SecureSubprocessRunner, "run_kicad_command")
    def test_run_kicad_command_convenience(self, mock_run_command):
        """Test run_kicad_command convenience function."""
        mock_result = MagicMock(returncode=0)
        mock_run_command.return_value = mock_result

        result = run_kicad_command(["--version"])
        assert result is mock_result

    @patch.object(SecureSubprocessRunner, "create_temp_file")
    def test_create_temp_file_convenience(self, mock_create_temp):
        """Test create_temp_file convenience function."""
        mock_create_temp.return_value = "/tmp/test_file.txt"

        result = create_temp_file(suffix=".txt", prefix="test_")
        assert result == "/tmp/test_file.txt"
