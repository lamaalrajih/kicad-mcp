"""
Tests for path validation utility.
"""

import os
import tempfile

import pytest

from kicad_mcp.utils.path_validator import (
    PathValidationError,
    PathValidator,
    validate_directory,
    validate_kicad_file,
    validate_path,
)


class TestPathValidator:
    """Test cases for PathValidator class."""

    def test_init_with_default_trusted_root(self):
        """Test initialization with default trusted root."""
        validator = PathValidator()
        assert len(validator.trusted_roots) == 1
        assert os.getcwd() in [os.path.realpath(root) for root in validator.trusted_roots]

    def test_init_with_custom_trusted_roots(self):
        """Test initialization with custom trusted roots."""
        roots = {"/tmp", "/home/user"}
        validator = PathValidator(trusted_roots=roots)

        # Should normalize paths
        expected_roots = {os.path.realpath(root) for root in roots}
        assert validator.trusted_roots == expected_roots

    def test_add_trusted_root(self):
        """Test adding trusted root."""
        validator = PathValidator(trusted_roots={"/tmp"})
        validator.add_trusted_root("/home/user")

        assert os.path.realpath("/home/user") in validator.trusted_roots

    def test_validate_path_success(self):
        """Test successful path validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            validator = PathValidator(trusted_roots={temp_dir})
            test_file = os.path.join(temp_dir, "test.txt")

            # Create test file
            with open(test_file, "w") as f:
                f.write("test")

            # Should succeed
            result = validator.validate_path(test_file, must_exist=True)
            assert result == os.path.realpath(test_file)

    def test_validate_path_traversal_attack(self):
        """Test path traversal attack prevention."""
        with tempfile.TemporaryDirectory() as temp_dir:
            validator = PathValidator(trusted_roots={temp_dir})

            # Try to access parent directory
            malicious_path = os.path.join(temp_dir, "..", "..", "etc", "passwd")

            with pytest.raises(PathValidationError, match="outside trusted directories"):
                validator.validate_path(malicious_path)

    def test_validate_path_empty_string(self):
        """Test validation with empty string."""
        validator = PathValidator()

        with pytest.raises(PathValidationError, match="non-empty string"):
            validator.validate_path("")

    def test_validate_path_none(self):
        """Test validation with None."""
        validator = PathValidator()

        with pytest.raises(PathValidationError, match="non-empty string"):
            validator.validate_path(None)

    def test_validate_path_nonexistent_when_required(self):
        """Test validation of nonexistent file when existence required."""
        with tempfile.TemporaryDirectory() as temp_dir:
            validator = PathValidator(trusted_roots={temp_dir})
            nonexistent_file = os.path.join(temp_dir, "nonexistent.txt")

            with pytest.raises(PathValidationError, match="does not exist"):
                validator.validate_path(nonexistent_file, must_exist=True)

    def test_validate_kicad_file_success(self):
        """Test successful KiCad file validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            validator = PathValidator(trusted_roots={temp_dir})
            project_file = os.path.join(temp_dir, "test.kicad_pro")

            # Create test file
            with open(project_file, "w") as f:
                f.write("{}")

            result = validator.validate_kicad_file(project_file, "project")
            assert result == os.path.realpath(project_file)

    def test_validate_kicad_file_wrong_extension(self):
        """Test KiCad file validation with wrong extension."""
        with tempfile.TemporaryDirectory() as temp_dir:
            validator = PathValidator(trusted_roots={temp_dir})
            wrong_file = os.path.join(temp_dir, "test.txt")

            with open(wrong_file, "w") as f:
                f.write("test")

            with pytest.raises(PathValidationError, match="must have .kicad_pro extension"):
                validator.validate_kicad_file(wrong_file, "project")

    def test_validate_kicad_file_unknown_type(self):
        """Test KiCad file validation with unknown file type."""
        with tempfile.TemporaryDirectory() as temp_dir:
            validator = PathValidator(trusted_roots={temp_dir})
            test_file = os.path.join(temp_dir, "test.txt")

            with open(test_file, "w") as f:
                f.write("test")

            with pytest.raises(PathValidationError, match="Unknown KiCad file type"):
                validator.validate_kicad_file(test_file, "unknown_type")

    def test_validate_directory_success(self):
        """Test successful directory validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            validator = PathValidator(trusted_roots={temp_dir})
            sub_dir = os.path.join(temp_dir, "subdir")
            os.makedirs(sub_dir)

            result = validator.validate_directory(sub_dir)
            assert result == os.path.realpath(sub_dir)

    def test_validate_directory_not_directory(self):
        """Test directory validation on file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            validator = PathValidator(trusted_roots={temp_dir})
            test_file = os.path.join(temp_dir, "test.txt")

            with open(test_file, "w") as f:
                f.write("test")

            with pytest.raises(PathValidationError, match="not a directory"):
                validator.validate_directory(test_file)

    def test_validate_project_directory(self):
        """Test project directory validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            validator = PathValidator(trusted_roots={temp_dir})
            project_file = os.path.join(temp_dir, "test.kicad_pro")

            with open(project_file, "w") as f:
                f.write("{}")

            result = validator.validate_project_directory(project_file)
            assert result == os.path.realpath(temp_dir)

    def test_create_safe_temp_path(self):
        """Test safe temporary path creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            validator = PathValidator(trusted_roots={temp_dir})

            temp_path = validator.create_safe_temp_path("test", ".txt")

            # Should be within trusted directory (handle symlinks with realpath)
            assert os.path.realpath(temp_path).startswith(os.path.realpath(temp_dir))
            assert temp_path.endswith(".txt")
            assert "test" in os.path.basename(temp_path)

    def test_symlink_resolution(self):
        """Test symbolic link resolution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            validator = PathValidator(trusted_roots={temp_dir})

            # Create file and symlink
            real_file = os.path.join(temp_dir, "real.txt")
            link_file = os.path.join(temp_dir, "link.txt")

            with open(real_file, "w") as f:
                f.write("test")

            os.symlink(real_file, link_file)

            # Both should resolve to same real path
            real_result = validator.validate_path(real_file, must_exist=True)
            link_result = validator.validate_path(link_file, must_exist=True)

            assert real_result == link_result == os.path.realpath(real_file)


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_validate_path_convenience(self):
        """Test validate_path convenience function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Add temp_dir to default validator
            from kicad_mcp.utils.path_validator import get_default_validator

            get_default_validator().add_trusted_root(temp_dir)

            test_file = os.path.join(temp_dir, "test.txt")
            with open(test_file, "w") as f:
                f.write("test")

            result = validate_path(test_file, must_exist=True)
            assert result == os.path.realpath(test_file)

    def test_validate_kicad_file_convenience(self):
        """Test validate_kicad_file convenience function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Add temp_dir to default validator
            from kicad_mcp.utils.path_validator import get_default_validator

            get_default_validator().add_trusted_root(temp_dir)

            project_file = os.path.join(temp_dir, "test.kicad_pro")
            with open(project_file, "w") as f:
                f.write("{}")

            result = validate_kicad_file(project_file, "project")
            assert result == os.path.realpath(project_file)

    def test_validate_directory_convenience(self):
        """Test validate_directory convenience function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Add temp_dir to default validator
            from kicad_mcp.utils.path_validator import get_default_validator

            get_default_validator().add_trusted_root(temp_dir)

            result = validate_directory(temp_dir)
            assert result == os.path.realpath(temp_dir)
