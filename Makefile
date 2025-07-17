.PHONY: help install test lint format clean build

help:
	@echo "Available commands:"
	@echo "  install     Install dependencies"
	@echo "  test        Run tests"
	@echo "  lint        Run linting"
	@echo "  format      Format code"
	@echo "  clean       Clean build artifacts"
	@echo "  build       Build package"

install:
	uv sync --group dev

test:
	uv run python -m pytest tests/ -v

lint:
	uv run ruff check kicad_mcp/ tests/
	uv run mypy kicad_mcp/

format:
	uv run ruff format kicad_mcp/ tests/

clean:
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -f coverage.xml
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

build:
	uv build
