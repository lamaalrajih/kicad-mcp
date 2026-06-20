# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python MCP (Model Context Protocol) server enabling AI assistants to interact with KiCad EDA projects. Built with **FastMCP 2.x**, managed with **uv**, and using **Hatchling** as the build backend. Python >= 3.10.

## Commands

```bash
make install               # uv sync --group dev (creates .venv/)
make run                   # Start MCP server (uv run python main.py)
make test                  # Run all tests with pytest (80% coverage threshold)
make test tests/unit/utils/test_secure_subprocess.py  # Run a single test file
make lint                  # ruff check + mypy
make format                # ruff format
make build                 # uv build
```

For direct uv usage without make:
```bash
uv run pytest tests/unit/utils/test_secure_subprocess.py -v   # Single test file
uv run pytest -k "test_command_whitelist"                      # Single test by name
uv run pytest -m unit                                          # Tests by marker
uv run ruff check --fix kicad_mcp/                             # Auto-fix lint issues
```

## Architecture

**Startup flow**: `main.py` → configures logging + loads `.env` → calls `server_main()` → `kicad_mcp/server.py:create_server()` creates a `FastMCP("KiCad")` instance and registers all tools/resources/prompts.

**Registration pattern**: Each module has a `register_*(mcp: FastMCP)` function. Inside, functions are decorated with `@mcp.tool()`, `@mcp.resource("kicad://...")`, or `@mcp.prompt()` to register them with the server. New features follow this same pattern.

**Key modules**:
- `kicad_mcp/config.py` — Platform-specific KiCad paths, search directories (from env + auto-detect), constants
- `kicad_mcp/context.py` — `KiCadAppContext` dataclass + `kicad_lifespan` async context manager
- `kicad_mcp/server.py` — `create_server()` wires everything together, signal handlers, cleanup

**Tool modules** (`kicad_mcp/tools/`): `project_tools` (find/open projects), `analysis_tools` (validate), `export_tools` (PCB thumbnails via kicad-cli), `drc_tools` (DRC checks + history), `bom_tools` (BOM parse/export), `netlist_tools` (S-expression netlist extraction), `pattern_tools` (circuit pattern recognition), `validation_tools` (boundary checking — **note: not registered in server**).

**Utility layer** (`kicad_mcp/utils/`): `netlist_parser.py` (SchematicParser for S-expressions), `pattern_recognition.py` (power supplies, amplifiers, filters, oscillators, digital interfaces, MCUs, sensors), `secure_subprocess.py` (whitelist-based subprocess runner), `path_validator.py` (traversal prevention, KiCad file validation), `kicad_cli.py` (KiCadCLIManager with caching).

## Adding New Tools/Resources/Prompts

1. Create implementation in the appropriate `kicad_mcp/tools/`, `kicad_mcp/resources/`, or `kicad_mcp/prompts/` module
2. Add a `register_*(mcp)` function with `@mcp.tool()` / `@mcp.resource()` / `@mcp.prompt()` decorators
3. Import and call the register function in `kicad_mcp/server.py:create_server()`
4. The `ctx: Context | None` pattern is used throughout to handle null context (workaround for FastMCP issue)

## Configuration

Environment variables (set in `.env` or shell):
- `KICAD_SEARCH_PATHS` — comma-separated extra directories to search for `.kicad_pro` files
- `KICAD_USER_DIR` — override default KiCad user directory
- `KICAD_APP_PATH` — override KiCad application path
- `KICAD_CLI_PATH` — override kicad-cli binary location

## Testing

Tests in `tests/unit/utils/` — currently `test_secure_subprocess.py` and `test_path_validator.py`. Pytest config in `pyproject.toml` requires 80% coverage (30% in CI). Available markers: `unit`, `integration`, `slow`, `requires_kicad`, `performance`.

## Known Issues

- `validation_tools.py` defines `register_validation_tools()` but it is **never called** from `server.py`, so those tools are unavailable
- `boundary_validator.py` imports from nonexistent modules (`component_layout`, `coordinate_converter`) — would fail if validation_tools were registered
- Three separate kicad-cli detection implementations exist: `drc_impl/cli_drc.py`, `kicad_cli.py` (KiCadCLIManager), and `kicad_api_detection.py`
- `netlist_parser.py:SchematicParser._build_netlist()` is incomplete — doesn't trace wire connections
- The former KiCad Python module API has been removed; all KiCad interaction now goes through `kicad-cli` subprocess calls
