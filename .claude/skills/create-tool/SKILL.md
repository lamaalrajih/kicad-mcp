---
name: create-tool
description: Scaffold new kicad-mcp tools with guided workflow. Use when user wants to add new MCP tools or extend kicad-mcp functionality.
allowed-tools: Read, Glob, Grep, Write, Edit, Bash(uv run ruff:*), Bash(uv run pytest:*), Bash(uv run mypy:*)
model: sonnet
context: fork
agent: general-purpose
user-invocable: true
---

# Create Tool Skill

You help users create new kicad-mcp tools following established patterns.

## References

Before starting, read these files to understand the codebase patterns:

1. **Style Guide**: Read `/.local/STYLE_GUIDE.md` for code patterns and common mistakes
2. **Templates**: Read `TEMPLATES.md` (in this directory) for skeleton code
3. **Checklist**: Read `CHECKLIST.md` (in this directory) for registration steps

## Workflow

### Phase 1: Requirements Gathering

Ask the user these questions (use the AskUserQuestion tool):

1. **Purpose**: What should this tool do? What problem does it solve?
2. **Name**: What's the tool name? (e.g., `analyze_traces`, `export_gerbers`)
3. **Category**: Which category fits best?
   - `project` - Project listing/management
   - `analysis` - Design analysis
   - `drc` - Design rule checking
   - `bom` - Bill of materials
   - `netlist` - Netlist extraction
   - `pattern` - Circuit pattern recognition
   - `export` - Export functionality
   - `validation` - Component validation
4. **Parameters**: What input parameters? (types, required vs optional)
5. **Async**: Does it need progress reporting? (async if yes, sync if no)

### Phase 2: Pattern Exploration

Explore the existing codebase to find patterns:

1. Search `kicad_mcp/tools/` for similar tools
2. Read the most relevant existing tool file
3. Check `kicad_mcp/utils/` for reusable utilities
4. Note the import patterns, return formats, and error handling

Tell the user what patterns you found and how you'll apply them.

### Phase 3: Preview Generation

**IMPORTANT**: Before writing any files, show the user a preview of what will be created:

```
## Preview: Files to Create/Modify

### 1. Tool File: `kicad_mcp/tools/<category>_tools.py`
[Show the tool function code]

### 2. Utility File (if needed): `kicad_mcp/utils/<utility>.py`
[Show any helper utilities]

### 3. Tool Tests (REQUIRED): `tests/unit/tools/test_<category>_tools.py`
[Show test classes with test methods for each tool function]

### 4. Utility Tests (if utilities created): `tests/unit/utils/test_<utility>.py`
[Show test classes for utility functions]

### 5. Server Registration: `kicad_mcp/server.py`
[Show the import and registration lines to add]

### 6. Dependencies (if any): `pyproject.toml`
[Show any new dependencies needed]
```

Ask the user to confirm before proceeding.

### Phase 4: Confirmation & Generation

After user approves the preview:

1. **Generate/modify tool file** - Add the new tool function
2. **Generate test file** - Create comprehensive unit tests (MANDATORY)
3. **Generate utility tests** - If you created utility files (MANDATORY)
4. **Update server.py** - Add import and registration
5. **Update pyproject.toml** - Add dependencies if needed

Use the Write tool for new files, Edit tool for modifications.

**Do not skip test creation.** Every tool needs tests before validation.

### Phase 5: Validation

Run the verification commands and report results:

```bash
uv run ruff check . && uv run ruff format .
uv run mypy kicad_mcp/
uv run pytest tests/unit/ -v
```

If any checks fail:
1. Show the error
2. Explain what's wrong
3. Fix it automatically
4. Re-run validation

## Important Guidelines

- **Always read STYLE_GUIDE.md first** - It contains critical patterns
- **One tool at a time** - Focus on quality over quantity
- **Preview before writing** - Never write without user confirmation
- **Follow existing patterns** - Match the style of similar tools
- **Include proper error handling** - Return `{"success": False, "error": "..."}` on failure
- **Add progress reporting** - Use `await ctx.report_progress()` for async tools
- **Use print() for logging** - Not the logging module (per style guide)

## CRITICAL: Tests Are Mandatory

**Every new tool MUST have unit tests.** This is not optional.

For each tool you create, you MUST also create:

1. **Tool tests** in `tests/unit/tools/test_<category>_tools.py`:
   - Test success cases
   - Test error handling (invalid inputs, missing files)
   - Test edge cases
   - Use mocks for external dependencies

2. **Utility tests** (if you create utilities) in `tests/unit/utils/test_<utility>.py`:
   - Test each helper function
   - Test cache operations if applicable
   - Test file parsing if applicable

Look at existing tests for patterns:
- `tests/unit/tools/test_symbol_tools.py` - Tool testing patterns
- `tests/unit/utils/test_symbol_cache.py` - Cache utility testing patterns

Run tests after creation:
```bash
uv run pytest tests/unit/tools/test_<your_tool>.py -v --no-cov
```
