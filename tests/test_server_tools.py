#!/usr/bin/env python3
"""Test that MCP server has schematic DSL tools registered"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from kicad_mcp.server import create_server

# Create server
print("Creating KiCAD MCP server...")
server = create_server()

# Get registered tools
tools = server._mcp_server.list_tools()
# If it's a coroutine, we need to await it, but let's just access the tools dict
tools_dict = server._mcp_server._tool_manager._tools if hasattr(server._mcp_server, '_tool_manager') else {}
if not tools_dict:
    # FastMCP 2.x stores tools differently
    tools_dict = server._tools if hasattr(server, '_tools') else {}
tools = list(tools_dict.values()) if tools_dict else []

print(f"\nTotal tools registered: {len(tools)}")
print("\nSearching for schematic DSL tools:")
print("-" * 60)

dsl_tools = [
    "get_schematic_index",
    "get_schematic_page",
    "get_schematic_context"
]

found = []
missing = []

for tool_name in dsl_tools:
    tool = next((t for t in tools if t.name == tool_name), None)
    if tool:
        found.append(tool_name)
        print(f"[FOUND] {tool_name}")
        print(f"  Description: {tool.description[:80]}...")
    else:
        missing.append(tool_name)
        print(f"[MISSING] {tool_name}")

print("\n" + "=" * 60)
if missing:
    print(f"PROBLEM: {len(missing)} DSL tools missing: {missing}")
    sys.exit(1)
else:
    print(f"SUCCESS: All {len(found)} schematic DSL tools registered!")
    print("\nServer is ready to use with schematic DSL functionality.")
