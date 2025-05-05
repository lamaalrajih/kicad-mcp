# KiCad MCP Server

[![smithery badge](https://smithery.ai/badge/@lamaalrajih/kicad-mcp)](https://smithery.ai/server/@lamaalrajih/kicad-mcp)

This guide will help you set up a Model Context Protocol (MCP) server for KiCad. While the examples in this guide often reference Claude Desktop, the server is compatible with **any MCP-compliant client**. You can use it with Claude Desktop, your own custom MCP clients, or any other application that implements the Model Context Protocol.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation Steps](#installation-steps)
- [Understanding MCP Components](#understanding-mcp-components)
- [Feature Highlights](#feature-highlights)
- [Natural Language Interaction](#natural-language-interaction)
- [Documentation](#documentation)
- [Configuration](#configuration)
- [Development Guide](#development-guide)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Future Development Ideas](#future-development-ideas)
- [License](#license)

## Prerequisites

- macOS, Windows, or Linux
- Python 3.10 or higher
- KiCad 9.0 or higher
- Claude Desktop (or another MCP client)

## Installation Steps

### Installing via Smithery

To install KiCad MCP Server for Claude Desktop automatically via [Smithery](https://smithery.ai/server/@lamaalrajih/kicad-mcp):

```bash
npx @smithery/cli -y install @lamaalrajih/kicad-mcp --client claude
```

### 1. Set Up Your Python Environment

First, let's install dependencies and set up our environment:

```bash
# Clone the repository
git clone https://github.com/lamaalrajih/kicad-mcp.git .

# Create a virtual environment and activate it
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the MCP SDK and other dependencies
pip install -r requirements.txt
```

### 2. Configure Your Environment

Create a `.env` file to customize where the server looks for your KiCad projects:

```bash
# Copy the example environment file
cp .env.example .env

# Edit the .env file
vim .env
```

In the `.env` file, add your custom project directories:

```
# Add paths to your KiCad projects (comma-separated)
KICAD_SEARCH_PATHS=~/pcb,~/Electronics,~/Projects/KiCad
```

### 3. Run the Server

Once the environment is set up, you can run the server:

```bash
python main.py
```

### 4. Configure an MCP Client

Now, let's configure Claude Desktop to use our MCP server:

1. Create or edit the Claude Desktop configuration file:

```bash
# Create the directory if it doesn't exist
mkdir -p ~/Library/Application\ Support/Claude

# Edit the configuration file
vim ~/Library/Application\ Support/Claude/claude_desktop_config.json
```

2. Add the KiCad MCP server to the configuration:

```json
{
    "mcpServers": {
        "kicad": {
            "command": "/ABSOLUTE/PATH/TO/YOUR/PROJECT/kicad-mcp/venv/bin/python",
            "args": [
                "/ABSOLUTE/PATH/TO/YOUR/PROJECT/kicad-mcp/main.py"
            ]
        }
    }
}
```

Replace `/ABSOLUTE/PATH/TO/YOUR/PROJECT/kicad-mcp` with the actual path to your project directory.

### 5. Restart Your MCP Client

Close and reopen your MCP client to load the new configuration.

## Understanding MCP Components

The Model Context Protocol (MCP) defines three primary ways to provide capabilities:

### Resources vs Tools vs Prompts

**Resources** are read-only data sources that LLMs can reference:
- Similar to GET endpoints in REST APIs
- Provide data without performing significant computation
- Used when the LLM needs to read information
- Typically accessed programmatically by the client application
- Example: `kicad://projects` returns a list of all KiCad projects

**Tools** are functions that perform actions or computations:
- Similar to POST/PUT endpoints in REST APIs
- Can have side effects (like opening applications or generating files)
- Used when the LLM needs to perform actions in the world
- Typically invoked directly by the LLM (with user approval)
- Example: `open_project()` launches KiCad with a specific project

**Prompts** are reusable templates for common interactions:
- Pre-defined conversation starters or instructions
- Help users articulate common questions or tasks
- Invoked by user choice (typically from a menu)
- Example: The `debug_pcb_issues` prompt helps users troubleshoot PCB problems

For more information on resources vs tools vs prompts, read the [MCP docs](https://modelcontextprotocol.io/docs/concepts/architecture).

## Feature Highlights

The KiCad MCP Server provides several key features, each with detailed documentation:

- **Project Management**: List, examine, and open KiCad projects
  - *Example:* "Show me all my recent KiCad projects" → Lists all projects sorted by modification date
  
- **PCB Design Analysis**: Get insights about your PCB designs and schematics
  - *Example:* "Analyze the component density of my temperature sensor board" → Provides component spacing analysis
  
- **Netlist Extraction**: Extract and analyze component connections from schematics
  - *Example:* "What components are connected to the MCU in my Arduino shield?" → Shows all connections to the microcontroller
  
- **BOM Management**: Analyze and export Bills of Materials
  - *Example:* "Generate a BOM for my smart watch project" → Creates a detailed bill of materials
  
  - **Design Rule Checking**: Run DRC checks using the KiCad CLI and track your progress over time
  - *Example:* "Run DRC on my power supply board and compare to last week" → Shows progress in fixing violations

- **PCB Visualization**: Generate visual representations of your PCB layouts
  - *Example:* "Show me a thumbnail of my audio amplifier PCB" → Displays a visual render of the board
  
- **Circuit Pattern Recognition**: Automatically identify common circuit patterns in your schematics
  - *Example:* "What power supply topologies am I using in my IoT device?" → Identifies buck, boost, or linear regulators

For more examples and details on each feature, see the dedicated guides in the documentation. You can also ask the LLM what tools it has access to!

## Natural Language Interaction

While our documentation often shows examples like:

```
Show me the DRC report for /Users/username/Documents/KiCad/my_project/my_project.kicad_pro
```

You don't need to type the full path to your files! The LLM can understand more natural language requests.

For example, instead of the formal command above, you can simply ask:

```
Can you check if there are any design rule violations in my Arduino shield project?
```

Or:

```
I'm working on the temperature sensor circuit. Can you identify what patterns it uses?
```

The LLM will understand your intent and request the relevant information from the KiCad MCP Server. If it needs clarification about which project you're referring to, it will ask.

## Documentation

Detailed documentation for each feature is available in the `docs/` directory:

- [Project Management](docs/project_guide.md)
- [PCB Design Analysis](docs/analysis_guide.md)
- [Netlist Extraction](docs/netlist_guide.md)
- [Bill of Materials (BOM)](docs/bom_guide.md)
- [Design Rule Checking (DRC)](docs/drc_guide.md)
- [PCB Visualization](docs/thumbnail_guide.md)
- [Circuit Pattern Recognition](docs/pattern_guide.md)
- [Prompt Templates](docs/prompt_guide.md)

## Configuration

The KiCad MCP Server can be configured using environment variables or a `.env` file:

### Key Configuration Options
| Environment Variable | Description | Example |
|---------------------|-------------|---------|
| `KICAD_SEARCH_PATHS` | Comma-separated list of directories to search for KiCad projects | `~/pcb,~/Electronics,~/Projects` |
| `KICAD_USER_DIR` | Override the default KiCad user directory | `~/Documents/KiCadProjects` |
| `KICAD_APP_PATH` | Override the default KiCad application path | `/Applications/KiCad7/KiCad.app` |

See [Configuration Guide](docs/configuration.md) for more details.

## Development Guide

### Project Structure

The KiCad MCP Server is organized into a modular structure:

```
kicad-mcp/
├── README.md                       # Project documentation
├── main.py                         # Entry point that runs the server
├── requirements.txt                # Python dependencies
├── .env.example                    # Example environment configuration
├── kicad_mcp/                      # Main package directory
│   ├── __init__.py
│   ├── server.py                   # MCP server setup
│   ├── config.py                   # Configuration constants and settings
│   ├── context.py                  # Lifespan management and shared context
│   ├── resources/                  # Resource handlers
│   ├── tools/                      # Tool handlers
│   ├── prompts/                    # Prompt templates
│   └── utils/                      # Utility functions
├── docs/                           # Documentation
└── tests/                          # Unit tests
```

### Adding New Features

To add new features to the KiCad MCP Server, follow these steps:

1. Identify the category for your feature (resource, tool, or prompt)
2. Add your implementation to the appropriate module
3. Register your feature in the corresponding register function
4. Test your changes with the development tools

See [Development Guide](docs/development.md) for more details.

## Troubleshooting

If you encounter issues:

1. **Server Not Appearing in MCP Client:**
   - Check your client's configuration file for errors
   - Make sure the path to your project and Python interpreter is correct
   - Ensure Python can access the `mcp` package
   - Check if your KiCad installation is detected

2. **Server Errors:**
   - Check the terminal output when running the server in development mode
   - Check Claude logs at:
     - `~/Library/Logs/Claude/mcp-server-kicad.log` (server-specific logs)
     - `~/Library/Logs/Claude/mcp.log` (general MCP logs)

3. **Working Directory Issues:**
   - The working directory for servers launched via client configs may be undefined
   - Always use absolute paths in your configuration and .env files
   - For testing servers via command line, the working directory will be where you run the command

See [Troubleshooting Guide](docs/troubleshooting.md) for more details.

If you're still not able to troubleshoot, please open a Github issue. 

## Contributing

Want to contribute to the KiCad MCP Server? Here's how you can help improve this project:

1. Fork the repository
2. Create a feature branch
3. Add your changes
4. Submit a pull request

Key areas for contribution:
- Adding support for more component patterns in the Circuit Pattern Recognition system
- Improving documentation and examples
- Adding new features or enhancing existing ones
- Fixing bugs and improving error handling

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed contribution guidelines.

## Future Development Ideas

Interested in contributing? Here are some ideas for future development:

1. **3D Model Visualization** - Implement tools to visualize 3D models of PCBs
2. **PCB Review Tools** - Create annotation features for design reviews
3. **Manufacturing File Generation** - Add support for generating Gerber files and other manufacturing outputs
4. **Component Search** - Implement search functionality for components across KiCad libraries
5. **BOM Enhancement** - Add supplier integration for component sourcing and pricing
6. **Interactive Design Checks** - Develop interactive tools for checking design quality
7. **Web UI** - Create a simple web interface for configuration and monitoring
8. **Circuit Analysis** - Add automated circuit analysis features
9. **Test Coverage** - Improve test coverage across the codebase
10. **Circuit Pattern Recognition** - Expand the pattern database with more component types and circuit topologies

## License

This project is open source under the MIT license.
