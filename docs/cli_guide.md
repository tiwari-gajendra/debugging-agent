# Debugging Agents CLI Guide

This guide explains how to properly use the Debugging Agents CLI tool, including how to ensure imports work correctly when running the script from various locations.

## Basic Usage

The Debugging Agents CLI provides several commands for debugging and forecasting:

```bash
debug-agent info
debug-agent debug ISSUE-123
debug-agent forecast
```

## Installation Options

There are several ways to use the Debugging Agents CLI:

### Option 1: Install as a Package (Recommended)

The most reliable way to use the CLI is to install it as a Python package:

```bash
cd /path/to/debugging-agents
pip install -e .
```

This will install the CLI tool in development mode, making the `debug-agent` command available from any directory:

```bash
# Run from any directory
debug-agent info
debug-agent debug ISSUE-123 --open-doc
debug-agent forecast --verbose
```

Benefits of this approach:
- No path issues - imports work correctly from any directory
- Simple commands without specifying the full path
- Changes to the code are immediately reflected (due to the `-e` flag)

### Option 2: Use the Simple CLI Script

We provide a simple entry point script in the project root that handles Python path setup automatically:

```bash
cd /path/to/debugging-agents
python cli.py info
# Or make it executable
chmod +x cli.py
./cli.py info
```

This is the easiest way to run commands directly without installing the package.

### Option 3: Running from Project Root

If you prefer the original script, you can run it directly from the project root directory:

```bash
cd /path/to/debugging-agents
python debug_agent_cli.py <command>
```

This works because the script adds the project root to the Python path automatically.

### Option 4: Running from Any Directory

If you need to run the script from any directory without installing it, use the full path to the script:

```bash
python /path/to/debugging-agents/debug_agent_cli.py <command>
```

The script will automatically determine its location and set the Python path accordingly.

### Option 5: Setting PYTHONPATH Manually

You can also set the PYTHONPATH environment variable before running the script:

```bash
# For bash/zsh
export PYTHONPATH=/path/to/debugging-agents:$PYTHONPATH
python /path/to/debugging-agents/debug_agent_cli.py <command>

# For Windows Command Prompt
set PYTHONPATH=C:\path\to\debugging-agents;%PYTHONPATH%
python C:\path\to\debugging-agents\debug_agent_cli.py <command>
```

## Command Reference

### Debug Command

```bash
debug-agent debug ISSUE-ID [options]
```

Options:
- `--time-window N` - Time window in minutes to look back (default: 60)
- `--open-doc` - Open document in browser when done
- `--llm-provider {openai,ollama,bedrock}` - LLM provider to use

Example:
```bash
debug-agent debug CPU-HIGH-123 --open-doc --llm-provider ollama
```

### Forecast Command

```bash
debug-agent forecast [options]
```

Options:
- `--verbose` or `-v` - Show detailed output

Example:
```bash
debug-agent forecast --verbose
```

### Info Command

```bash
debug-agent info
```

Displays system information including version, Python version, and LLM configuration.

## How the Path Setup Works

The CLI script uses the following code to ensure proper imports:

```python
import os
import sys

# Get the directory containing the script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the project root (the directory containing the script)
project_root = os.path.dirname(script_dir)
# Add the project root to the Python path
sys.path.insert(0, project_root)
```

This ensures that the script can import modules from the project regardless of where it's run from.

## Package Structure

When installed as a package, the project has the following structure:

```
debugging-agents/
├── cli.py                       # Simple CLI entry point
├── debugging_agents/            # Python package directory
│   ├── __init__.py              # Package initialization
│   └── debug_agent_cli.py       # CLI module
├── src/                         # Source code
│   ├── __init__.py
│   ├── main.py
│   ├── realtime/
│   └── forecasting/
├── setup.py                     # Package setup script
└── ...
```

The entry point is defined in `setup.py`, which allows the `debug-agent` command to work correctly.

## Troubleshooting Import Issues

If you encounter import errors when running the CLI script, try the following:

1. Verify that you're using the correct Python environment (virtualenv, conda, etc.)
2. Install the package using `pip install -e .` (easiest solution)
3. Use the simple CLI script: `python cli.py <command>`
4. Check that the path to the project root is correct
5. Try setting the PYTHONPATH manually as described above
6. Run the script with the `--verbose` flag for more information:

```bash
debug-agent --verbose info
```

## Common Error Messages

### ModuleNotFoundError

If you see an error like:

```
ModuleNotFoundError: No module named 'src'
```

This indicates that Python cannot find the project modules. Install the package using `pip install -e .` or use one of the approaches described above to fix the path.

### ImportError

If you see an error like:

```
ImportError: cannot import name 'ContextBuilder' from 'src.realtime'
```

This indicates that Python found the module but couldn't import a specific class or function. Check that the class exists and that you're using the correct import path. 