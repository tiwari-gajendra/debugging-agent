#!/usr/bin/env python3
"""
Simple entry point to the Debug Agent CLI
This script forwards execution to the actual CLI implementation.
"""

import os
import sys

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import and run the CLI
from debugging_agents.debug_agent_cli import main

if __name__ == "__main__":
    main() 