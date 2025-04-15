# Project Structure

This document outlines the organization of the Debugging Agents codebase.

```
debugging-agents/
├── config/                    # Configuration files
│   ├── logging.yaml          # Logging configuration
│   └── loki/                 # Loki-specific configurations
│
├── data/                     # Data storage
│   ├── logs/                 # Log files
│   │   ├── debug_agent/      # Debug agent logs
│   │   └── service_logs/     # Service logs
│   ├── plots/                # Generated plots and visualizations
│   ├── reports/              # Generated debugging reports
│   └── templates/            # Report templates
│
├── docs/                     # Documentation
│   ├── api_reference.md      # API documentation
│   ├── architecture.md       # System architecture
│   ├── bedrock_setup.md      # AWS Bedrock setup guide
│   ├── cli_guide.md          # CLI usage guide
│   ├── project_structure.md  # This file
│   └── usage_guide.md        # General usage guide
│
├── src/                      # Source code
│   ├── integrations/         # External service integrations
│   │   ├── __init__.py
│   │   ├── loki_client.py   # Loki log client
│   │   └── slack_handler.py # Slack integration
│   │
│   ├── manager/             # Core management components
│   │   ├── __init__.py
│   │   └── crew_manager.py  # Crew management
│   │
│   ├── realtime/           # Real-time debugging components
│   │   ├── __init__.py
│   │   ├── analyzer.py     # Analysis engine
│   │   ├── context_builder.py  # Context gathering
│   │   ├── debug_plan_creator.py  # Debug plan generation
│   │   ├── document_generator.py  # Report generation
│   │   └── executor.py     # Plan execution
│   │
│   └── utils/              # Utility functions
│       ├── __init__.py
│       └── template_manager.py  # Template management
│
├── tests/                  # Test files
│   └── test_logs.py       # Log testing
│
├── .env                   # Environment variables
├── .env.example          # Example environment variables
├── .gitignore           # Git ignore rules
├── debug_agent_cli.py   # CLI interface
├── main.py             # Main application entry
├── README.md          # Project overview
└── requirements.txt   # Python dependencies
```

## Directory Descriptions

### config/
Contains all configuration files for the application, including logging settings and external service configurations.

### data/
Stores all generated and collected data:
- `logs/`: Contains debug agent logs and service logs
- `plots/`: Stores generated visualizations and plots
- `reports/`: Contains generated debugging reports
- `templates/`: Holds report templates in various formats

### docs/
Comprehensive documentation including:
- API reference
- Architecture overview
- Setup guides
- Usage instructions
- Project structure

### src/
Core source code organized into modules:
- `integrations/`: External service integrations (Loki, Slack)
- `manager/`: Core management components
- `realtime/`: Real-time debugging components
- `utils/`: Utility functions and helpers

### tests/
Contains test files for the application.

## Key Files

- `debug_agent_cli.py`: Command-line interface for the debugging system
- `main.py`: Main application entry point
- `requirements.txt`: Python package dependencies
- `.env`: Environment variables (not committed to version control)
- `.env.example`: Example environment variables 