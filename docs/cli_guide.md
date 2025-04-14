# Debugging Agents CLI Guide

This guide explains how to use the Debugging Agents CLI tool effectively.

## Command Overview

The Debugging Agents CLI provides the following commands:

```bash
python debug_agent_cli.py info                    # Display system information
python debug_agent_cli.py debug ISSUE-ID          # Debug an issue
```

## Installation

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- virtualenv (recommended)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/debugging-agents.git
   cd debugging-agents
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Command Reference

### Info Command

Display system information and configuration:

```bash
python debug_agent_cli.py info
```

Output includes:
- Python version
- LLM provider configuration
- Available agents
- Logging status

### Debug Command

Debug an issue with a specific ID:

```bash
python debug_agent_cli.py debug ISSUE-ID [options]
```

Options:
- `--llm-provider {openai,ollama,bedrock}` - LLM provider to use
- `--open-doc` - Open the generated report when complete
- `--verbose` - Show detailed output

Examples:
```bash
# Debug with default settings
python debug_agent_cli.py debug TEST-123

# Debug with specific LLM provider
python debug_agent_cli.py debug TEST-123 --llm-provider ollama

# Debug with report opening
python debug_agent_cli.py debug TEST-123 --open-doc

# Debug with verbose output
python debug_agent_cli.py debug TEST-123 --verbose
```

## Environment Variables

The CLI uses the following environment variables:

### LLM Configuration
```bash
# General
export LLM_PROVIDER=ollama  # or openai, bedrock
export TEMPERATURE=0.2

# Ollama
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_MODEL=deepseek-r1:8b

# OpenAI
export OPENAI_API_KEY=your_key_here
export OPENAI_MODEL=gpt-4-turbo

# AWS Bedrock
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-east-1
export BEDROCK_MODEL=anthropic.claude-3-sonnet-20240229-v1:0
```

### Logging Configuration
```bash
# Optional: Override default log locations
export DEBUG_AGENT_LOG_DIR=data/logs/debug_agent
export SERVICE_LOGS_DIR=data/logs/service_logs
```

## Directory Structure

The CLI expects the following directory structure:

```
debugging-agents/
├── config/
│   ├── logging.yaml           # Logging configuration
│   └── agents.yaml            # Agent configuration
├── data/
│   ├── logs/
│   │   ├── debug_agent/      # Debug agent logs
│   │   └── service_logs/     # Service logs for analysis
│   ├── reports/              # Generated reports
│   └── templates/            # Report templates
├── src/                      # Source code
├── debug_agent_cli.py        # Main CLI script
├── requirements.txt          # Python dependencies
└── README.md                # Project documentation
```

## Logging

The CLI uses a hierarchical logging system:

1. Console output shows INFO and above
2. File logging includes DEBUG and above
3. Logs are written to `data/logs/debug_agent/debug_agent.log`
4. Log rotation is handled automatically

To see detailed logs:
```bash
tail -f data/logs/debug_agent/debug_agent.log
```

## Error Handling

Common error scenarios and solutions:

### ModuleNotFoundError
```
ModuleNotFoundError: No module named 'src'
```
- Ensure you're in the correct directory
- Activate the virtual environment
- Install dependencies with pip

### LLM Provider Errors
```
Error: Could not connect to LLM provider
```
- Check if Ollama is running (for Ollama)
- Verify API keys (for OpenAI/Bedrock)
- Check network connectivity

### Permission Errors
```
PermissionError: [Errno 13] Permission denied
```
- Check file/directory permissions
- Ensure write access to log directories
- Run with appropriate user permissions

## Best Practices

1. **Virtual Environment**
   - Always use a virtual environment
   - Keep dependencies up to date
   - Use the same Python version as the project

2. **Directory Structure**
   - Maintain the expected directory structure
   - Don't modify config files directly
   - Use environment variables for configuration

3. **Logging**
   - Check logs for debugging
   - Use --verbose for detailed output
   - Rotate logs regularly

4. **Error Handling**
   - Read error messages carefully
   - Check logs for details
   - Follow the troubleshooting guide

## Contributing

To contribute to the CLI:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

See CONTRIBUTING.md for detailed guidelines.

## Support

If you need help:

1. Check the documentation
2. Look for similar issues on GitHub
3. Join our Discord community
4. Open a new issue if needed

## License

This project is licensed under the MIT License. See LICENSE for details. 