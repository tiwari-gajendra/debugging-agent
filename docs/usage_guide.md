# Debugging Agents Usage Guide

This guide provides instructions on how to set up and use the Debugging Agents system for automated troubleshooting of production issues.

## Installation

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- Access to one of the supported LLM providers:
  - OpenAI API
  - AWS Bedrock
  - Ollama (for local LLM usage)

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

4. Configure environment variables according to your chosen LLM provider:

   **For OpenAI:**
   ```bash
   export LLM_PROVIDER=openai
   export OPENAI_API_KEY=your_api_key_here
   export OPENAI_MODEL=gpt-4-turbo  # or another model
   export TEMPERATURE=0.2
   ```

   **For AWS Bedrock:**
   ```bash
   export LLM_PROVIDER=bedrock
   export AWS_ACCESS_KEY_ID=your_access_key
   export AWS_SECRET_ACCESS_KEY=your_secret_key
   export AWS_DEFAULT_REGION=your_aws_region
   export BEDROCK_MODEL=anthropic.claude-3-sonnet-20240229-v1:0
   export TEMPERATURE=0.2
   ```

   **For Ollama:**
   ```bash
   export LLM_PROVIDER=ollama
   export OLLAMA_BASE_URL=http://localhost:11434
   export OLLAMA_MODEL=deepseek-r1:8b  # or another model
   export TEMPERATURE=0.2
   ```

5. Set up logging and data directories:
   ```bash
   mkdir -p data/logs/debug_agent
   mkdir -p data/logs/service_logs
   mkdir -p data/reports
   ```

## Basic Usage

### Verifying Installation

First, verify that your installation is working correctly:

```bash
python debug_agent_cli.py info
```

This will show:
- System version and configuration
- LLM provider status
- Logging configuration
- Available agents

### Running the Debug Command

To debug an issue:

```bash
python debug_agent_cli.py debug ISSUE-ID [options]
```

Options:
- `--llm-provider`: Override the default LLM provider (openai, bedrock, ollama)
- `--open-doc`: Open the generated report when complete
- `--verbose`: Show detailed output

Example:
```bash
python debug_agent_cli.py debug TEST-123 --llm-provider ollama --open-doc
```

### Understanding the Output

The debug command will:
1. Initialize the CrewAI agents
2. Gather context about the issue
3. Create a debugging plan
4. Execute the plan
5. Analyze results
6. Generate a report

The report will be saved in `data/reports/` and includes:
- Executive summary
- Root cause analysis
- Debugging steps taken
- Recommendations

## Configuration

### LLM Configuration

Create a `.env` file in the project root:

```env
# LLM Provider Settings
LLM_PROVIDER=ollama  # or openai, bedrock
TEMPERATURE=0.2

# Ollama Settings
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=deepseek-r1:8b

# OpenAI Settings (if using OpenAI)
#OPENAI_API_KEY=your_key_here
#OPENAI_MODEL=gpt-4-turbo

# AWS Bedrock Settings (if using Bedrock)
#AWS_ACCESS_KEY_ID=your_key
#AWS_SECRET_ACCESS_KEY=your_secret
#AWS_DEFAULT_REGION=us-east-1
#BEDROCK_MODEL=anthropic.claude-3-sonnet-20240229-v1:0
```

### Logging Configuration

The system uses a YAML-based logging configuration in `config/logging.yaml`:

```yaml
version: 1
disable_existing_loggers: false

formatters:
  standard:
    format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    datefmt: "%Y-%m-%d %H:%M:%S"

handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: standard
    stream: ext://sys.stdout

  file:
    class: logging.FileHandler
    level: DEBUG
    formatter: standard
    filename: data/logs/debug_agent/debug_agent.log
    mode: a
    encoding: utf8

loggers:
  '':  # root logger
    handlers: [console, file]
    level: INFO
    propagate: true

  httpx:
    level: WARNING
    handlers: [console, file]
    propagate: false

  crewai:
    level: INFO
    handlers: [console, file]
    propagate: false
```

### Agent Configuration

Each agent can be configured in `config/agents.yaml`:

```yaml
context_builder:
  enabled: true
  log_sources:
    - loki
    - files
  log_path: data/logs/service_logs

debug_plan_creator:
  enabled: true
  max_steps: 10
  confidence_threshold: 0.7

executor:
  enabled: true
  timeout: 300
  max_retries: 3
  parallel_execution: false

analyzer:
  enabled: true
  min_confidence: 0.7
  max_root_causes: 3

document_generator:
  enabled: true
  format: html
  template: rca_template.json
  output_dir: data/reports
```

## Troubleshooting

### Common Issues

1. **LLM Provider Not Available**
   ```
   Error: Could not connect to LLM provider
   ```
   - Check if the provider is running (especially for Ollama)
   - Verify API keys and environment variables
   - Check network connectivity

2. **Missing Log Sources**
   ```
   Warning: No log sources available
   ```
   - Ensure log directories exist
   - Check Loki connection if using Loki
   - Verify log file permissions

3. **Report Generation Fails**
   ```
   Error: Could not generate report
   ```
   - Check write permissions in reports directory
   - Verify template file exists
   - Check disk space

### Getting Help

1. Check the logs in `data/logs/debug_agent/debug_agent.log`
2. Run commands with `--verbose` flag
3. Check the GitHub issues page
4. Join our Discord community

## Best Practices

1. **LLM Selection**
   - Use Ollama for development and testing
   - Use OpenAI/Bedrock for production
   - Consider latency and cost requirements

2. **Log Management**
   - Regularly rotate log files
   - Monitor disk usage
   - Set appropriate log levels

3. **Security**
   - Never commit API keys
   - Use environment variables
   - Regularly rotate credentials

4. **Performance**
   - Use appropriate timeouts
   - Monitor resource usage
   - Cache results when possible

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

See CONTRIBUTING.md for detailed guidelines.

## Advanced Usage

### Custom Debug Plans

You can provide a custom debugging plan in JSON format:

```bash
python debug_agent_cli.py debug --issue-id CUSTOM-DEBUG --plan-file my_plan.json
```

Example plan file:
```json
{
  "steps": [
    {
      "name": "Check CPU Usage",
      "action": "execute_command",
      "parameters": {
        "command": "top -b -n 1"
      }
    },
    {
      "name": "Check Memory Usage",
      "action": "execute_command",
      "parameters": {
        "command": "free -m"
      }
    }
  ]
}
```

### Using Different LLM Models

Switch between different LLM models for cost or performance reasons:

```bash
# Use a smaller model for faster results
export LLM_MODEL=gpt-3.5-turbo
python debug_agent_cli.py debug --issue-id SMALL-ISSUE

# Use a more powerful model for complex analysis
export LLM_MODEL=gpt-4-turbo
python debug_agent_cli.py debug --issue-id COMPLEX-ISSUE
```

## Advanced LLM Configuration

### Creating an LLM Instance

You can create an LLM instance directly using the factory:

```python
from src.utils.llm_factory import LLMFactory

# Create a specific LLM
llm = LLMFactory.create_llm("openai")  # Use default model from env
llm = LLMFactory.create_llm("ollama", model="llama3")  # Specify model
```

### Switching Providers

To switch between different providers, update your .env file:

```
# Set in .env file
LLM_PROVIDER=openai
OPENAI_MODEL=gpt-4

# Or for Ollama
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3
```

## References

- [Architecture Document](architecture.md) - System architecture overview
- [API Reference](api_reference.md) - Detailed API documentation
- [GitHub Repository](https://github.com/yourusername/debugging-agents) - Source code and examples 