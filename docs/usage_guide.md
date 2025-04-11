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

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables according to your chosen LLM provider:

   **For OpenAI:**
   ```bash
   export LLM_PROVIDER=openai
   export OPENAI_API_KEY=your_api_key_here
   export LLM_MODEL=gpt-4-turbo
   ```

   **For AWS Bedrock:**
   ```bash
   export LLM_PROVIDER=bedrock
   export AWS_ACCESS_KEY_ID=your_access_key
   export AWS_SECRET_ACCESS_KEY=your_secret_key
   export AWS_REGION=your_aws_region
   export LLM_MODEL=anthropic.claude-3-sonnet-20240229-v1:0
   ```

   **For Ollama:**
   ```bash
   export LLM_PROVIDER=ollama
   export OLLAMA_API_BASE=http://localhost:11434
   export LLM_MODEL=llama3
   ```

4. Set data directory (optional):
   ```bash
   export DATA_DIR=/path/to/data/directory
   ```

## Basic Usage

The Debugging Agents system is operated through a command-line interface (CLI) with several subcommands.

### Displaying System Information

To verify your setup and see the current configuration:

```bash
python debug_agent_cli.py info
```

This will display:
- System version
- Python version
- LLM provider and model
- Data directory location

### Debugging an Issue

To debug an issue with a specific ID:

```bash
python debug_agent_cli.py debug TEST-FIXED-123 --open-doc
```

Optional flags:
- `--open-report`: Automatically open the generated report in a web browser
- `--interactive`: Run in interactive mode with step-by-step confirmation
- `--verbose`: Enable verbose logging

Example:
```bash
python debug_agent_cli.py debug TEST-FIXED-123 --open-doc
```

### Forecasting Potential Issues

To predict potential issues before they occur:

```bash
python debug_agent_cli.py forecast
```

Optional flags:
- `--service`: Focus forecasting on a specific service
- `--lookback`: Historical data window to consider (in hours)
- `--confidence`: Minimum confidence threshold for predictions

Example:
```bash
python debug_agent_cli.py forecast --service payment-processor --lookback 24
```

## Configuration File

For more advanced configuration, you can create a `config.yaml` file in the root directory:

```yaml
llm:
  provider: openai
  model: gpt-4-turbo
  temperature: 0.2
  max_tokens: 4096

data:
  logs_source: elasticsearch
  metrics_source: prometheus
  traces_source: jaeger
  config_source: consul

executor:
  timeout: 300
  max_retries: 3
  parallel_execution: false

analyzer:
  confidence_threshold: 0.7
  max_root_causes: 3

document:
  format: html
  include_raw_data: false
  template: default
```

## Data Integration

### Log Sources

The system supports various log sources that can be configured in the `config.yaml` file:

```yaml
data:
  logs_source: elasticsearch
  logs_config:
    url: http://elasticsearch:9200
    index_pattern: logs-*
    username: elastic
    password: ${ES_PASSWORD}
```

Supported log sources:
- Elasticsearch
- Splunk
- CloudWatch
- Local log files

### Metrics Sources

Configure metrics sources:

```yaml
data:
  metrics_source: prometheus
  metrics_config:
    url: http://prometheus:9090
    query_timeout: 30
```

Supported metrics sources:
- Prometheus
- Datadog
- CloudWatch Metrics
- Grafana

### Trace Sources

Configure distributed tracing:

```yaml
data:
  traces_source: jaeger
  traces_config:
    url: http://jaeger:16686
    service_name: ${SERVICE}
```

Supported trace sources:
- Jaeger
- Zipkin
- AWS X-Ray
- OpenTelemetry

## Example Workflows

### Basic Troubleshooting

1. Receive alert for a service issue with ID "SVC-DOWN-123"
2. Run the debug command:
   ```bash
   python debug_agent_cli.py debug TEST-FIXED-123 --open-report
   ```
3. Review the generated HTML report for root cause and recommendations

### Proactive Monitoring

1. Schedule a daily forecasting job:
   ```bash
   # In crontab
   0 6 * * * cd /path/to/debugging-agents && python debug_agent_cli.py forecast --service all
   ```
2. Analyze predicted issues in the generated reports
3. Take preventive action based on recommendations

## Troubleshooting the Debugging Agents

### Common Issues

#### Missing Environment Variables
If you see errors about missing environment variables, ensure you've set all required variables for your LLM provider.

#### LLM Connection Errors
For OpenAI or Bedrock connection issues, verify your API keys and network connectivity.

#### Data Source Connection Issues
Check connectivity to your configured data sources and ensure credentials are correct.

#### Report Generation Failures
Verify write permissions for the data directory and ensure enough disk space is available.

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