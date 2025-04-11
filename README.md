# AI Debug Agents

An intelligent multi-agent system for forecasting service anomalies and debugging issues in real-time.

## Architecture

### Forecasting Agents (Offline/Periodic)
- **Service Log Forecaster Agent**: Predicts service anomalies using time-series models
- **Alert Forecaster Agent**: Predicts alert bursts or future incidents using NLP

### Real-time Debugging Pipeline
1. **Issue Context Builder**: Gathers relevant logs, metrics, traces, and past issues
2. **Debug Plan Creator**: Creates a plan for debugging based on the context
3. **Executor Agent**: Follows the plan by downloading logs, querying DBs, etc.
4. **Analyzer Agent**: Analyzes the data to determine root cause
5. **Document Generator**: Creates documentation with findings

### Agent Coordination
Uses CrewAI for agent management, delegation, and coordination

## Setup

1. Create a virtual environment:
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Create a .env file with your API keys:
```
OPENAI_API_KEY=your_openai_api_key
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
```

4. Install the debugging agents CLI:
```
pip install -e .
```

## Using the CLI

The system includes a command-line interface for easy interaction:

```bash
# Show system information
debug-agent info

# Debug a specific issue
debug-agent debug YOUR-ISSUE-123 --open-doc

# Run forecasting pipeline
debug-agent forecast --verbose
```

For detailed CLI usage instructions, see [CLI Guide](docs/cli_guide.md).

## Running with Local Models (Ollama)

This system supports Ollama for local LLM inference. To use Ollama:

1. Make sure Ollama is running on your machine (default: http://localhost:11434)
2. Set the following in your `.env` file:
```
LLM_PROVIDER=ollama
OLLAMA_MODEL=deepseek-r1    # or any model you have pulled in Ollama
OLLAMA_BASE_URL=http://localhost:11434  # default Ollama URL
```
3. Run the CLI normally:
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd) && python debug_agent_cli.py info
export PYTHONPATH=$PYTHONPATH:$(pwd) && python debug_agent_cli.py debug YOUR-ISSUE-123
```

When using Ollama provider, the system automatically bypasses the standard agent framework and uses direct API calls to Ollama for better compatibility.

## Model Switching

The system now supports easy switching between different LLM models:

```bash
# Use a specific model by name
debug-agent debug YOUR-ISSUE-123 --model claude-3-sonnet

# Or use a provider with its default model
debug-agent debug YOUR-ISSUE-123 --llm-provider bedrock
```

Available models include:

- **OpenAI Models**: `gpt-4`, `gpt-4-turbo`, `gpt-3.5-turbo`
- **Bedrock Models**: `claude-3-sonnet`, `claude-3-haiku`, `llama3-70b`
- **Ollama Models**: `llama3`, `mistral` (local models)

You can add custom models programmatically:

```python
from src.utils.llm_factory import LLMFactory

# Register a custom model
LLMFactory.register_custom_model(
    name="my-custom-model",
    provider="bedrock",
    model_id="anthropic.claude-3-opus-20240229-v1:0",
    options={"max_tokens": 4000}
)

# Then use it from the CLI
# debug-agent debug ISSUE-123 --model my-custom-model
```

For more details, see [Usage Guide](docs/usage_guide.md).

## Project Structure

```
debugging-agents/
├── requirements.txt
├── README.md
├── .env
├── cli.py                  # Simple CLI entry point
├── debugging_agents/       # Python package for CLI
├── setup.py                # Package installation config
├── src/
│   ├── main.py
│   ├── forecasting/
│   │   ├── __init__.py
│   │   ├── service_log_forecaster.py
│   │   └── alert_forecaster.py
│   ├── realtime/
│   │   ├── __init__.py
│   │   ├── context_builder.py
│   │   ├── debug_plan_creator.py
│   │   ├── executor.py
│   │   ├── analyzer.py
│   │   └── document_generator.py
│   ├── coordination/
│   │   ├── __init__.py
│   │   └── crew_manager.py
│   └── utils/
│       ├── __init__.py
│       ├── data_fetchers.py
│       ├── vector_store.py
│       └── logging.py
├── tests/
│   ├── __init__.py
│   ├── test_forecasting.py
│   └── test_realtime.py
└── data/
    ├── models/
    └── vector_store/
``` 