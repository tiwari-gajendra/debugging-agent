# Debugging Agents API Reference

This document provides detailed information about the core classes and interfaces in the Debugging Agents system.

## Core Components

### `ContextBuilder`

The ContextBuilder collects and organizes all contextual information relevant to a debugging task.

```python
class ContextBuilder:
    def __init__(self, data_dir: str = None):
        # Initialize with optional data directory
        
    def build_context(self, issue_id: str) -> dict:
        # Build comprehensive context for an issue
        
    def fetch_system_metrics(self, timeframe: int = 24) -> dict:
        # Fetch relevant system metrics for the specified timeframe (hours)
        
    def fetch_logs(self, service: str = None, timeframe: int = 24) -> list:
        # Fetch logs for the specified service within the timeframe
        
    def fetch_traces(self, service: str = None, timeframe: int = 24) -> list:
        # Fetch distributed traces for the service
        
    def fetch_alerts(self, timeframe: int = 24) -> list:
        # Fetch recent alerts from monitoring systems
```

### `DebugPlanCreator`

The DebugPlanCreator uses LLM capabilities to generate structured debugging plans.

```python
class DebugPlanCreator:
    def __init__(self, llm_provider: str, llm_model: str):
        # Initialize with LLM provider and model
        
    def create_plan(self, context: dict) -> dict:
        # Create a debugging plan based on the context
        
    def validate_plan(self, plan: dict) -> bool:
        # Validate the structure and actions in a plan
        
    def explain_plan(self, plan: dict) -> str:
        # Generate a human-readable explanation of the plan
```

### `Executor`

The Executor carries out the actions defined in the debugging plan.

```python
class Executor:
    def __init__(self, timeout: int = 300, max_retries: int = 3):
        # Initialize with execution parameters
        
    def execute_plan(self, plan: dict) -> dict:
        # Execute all steps in the debugging plan
        
    def execute_step(self, step: dict) -> dict:
        # Execute a single step from the plan
        
    def execute_command(self, command: str, env: dict = None) -> dict:
        # Execute a shell command
        
    def execute_api_call(self, url: str, method: str = "GET", 
                         headers: dict = None, data: dict = None) -> dict:
        # Execute an API call
        
    def execute_diagnostic(self, diagnostic_type: str, parameters: dict) -> dict:
        # Execute a predefined diagnostic action
```

### `Analyzer`

The Analyzer processes execution results to determine root causes and recommendations.

```python
class Analyzer:
    def __init__(self, llm_provider: str, llm_model: str):
        # Initialize with LLM provider and model
        
    def analyze_results(self, context: dict, plan: dict, 
                       execution_results: dict) -> dict:
        # Analyze debugging results and identify root causes
        
    def identify_patterns(self, data: dict) -> list:
        # Identify patterns in the data
        
    def evaluate_service_health(self, metrics: dict) -> dict:
        # Evaluate the health of services based on metrics
        
    def generate_recommendations(self, analysis: dict) -> list:
        # Generate actionable recommendations
```

### `DocumentGenerator`

The DocumentGenerator creates detailed reports from debugging results.

```python
class DocumentGenerator:
    def __init__(self, output_format: str = "html"):
        # Initialize with output format specification
        
    def generate_document(self, context: dict, plan: dict, 
                         execution_results: dict, analysis: dict) -> str:
        # Generate a comprehensive debugging document
        
    def _generate_html_report(self, data: dict) -> str:
        # Generate HTML format report
        
    def _generate_markdown_report(self, data: dict) -> str:
        # Generate Markdown format report
        
    def _generate_pdf_report(self, data: dict) -> str:
        # Generate PDF format report
```

### `ForecastingEngine`

The ForecastingEngine predicts potential issues based on historical patterns.

```python
class ForecastingEngine:
    def __init__(self, llm_provider: str, llm_model: str, 
                data_dir: str = None):
        # Initialize with LLM and data configurations
        
    def run_forecast(self, hours: int = 24, 
                    threshold: float = 0.7) -> dict:
        # Run a forecasting pipeline
        
    def identify_anomalies(self, metrics: dict) -> list:
        # Identify anomalies in metrics
        
    def predict_service_issues(self, 
                              anomalies: list, 
                              historical_data: dict) -> list:
        # Predict potential service issues
```

### `CrewManager`

The CrewManager coordinates specialized AI agents for complex debugging tasks.

```python
class CrewManager:
    def __init__(self, llm_provider: str, llm_model: str):
        # Initialize with LLM provider and model
        
    def create_crew(self, task_type: str) -> list:
        # Create a specialized crew for a task type
        
    def assign_task(self, crew: list, task: dict) -> None:
        # Assign a task to the crew
        
    def collect_results(self, crew: list) -> dict:
        # Collect results from all crew members
        
    def merge_analyses(self, analyses: list) -> dict:
        # Merge analyses from multiple agents
```

## Utility Classes

### `LLMProvider`

Manages connections to different LLM providers.

```python
class LLMProvider:
    def __init__(self, provider: str, model: str):
        # Initialize with provider and model
        
    def complete(self, prompt: str, 
                max_tokens: int = 1000, 
                temperature: float = 0.2) -> str:
        # Get a completion from the LLM
        
    def chat(self, messages: list, 
            max_tokens: int = 1000, 
            temperature: float = 0.2) -> str:
        # Get a chat completion from the LLM
```

### `DataManager`

Handles data storage and retrieval.

```python
class DataManager:
    def __init__(self, data_dir: str):
        # Initialize with data directory
        
    def save_context(self, issue_id: str, context: dict) -> str:
        # Save context data to file
        
    def save_plan(self, issue_id: str, plan: dict) -> str:
        # Save plan data to file
        
    def save_results(self, issue_id: str, results: dict) -> str:
        # Save execution results to file
        
    def save_analysis(self, issue_id: str, analysis: dict) -> str:
        # Save analysis data to file
        
    def save_document(self, issue_id: str, document: str, 
                     format: str = "html") -> str:
        # Save generated document to file
        
    def load_data(self, file_path: str) -> dict:
        # Load data from file
```

### `Logger`

Configures and manages logging.

```python
class Logger:
    def __init__(self, log_level: str = "INFO", 
                log_file: str = None):
        # Initialize logging configuration
        
    def get_logger(self, name: str) -> logging.Logger:
        # Get a configured logger instance
```

## Interfaces

### `DataSourceInterface`

Interface for data source plugins.

```python
class DataSourceInterface(ABC):
    @abstractmethod
    def connect(self) -> bool:
        # Connect to the data source
        pass
        
    @abstractmethod
    def fetch_data(self, query: str, parameters: dict = None) -> dict:
        # Fetch data from the source
        pass
        
    @abstractmethod
    def validate_connection(self) -> bool:
        # Validate connection is working
        pass
```

### `DiagnosticInterface`

Interface for diagnostic action plugins.

```python
class DiagnosticInterface(ABC):
    @abstractmethod
    def run_diagnostic(self, parameters: dict) -> dict:
        # Run a diagnostic operation
        pass
        
    @abstractmethod
    def get_capabilities(self) -> list:
        # Return capabilities of this diagnostic
        pass
```

## Usage Examples

### Creating a Context Builder

```python
from debugging_agents.context_builder import ContextBuilder

# Initialize the context builder
context_builder = ContextBuilder(data_dir="/path/to/data")

# Build context for an issue
context = context_builder.build_context("SERVICE-DOWN-123")
```

### Generating a Debug Plan

```python
from debugging_agents.debug_plan_creator import DebugPlanCreator

# Initialize the plan creator
plan_creator = DebugPlanCreator(llm_provider="openai", 
                                llm_model="gpt-4-turbo")

# Create a plan based on context
plan = plan_creator.create_plan(context)
```

### Executing a Debug Plan

```python
from debugging_agents.executor import Executor

# Initialize the executor
executor = Executor(timeout=600)

# Execute the plan
results = executor.execute_plan(plan)
```

### Analyzing Results

```python
from debugging_agents.analyzer import Analyzer

# Initialize the analyzer
analyzer = Analyzer(llm_provider="openai", llm_model="gpt-4-turbo")

# Analyze the results
analysis = analyzer.analyze_results(context, plan, results)
```

### Generating a Report

```python
from debugging_agents.document_generator import DocumentGenerator

# Initialize the document generator
doc_generator = DocumentGenerator(output_format="html")

# Generate a debugging report
report_path = doc_generator.generate_document(context, plan, results, analysis)
```

### Running a Forecast

```python
from debugging_agents.forecasting_engine import ForecastingEngine

# Initialize the forecasting engine
forecasting = ForecastingEngine(llm_provider="openai", 
                               llm_model="gpt-4-turbo")

# Run a forecast
forecast = forecasting.run_forecast(hours=48, threshold=0.8)
```

## Extension Points

The Debugging Agents system can be extended in several ways:

1. **Custom Data Sources**: Implement the `DataSourceInterface` to add new data sources.
2. **Custom Diagnostics**: Implement the `DiagnosticInterface` to add new diagnostic capabilities.
3. **Report Templates**: Add new templates to the `DocumentGenerator` for custom report formats.
4. **LLM Providers**: Extend the `LLMProvider` class to support additional LLM services.

## Configuration Reference

The system can be configured through environment variables or a configuration file:

### Environment Variables

```
LLM_PROVIDER=openai|bedrock|ollama|anthropic|snowflake|cortex
LLM_MODEL=<model-name>
DATA_DIR=<path-to-data>
LOG_LEVEL=DEBUG|INFO|WARNING|ERROR
```

### Configuration File Format

```yaml
llm:
  provider: openai|bedrock|ollama|anthropic|snowflake|cortex
  model: <model-name>
  temperature: 0.2
  max_tokens: 4096

data:
  dir: <path-to-data-directory>
  logs_source: elasticsearch|splunk|cloudwatch|file
  metrics_source: prometheus|datadog|cloudwatch|grafana
  traces_source: jaeger|zipkin|xray|otel

executor:
  timeout: 300
  max_retries: 3
  parallel_execution: false

analyzer:
  confidence_threshold: 0.7
  max_root_causes: 3

document:
  format: doc|pdf
  include_raw_data: false
  template: default|minimal|detailed
```