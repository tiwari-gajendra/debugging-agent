# Debugging Agents

An AI-powered debugging assistant that uses multiple specialized agents to analyze and debug software issues.

## Features

- 🔍 Automated debugging with specialized AI agents
- 📊 Log analysis and pattern detection
- 📝 BIM document generation
- 🎯 JIRA integration
- 🚀 Streamlit web interface
- 🔄 Real-time progress tracking

## Requirements

- Python 3.8+
- Pandoc (for document format conversion)
- Python packages listed in requirements.txt

## Installation

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

4. Install pandoc:

macOS:
```bash
brew install pandoc
```

Linux:
```bash
sudo apt-get install pandoc
```

Windows:
Download the installer from https://pandoc.org/installing.html

5. Copy the example environment file and configure:
```bash
cp .env.example .env
# Edit .env with your settings
```

## Usage

### Web Interface

Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

### CLI

Use the command-line interface:
```bash
python debug_agent_cli.py --issue-id PROJ-123
```

## Project Structure

```
debugging-agents/
├── config/              # Configuration files
├── data/               # Data directories
│   ├── contexts/       # Debug context storage
│   ├── logs/          # Application logs
│   ├── models/        # ML model storage
│   ├── plots/         # Generated plots
│   ├── reports/       # Debug reports
│   ├── templates/     # Document templates
│   └── vector_store/  # Vector embeddings
├── docs/              # Documentation
├── scripts/           # Utility scripts
├── src/              # Source code
│   ├── forecasting/  # Forecasting models
│   ├── integrations/ # External integrations
│   ├── manager/      # Agent management
│   ├── realtime/     # Real-time analysis
│   ├── ui/           # UI components
│   └── utils/        # Utilities
└── tests/            # Test files
```

## Configuration

The application can be configured through environment variables in the `.env` file:

- `LLM_PROVIDER`: LLM provider (openai, ollama, bedrock, anthropic)
- `OLLAMA_MODEL`: Model name for Ollama (default: deepseek-r1:8b)
- `OLLAMA_BASE_URL`: Ollama API URL (default: http://localhost:11434)
- `JIRA_URL`: JIRA instance URL
- `JIRA_USERNAME`: JIRA username
- `JIRA_API_TOKEN`: JIRA API token

## Development

1. Install development dependencies:
```bash
pip install -r requirements.txt
```

2. Run tests:
```bash
pytest
```

3. Run linting:
```bash
pylint src tests
```

4. Run type checking:
```bash
mypy src
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 