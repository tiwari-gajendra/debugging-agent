# Debugging Agents Usage Guide

This guide explains how to use the Debugging Agents system for automated troubleshooting and debugging.

## System Setup

### Prerequisites
- Python 3.10 or higher
- Ollama (for local LLM) or access to other LLM providers
- JIRA access (for ticket integration)
- Loki (optional, for log collection)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-org/debugging-agents.git
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

4. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

### Running the System

The system can be run in two modes:

#### 1. Integrated Mode (UI + Backend)
```bash
streamlit run streamlit_app.py
```
This starts both the UI and backend services together. The UI will be available at `http://localhost:8501`.

#### 2. Separate Mode
If you need to run the backend separately (e.g., for development or testing):

a) Start the backend:
```bash
python src/main.py
```

b) Start the UI in a separate terminal:
```bash
streamlit run streamlit_app.py
```

## Context Preservation

The system automatically preserves debugging context in two ways:

### 1. File-based Storage
- Contexts are stored in `data/contexts/` directory
- Each ticket's current context: `{ticket_id}_context.json`
- Context history: `{ticket_id}_history.json`
- Example structure:
```json
{
    "current_context": {
        "ticket_id": "PROJ-123",
        "timestamp": "2024-04-14T12:00:00",
        "logs": [...],
        "metrics": {...},
        "analysis": {...}
    },
    "history": [
        {
            "timestamp": "2024-04-14T11:00:00",
            "context": {...}
        }
    ]
}
```

### 2. In-memory Cache
- Contexts are cached in memory during the session
- Cache is cleared when the application restarts
- Used for quick access to recent contexts

## Using the System

### 1. JIRA Ticket Debugging

1. Open the Streamlit UI at `http://localhost:8501`
2. Enter a JIRA ticket ID (e.g., "PROJ-123")
3. Choose an action:
   - "Generate BIM Document": Creates a structured document
   - "Run Debug Analysis": Performs automated debugging

### 2. Context Management

The system automatically:
- Preserves context between debugging sessions
- Maintains history of previous contexts
- Updates context with new findings
- Provides context-aware debugging

### 3. Document Generation

After a successful debug analysis, you can download the BIM document in your preferred format:

1. Select your desired format from the Document Format dropdown:
   - **doc**: Microsoft Word document format
   - **pdf**: Portable Document Format

2. Click the "Download" button to save the report to your computer.

3. The document includes:
   - A summary of the debugging process
   - Issue analysis
   - Root cause determination
   - Recommended solutions

> **Note:** Format conversion is handled automatically using built-in Python libraries. No external dependencies are required for document conversion.

## Troubleshooting

### Common Issues

1. **LLM Connection Issues**
   - Check Ollama is running: `curl http://localhost:11434/api/tags`
   - Verify API keys in `.env`

2. **JIRA Integration**
   - Ensure JIRA credentials are correct
   - Check network connectivity
   - Verify ticket access permissions

3. **Context Preservation**
   - Check `data/contexts/` directory exists
   - Verify file permissions
   - Monitor disk space

### Logs and Monitoring

- Application logs: `logs/debug_agent.log`
- Context files: `data/contexts/`
- Reports: `data/reports/`

## Advanced Usage

### Custom Context Management

You can customize context preservation by:

1. Modifying `ContextBuilder` class
2. Implementing custom storage backends
3. Adjusting cache settings

### Integration with Other Systems

The system can be extended to:
- Add new data sources
- Implement custom analyzers
- Create specialized reports

## Best Practices

1. **Context Management**
   - Regularly backup context files
   - Monitor context storage usage
   - Clean up old contexts periodically

2. **Performance**
   - Use appropriate time windows for data collection
   - Monitor memory usage with large contexts
   - Optimize query patterns

3. **Security**
   - Secure sensitive data in contexts
   - Implement proper access controls
   - Regularly rotate API keys

## Support

For issues or questions:
1. Check the documentation
2. Review logs in `logs/`
3. Contact the development team 