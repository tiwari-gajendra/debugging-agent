# Debugging Agents System

A system for automated root cause analysis and debugging of issues in distributed systems.

## Features

- Real-time issue detection and analysis
- Automated log collection and analysis
- Metrics monitoring and correlation
- Distributed tracing analysis
- Root Cause Analysis (RCA) report generation
- Slack integration for alerts and notifications

## Prerequisites

- Python 3.8+
- Docker and Docker Compose (for running Loki)
- Slack workspace (for notifications)

## Setup

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

4. Copy the example environment file and update it:
```bash
cp .env.example .env
```

5. Update the `.env` file with your configuration:
- Set `LOKI_URL` to your Loki instance URL
- Set `SLACK_APP_TOKEN` and `SLACK_BOT_TOKEN` for Slack integration

## Running the System

1. Start the Loki service:
```bash
docker-compose up -d
```

2. Start the debugging agents:
```bash
python src/main.py
```

## Configuration

### Templates

The system uses JSON-based templates for generating RCA reports. Templates are stored in `data/templates/`.

Example template structure:
```json
{
    "title": "Root Cause Analysis Report",
    "sections": [
        {
            "name": "Executive Summary",
            "template": "..."
        },
        {
            "name": "Error Analysis",
            "template": "..."
        }
    ],
    "footer": "..."
}
```

### Slack Integration

The system can be configured to:
- Monitor specific Slack channels for alerts
- Send RCA reports to designated channels
- Provide real-time updates on debugging progress

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License 