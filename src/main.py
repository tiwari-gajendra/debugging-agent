import os
import argparse
import logging
import logging.config
import yaml
from dotenv import load_dotenv
import sys
from logging.handlers import RotatingFileHandler
import asyncio
from pathlib import Path

# Import components
from src.manager.crew_manager import DebugCrew
from src.forecasting.service_log_forecaster import ServiceLogForecaster
from src.forecasting.alert_forecaster import AlertForecaster
from src.realtime.context_builder import ContextBuilder
from src.realtime.debug_plan_creator import DebugPlanCreator
from src.realtime.executor import Executor
from src.realtime.analyzer import Analyzer
from src.realtime.document_generator import DocumentGenerator
from src.utils.llm_factory import LLMFactory
from src.integrations.slack_handler import SlackHandler
from src.integrations.loki_client import LokiClient

# Load environment variables
load_dotenv()

# Get the path to the project root
script_dir = Path(__file__).parent.parent
project_root = script_dir

# Ensure logs directory exists
logs_dir = project_root / "data" / "logs" / "debug_agent"
logs_dir.mkdir(parents=True, exist_ok=True)

# Ensure service_logs directory exists
service_logs_dir = project_root / "data" / "logs" / "service_logs"
service_logs_dir.mkdir(parents=True, exist_ok=True)

# Load logging configuration
logging_config_path = project_root / "config" / "logging.yaml"
if logging_config_path.exists():
    with open(logging_config_path, 'r') as f:
        config = yaml.safe_load(f)
        # Update file handler path to be relative to project root
        config['handlers']['file']['filename'] = str(logs_dir / "debug_agent.log")
        logging.config.dictConfig(config)
else:
    # Set up basic logging as fallback
    log_file = logs_dir / "debug_agent.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode='a', encoding='utf8')
        ]
    )

# Get our application logger
logger = logging.getLogger(__name__)
logger.info("Logging initialized")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def setup_forecasting_pipeline():
    """Set up and run the forecasting pipeline"""
    logger.info("Setting up forecasting pipeline...")
    service_log_forecaster = ServiceLogForecaster()
    alert_forecaster = AlertForecaster()
    
    # Run forecasting
    service_anomalies = service_log_forecaster.predict_anomalies()
    alert_predictions = alert_forecaster.predict_alerts()
    
    logger.info(f"Forecasting complete. Found {len(service_anomalies)} potential service anomalies")
    logger.info(f"Alert prediction complete. Found {len(alert_predictions)} potential alert patterns")
    
    return service_anomalies, alert_predictions

async def run_realtime_debugging(issue_id=None, llm_provider_or_model=None):
    """
    Run the real-time debugging pipeline
    
    Args:
        issue_id: The ID of the issue to debug
        llm_provider_or_model: The LLM provider or model name to use (defaults to env var LLM_PROVIDER)
    
    Returns:
        Dictionary with debugging results
        
    Raises:
        RuntimeError: If no logs are available or other critical errors occur
    """
    logger.info(f"Starting real-time debugging for issue: {issue_id}")
    
    # Use specified provider/model or default from env
    raw_provider = llm_provider_or_model or os.getenv('LLM_PROVIDER', 'openai')
    
    # Clean the provider string (remove any comments)
    provider_or_model = raw_provider.split('#')[0].strip()
    
    logger.info(f"Using LLM provider/model: {provider_or_model}")
    
    try:
        # Initialize context builder first to check log source availability
        context_builder = ContextBuilder()
        
        # Initialize the Crew
        debug_crew = DebugCrew(llm_provider_or_model=provider_or_model)
        
        # Initialize other agents
        plan_creator = DebugPlanCreator()
        executor = Executor()
        analyzer = Analyzer()
        doc_generator = DocumentGenerator()
        
        # Add agents to crew
        debug_crew.add_agents([
            context_builder,
            plan_creator,
            executor,
            analyzer,
            doc_generator
        ])
        
        # Run the debugging process
        result = await debug_crew.run(issue_id=issue_id)
        
        # Log information about the result for debugging
        logger.debug(f"Result type: {type(result)}")
        logger.debug(f"Result keys: {result.keys() if hasattr(result, 'keys') else 'No keys (not a dict)'}")
        
        # If result has crew_output, log some info about it too
        if 'crew_output' in result:
            logger.debug(f"Crew output type: {type(result['crew_output'])}")
            if hasattr(result['crew_output'], 'raw_output'):
                logger.debug(f"Crew raw output available: {len(str(result['crew_output'].raw_output))} chars")
        
        logger.info("Debugging process completed")
        logger.info(f"Results available at: {result.get('document_url', 'N/A')}")
        
        return result
        
    except RuntimeError as e:
        logger.error(f"Critical error during debugging process: {str(e)}")
        raise  # Re-raise the error to fail the application
    except Exception as e:
        logger.error(f"Unexpected error during debugging process: {str(e)}")
        raise RuntimeError(f"Unexpected error occurred: {str(e)}")

async def start_slack_handler():
    """Start the Slack handler for real-time alerts"""
    try:
        slack_handler = SlackHandler()
        await slack_handler.start()
        logger.info("Slack handler started successfully")
    except Exception as e:
        logger.error(f"Error starting Slack handler: {str(e)}")

async def main():
    parser = argparse.ArgumentParser(description='AI Debugging Agents')
    parser.add_argument('--mode', choices=['forecast', 'debug', 'slack', 'all'], 
                        default='all', help='Operation mode')
    parser.add_argument('--issue-id', type=str, help='Issue ID for debugging')
    parser.add_argument('--llm-provider', type=str, choices=['openai', 'ollama', 'bedrock', 'anthropic'],
                        help='LLM provider to use (default: from .env)')
    
    args = parser.parse_args()
    
    if args.mode in ['forecast', 'all']:
        setup_forecasting_pipeline()
    
    if args.mode in ['debug', 'all']:
        if args.issue_id:
            await run_realtime_debugging(args.issue_id, llm_provider_or_model=args.llm_provider)
        else:
            logger.error("Error: Issue ID required for debugging mode")
            return
    
    if args.mode in ['slack', 'all']:
        await start_slack_handler()
    
if __name__ == "__main__":
    asyncio.run(main())

def validate_environment():
    """Validate that required environment variables are set."""
    from src.utils.llm_factory import LLMFactory
    
    llm_provider = os.getenv('LLM_PROVIDER', 'openai').lower()
    return LLMFactory.validate_environment(llm_provider) 