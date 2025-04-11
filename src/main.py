import os
import argparse
import logging
import yaml
from dotenv import load_dotenv
import sys
from logging.handlers import RotatingFileHandler

# Import components
from src.coordination.crew_manager import DebugCrew
from src.forecasting.service_log_forecaster import ServiceLogForecaster
from src.forecasting.alert_forecaster import AlertForecaster
from src.realtime.context_builder import ContextBuilder
from src.realtime.debug_plan_creator import DebugPlanCreator
from src.realtime.executor import Executor
from src.realtime.analyzer import Analyzer
from src.realtime.document_generator import DocumentGenerator
from src.utils.llm_factory import LLMFactory

# Load environment variables
load_dotenv()

# Get the path to the project root (one level up from src directory)
src_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(src_dir)

# Ensure logs directory exists
logs_dir = os.path.join(project_root, "logs")
os.makedirs(logs_dir, exist_ok=True)

# Ensure config directory exists
config_dir = os.path.join(project_root, "config")
os.makedirs(config_dir, exist_ok=True)

# Let debug_agent_cli.py handle the logging configuration to ensure consistency
# We'll just get a logger here and start using it
logger = logging.getLogger(__name__)

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

def run_realtime_debugging(issue_id=None, llm_provider_or_model=None):
    """
    Run the real-time debugging pipeline
    
    Args:
        issue_id: The ID of the issue to debug
        llm_provider_or_model: The LLM provider or model name to use (defaults to env var LLM_PROVIDER)
    
    Returns:
        Dictionary with debugging results
    """
    logger.info(f"Starting real-time debugging for issue: {issue_id}")
    
    # Use specified provider/model or default from env
    raw_provider = llm_provider_or_model or os.getenv('LLM_PROVIDER', 'openai')
    
    # Clean the provider string (remove any comments)
    provider_or_model = raw_provider.split('#')[0].strip()
    
    logger.info(f"Using LLM provider/model: {provider_or_model}")
    
    try:
        # Initialize the Crew
        debug_crew = DebugCrew(llm_provider_or_model=provider_or_model)
        
        # Initialize agents
        context_builder = ContextBuilder()
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
        result = debug_crew.run(issue_id=issue_id)
        
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
        
    except Exception as e:
        logger.error(f"Error during debugging process: {str(e)}")
        # Create a minimal result dictionary with error information
        error_result = {
            "error": str(e),
            "document_url": f"Error: {str(e)}"
        }
        return error_result

def main():
    parser = argparse.ArgumentParser(description='AI Debugging Agents')
    parser.add_argument('--mode', choices=['forecast', 'debug', 'both'], 
                        default='both', help='Operation mode')
    parser.add_argument('--issue-id', type=str, help='Issue ID for debugging')
    parser.add_argument('--llm-provider', type=str, choices=['openai', 'ollama', 'bedrock', 'anthropic'],
                        help='LLM provider to use (default: from .env)')
    
    args = parser.parse_args()
    
    if args.mode in ['forecast', 'both']:
        setup_forecasting_pipeline()
    
    if args.mode in ['debug', 'both']:
        if args.issue_id:
            run_realtime_debugging(args.issue_id, llm_provider_or_model=args.llm_provider)
        else:
            logger.error("Error: Issue ID required for debugging mode")
            return
    
if __name__ == "__main__":
    main()

def validate_environment():
    """Validate that required environment variables are set."""
    from src.utils.llm_factory import LLMFactory
    
    llm_provider = os.getenv('LLM_PROVIDER', 'openai').lower()
    return LLMFactory.validate_environment(llm_provider) 