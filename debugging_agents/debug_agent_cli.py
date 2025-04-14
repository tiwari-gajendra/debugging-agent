#!/usr/bin/env python3
"""
Debug Agent CLI - Command-line interface for the debugging agents system.
"""

import os
import sys
import argparse
import logging
import logging.config
import yaml
import asyncio
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Get the path to the project root
script_dir = Path(__file__).parent
project_root = script_dir  # The script is now in the project root

# Ensure logs directory exists
logs_dir = project_root / "data" / "logs" / "debug_agent"
logs_dir.mkdir(parents=True, exist_ok=True)

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

# Load environment variables
load_dotenv()

def validate_environment():
    """Validate that required environment variables are set."""
    from src.utils.llm_factory import LLMFactory
    
    llm_provider = os.getenv('LLM_PROVIDER', 'openai').lower()
    return LLMFactory.validate_environment(llm_provider)

async def forecast_command(args):
    """Run the forecasting pipeline."""
    try:
        from src.main import setup_forecasting_pipeline
        
        logger.info("Starting forecasting pipeline")
        service_anomalies, alert_predictions = setup_forecasting_pipeline()
        
        logger.info(f"Forecasting complete. Found {len(service_anomalies)} potential service anomalies")
        logger.info(f"Alert prediction complete. Found {len(alert_predictions)} potential alert patterns")
        
        # Print results summary
        print(f"\nResults Summary:")
        print(f"===============")
        print(f"Detected {len(service_anomalies)} potential service anomalies")
        print(f"Detected {len(alert_predictions)} potential alert patterns")
        
        if args.verbose:
            print("\nService Anomalies:")
            for i, anomaly in enumerate(service_anomalies[:5]):  # Show top 5
                print(f"  {i+1}. {anomaly.get('timestamp')}: Severity {anomaly.get('severity')}")
                if i >= 4 and len(service_anomalies) > 5:
                    print(f"  ... and {len(service_anomalies) - 5} more")
            
            print("\nAlert Patterns:")
            for i, alert in enumerate(alert_predictions[:5]):  # Show top 5
                print(f"  {i+1}. {alert.get('start_time')} to {alert.get('end_time')}")
                print(f"     Probability: {alert.get('incident_probability')}")
                print(f"     Recommendation: {alert.get('recommendation')}")
                if i >= 4 and len(alert_predictions) > 5:
                    print(f"  ... and {len(alert_predictions) - 5} more")
    except ImportError:
        logger.error("Failed to import modules. Make sure you're in the project directory.")
        sys.exit(1)

async def debug_command(args):
    """Run the debugging pipeline for an issue."""
    try:
        from src.main import run_realtime_debugging
        
        logger.info(f"Starting real-time debugging for issue: {args.issue_id}")
        
        print("\nDebugging Complete!")
        print("=================")
        
        result = await run_realtime_debugging(
            issue_id=args.issue_id,
            llm_provider_or_model=args.llm_provider or args.model
        )
        
        if result and 'error' in result:
            print(f"\nError: {result['error']}")
            print("Check debug_agent.log for more details.")
            return 1
            
        if result and 'document_url' in result:
            print(f"\nResults available at: {result['document_url']}")
            
            if args.open_doc:
                import webbrowser
                webbrowser.open(result['document_url'])
                
        return 0
        
    except ImportError as e:
        logger.error(f"Import error: {str(e)}")
        print(f"\nFailed to import required modules: {str(e)}")
        print("Make sure you're in the project directory or have installed the package correctly.")
        return 1
        
    except Exception as e:
        logger.error(f"Error executing debugging pipeline: {str(e)}")
        print(f"\nError: {str(e)}")
        print("Check debug_agent.log for more details.")
        return 1

async def main():
    parser = argparse.ArgumentParser(description='Debugging Agents CLI')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Forecast command
    forecast_parser = subparsers.add_parser('forecast', help='Run forecasting pipeline')
    forecast_parser.add_argument('--time-window', type=int, default=60,
                               help='Time window in minutes to look back')
    
    # Debug command
    debug_parser = subparsers.add_parser('debug', help='Run debugging pipeline for an issue')
    debug_parser.add_argument('issue_id', help='Issue ID to debug')
    debug_parser.add_argument('--time-window', type=int,
                            help='Time window in minutes to look back')
    debug_parser.add_argument('--open-doc', action='store_true',
                            help='Open document in browser when done')
    debug_parser.add_argument('--llm-provider', choices=['openai', 'ollama', 'bedrock', 'anthropic'],
                            help='LLM provider to use (default: from .env)')
    debug_parser.add_argument('--model', help='Specific model name to use (e.g., claude-3-sonnet, gpt-4-turbo)')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Display system information')
    
    args = parser.parse_args()
    
    if args.command == 'forecast':
        return await forecast_command(args)
    elif args.command == 'debug':
        return await debug_command(args)
    elif args.command == 'info':
        return info_command(args)
    else:
        parser.print_help()
        return 1

if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1) 