#!/usr/bin/env python3
"""
Debug Agent CLI - Command-line interface for the debugging agents system.
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from dotenv import load_dotenv

# Get the path to the project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Ensure logs directory exists
logs_dir = os.path.join(project_root, "logs")
os.makedirs(logs_dir, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(logs_dir, "debug_agent.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("debug_agent_cli")

# Load environment variables
load_dotenv()

def validate_environment():
    """Validate that required environment variables are set."""
    required_vars = ["OPENAI_API_KEY"]
    
    # Check for provider-specific requirements
    llm_provider = os.getenv('LLM_PROVIDER', 'openai').lower()
    
    if llm_provider == 'openai':
        required_vars = ["OPENAI_API_KEY"]
    elif llm_provider == 'bedrock':
        required_vars = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION"]
    elif llm_provider == 'ollama':
        # Ollama just needs a running server, no keys required
        required_vars = []
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables for {llm_provider}: {', '.join(missing_vars)}")
        logger.error("Please set these variables in your .env file")
        return False
    
    return True

def forecast_command(args):
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

def debug_command(args):
    """Run the debug pipeline for a specific issue."""
    try:
        from src.main import run_realtime_debugging
        
        issue_id = args.issue_id
        logger.info(f"Starting real-time debugging for issue: {issue_id}")
        
        try:
            # Run the debugging process
            result = run_realtime_debugging(issue_id, llm_provider=args.llm_provider)
            
            # Print results summary
            print(f"\nDebugging Complete!")
            print(f"=================")
            
            # Check if there was an error
            if result and 'error' in result:
                print(f"Error encountered: {result['error']}")
                print(f"Check debug_agent.log for more details.")
                return
            
            # Display document URL if available
            if result and 'document_url' in result:
                print(f"Results document: {result['document_url']}")
            else:
                print("No document URL returned. Check debug_agent.log for details.")
            
            # Open document in browser if requested
            if args.open_doc and result and 'document_url' in result:
                if not result['document_url'].startswith('Error:'):
                    try:
                        import webbrowser
                        webbrowser.open(result['document_url'])
                        print(f"Opening document in browser...")
                    except Exception as e:
                        logger.error(f"Failed to open document: {e}")
                        print(f"Could not open document in browser: {e}")
            
        except Exception as e:
            logger.error(f"Error executing debugging pipeline: {e}", exc_info=True)
            print(f"\nError: {str(e)}")
            print("Check debug_agent.log for more details.")
            
    except ImportError as e:
        logger.error(f"Import error: {e}")
        print(f"\nFailed to import required modules: {e}")
        print("Make sure you're in the project directory or have installed the package correctly.")
        sys.exit(1)

def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description='Debugging Agents CLI',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Forecast command
    forecast_parser = subparsers.add_parser('forecast', help='Run forecasting pipeline')
    forecast_parser.add_argument('-v', '--verbose', action='store_true', help='Show detailed output')
    
    # Debug command
    debug_parser = subparsers.add_parser('debug', help='Run debugging pipeline for an issue')
    debug_parser.add_argument('issue_id', help='Issue ID to debug')
    debug_parser.add_argument('--time-window', type=int, default=60, 
                             help='Time window in minutes to look back')
    debug_parser.add_argument('--open-doc', action='store_true', 
                             help='Open document in browser when done')
    debug_parser.add_argument('--llm-provider', type=str, choices=['openai', 'ollama', 'bedrock'],
                             help='LLM provider to use (default: from .env)')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Display system information')
    
    args = parser.parse_args()
    
    # Check if no command was provided
    if not args.command:
        parser.print_help()
        return
    
    # Validate environment variables
    if not validate_environment():
        return
    
    # Execute appropriate command
    if args.command == 'forecast':
        forecast_command(args)
    elif args.command == 'debug':
        debug_command(args)
    elif args.command == 'info':
        try:
            # Print system info
            from src import __version__
            print(f"Debugging Agents System v{__version__}")
            print(f"Python version: {sys.version}")
            llm_provider = os.getenv('LLM_PROVIDER', 'openai')
            llm_model = os.getenv(f"{llm_provider.upper()}_MODEL", 'unknown')
            print(f"LLM Provider: {llm_provider}")
            print(f"LLM Model: {llm_model}")
            
            # Show actual data directory at project root
            data_dir = os.path.join(project_root, "data")
            print(f"Data directory: {os.path.abspath(data_dir)}")
        except ImportError:
            # Fallback if we can't import the version
            print(f"Debugging Agents System")
            print(f"Python version: {sys.version}")
            llm_provider = os.getenv('LLM_PROVIDER', 'openai')
            llm_model = os.getenv(f"{llm_provider.upper()}_MODEL", 'unknown')
            print(f"LLM Provider: {llm_provider}")
            print(f"LLM Model: {llm_model}")
            
            # Show actual data directory at project root
            data_dir = os.path.join(project_root, "data")
            print(f"Data directory: {os.path.abspath(data_dir)}")

if __name__ == "__main__":
    main() 