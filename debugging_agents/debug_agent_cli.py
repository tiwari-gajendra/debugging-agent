#!/usr/bin/env python3
"""
Debug Agent CLI - Command-line interface for the debugging agents system.
"""

import os
import sys
import argparse
import logging
import yaml
from datetime import datetime
from dotenv import load_dotenv
from logging.handlers import RotatingFileHandler

# Get the path to the project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = script_dir  # The script is now in the project root

# Ensure logs directory exists
logs_dir = os.path.join(project_root, "logs")
os.makedirs(logs_dir, exist_ok=True)

# Ensure config directory exists
config_dir = os.path.join(project_root, "config")
os.makedirs(config_dir, exist_ok=True)

# Define default logging config
default_logging_config = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "date_format": "%Y-%m-%d %H:%M:%S",
    "handlers": {
        "file": "logs/debug_agent.log",
        "console": True,
        "rotation": {
            "enabled": True,
            "max_bytes": 10485760,  # 10MB
            "backup_count": 5
        }
    },
    "settings": {
        "module_levels": {
            "httpx": "WARNING",
            "matplotlib": "WARNING",
            "urllib3": "WARNING",
            "langchain": "INFO",
            "src.realtime": "DEBUG"
        }
    }
}

# Load logging config from file or use default
logging_config_path = os.path.join(config_dir, "logging.yaml")
if os.path.exists(logging_config_path):
    try:
        with open(logging_config_path, 'r') as file:
            logging_config = yaml.safe_load(file)
    except Exception as e:
        print(f"Error loading logging config: {e}. Using default.")
        logging_config = default_logging_config
else:
    # Create default config file
    try:
        with open(logging_config_path, 'w') as file:
            yaml.dump(default_logging_config, file)
        logging_config = default_logging_config
    except Exception as e:
        print(f"Error creating default logging config: {e}. Using default in memory.")
        logging_config = default_logging_config

# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(getattr(logging, logging_config.get("level", "INFO")))

# Clear any existing handlers
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

# Create formatters
log_format = logging_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
date_format = logging_config.get("date_format", "%Y-%m-%d %H:%M:%S")
formatter = logging.Formatter(log_format, datefmt=date_format)

# Add file handler with rotation if enabled
if logging_config.get("handlers", {}).get("file"):
    file_path = os.path.join(project_root, logging_config["handlers"]["file"])
    
    rotation_config = logging_config.get("handlers", {}).get("rotation", {})
    if rotation_config.get("enabled", False):
        file_handler = RotatingFileHandler(
            file_path,
            maxBytes=rotation_config.get("max_bytes", 10485760),
            backupCount=rotation_config.get("backup_count", 5)
        )
    else:
        file_handler = logging.FileHandler(file_path)
    
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

# Add console handler
if logging_config.get("handlers", {}).get("console", True):
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

# Set specific module log levels
module_levels = logging_config.get("settings", {}).get("module_levels", {})
for module, level in module_levels.items():
    module_logger = logging.getLogger(module)
    module_logger.setLevel(getattr(logging, level))

# Get our application logger
logger = logging.getLogger("debug_agent_cli")
logger.info("Logging initialized with configuration from: " + 
           (logging_config_path if os.path.exists(logging_config_path) else "default settings"))

# Load environment variables
load_dotenv()

def validate_environment():
    """Validate that required environment variables are set."""
    from src.utils.llm_factory import LLMFactory
    
    llm_provider = os.getenv('LLM_PROVIDER', 'openai').lower()
    return LLMFactory.validate_environment(llm_provider)

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
        
        # Use model if specified, otherwise use llm_provider
        llm_identifier = args.model or args.llm_provider
        
        try:
            # Run the debugging process
            result = run_realtime_debugging(issue_id, llm_provider_or_model=llm_identifier)
            
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
    debug_parser.add_argument('--llm-provider', type=str, choices=['openai', 'ollama', 'bedrock', 'anthropic'],
                             help='LLM provider to use (default: from .env)')
    debug_parser.add_argument('--model', type=str, 
                             help='Specific model name to use (e.g., claude-3-sonnet, gpt-4-turbo)')
    
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
            # Clean the provider string (remove any comments)
            llm_provider = llm_provider.split('#')[0].strip()
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
            # Clean the provider string (remove any comments)
            llm_provider = llm_provider.split('#')[0].strip()
            llm_model = os.getenv(f"{llm_provider.upper()}_MODEL", 'unknown')
            print(f"LLM Provider: {llm_provider}")
            print(f"LLM Model: {llm_model}")
            
            # Show actual data directory at project root
            data_dir = os.path.join(project_root, "data")
            print(f"Data directory: {os.path.abspath(data_dir)}")

if __name__ == "__main__":
    main() 