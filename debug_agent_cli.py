#!/usr/bin/env python3
"""
Debugging Agents CLI - Command line interface for the debugging agents system.
"""

import os
import sys
import argparse
import logging
import logging.config
import yaml
import signal
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Add the src directory to the Python path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

# Import the main module
from main import run_realtime_debugging, setup_forecasting_pipeline, start_slack_handler

# Load environment variables
load_dotenv()

# Global flag for graceful shutdown
shutdown_requested = False

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global shutdown_requested
    if not shutdown_requested:
        print("\nShutting down gracefully...")
        shutdown_requested = True
        # Let the current operation complete
        if signum == signal.SIGINT:
            sys.exit(0)
        elif signum == signal.SIGTERM:
            sys.exit(0)

def setup_logging():
    """Set up logging configuration."""
    # Get the path to the project root
    project_root = Path(__file__).parent
    
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
    
    return logging.getLogger(__name__)

def main():
    """Main entry point for the CLI."""
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Set up logging
    logger = setup_logging()
    logger.info("Initializing Debugging Agents CLI")
    
    # Create argument parser
    parser = argparse.ArgumentParser(description='AI Debugging Agents CLI')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Display system information')
    
    # Debug command
    debug_parser = subparsers.add_parser('debug', help='Debug an issue')
    debug_parser.add_argument('issue_id', type=str, help='Issue ID to debug')
    debug_parser.add_argument('--llm-provider', type=str, 
                            choices=['openai', 'ollama', 'bedrock', 'anthropic'],
                            help='LLM provider to use (default: from .env)')
    
    # Forecast command
    forecast_parser = subparsers.add_parser('forecast', help='Run forecasting pipeline')
    
    # Slack command
    slack_parser = subparsers.add_parser('slack', help='Start Slack handler')
    
    # Parse arguments
    args = parser.parse_args()
    
    try:
        if args.command == 'info':
            # Display system information
            logger.info("System Information:")
            logger.info(f"Python Version: {sys.version}")
            logger.info(f"LLM Provider: {os.getenv('LLM_PROVIDER', 'Not set')}")
            logger.info(f"Log Directory: {Path(__file__).parent / 'data' / 'logs' / 'debug_agent'}")
            
        elif args.command == 'debug':
            # Run debugging process
            logger.info(f"Starting debugging process for issue {args.issue_id}")
            asyncio.run(run_realtime_debugging(args.issue_id, args.llm_provider))
            
        elif args.command == 'forecast':
            # Run forecasting pipeline
            logger.info("Starting forecasting pipeline")
            setup_forecasting_pipeline()
            
        elif args.command == 'slack':
            # Start Slack handler
            logger.info("Starting Slack handler")
            asyncio.run(start_slack_handler())
            
        else:
            parser.print_help()
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down gracefully...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 