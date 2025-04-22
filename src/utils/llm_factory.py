"""
LLM Factory - Simple interface for creating LLM instances from different providers
"""

import os
import sys
import json
import logging
import httpx
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# LangChain imports
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM
from langchain_aws import ChatBedrock
try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("ChatAnthropic not available. Install with 'pip install langchain-anthropic'")

# Import boto3 for AWS services
import boto3

# Import LiteLLM for CrewAI compatibility
try:
    import litellm
    from crewai.llm import LLM as CrewLLM
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("LiteLLM or CrewAI not available. Some integration features may be limited.")

# Set up logger
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class LLMFactory:
    """Simple factory for creating LLM instances based on provider type"""
    
    @staticmethod
    def create_llm(provider: str = None, model: str = None, **kwargs) -> BaseChatModel:
        """
        Create an LLM instance based on the provider and model.
        
        Args:
            provider: The LLM provider (openai, bedrock, ollama, anthropic, snowflake)
            model: The specific model to use (defaults to env variable)
            **kwargs: Additional arguments for the LLM
            
        Returns:
            An instance of a chat model
        """
        # Default to environment variables if not provided
        provider = provider or os.getenv('LLM_PROVIDER', 'openai')
        
        # Clean the provider string (remove any comments)
        provider = provider.split('#')[0].strip().lower()
        
        # Get temperature from env or use default
        temperature = kwargs.get('temperature', float(os.getenv('TEMPERATURE', 0.2)))
        if 'temperature' not in kwargs:
            kwargs['temperature'] = temperature
            
        logger.info(f"Creating LLM with provider={provider}")
        
        # Create LLM based on provider
        if provider == 'openai':
            model_name = model or os.getenv('OPENAI_MODEL', 'gpt-4')
            api_key = os.getenv('OPENAI_API_KEY')
            
            logger.info(f"Using OpenAI with model={model_name}")
            
            return ChatOpenAI(
                model=model_name,
                api_key=api_key,
                **kwargs
            )
        
        elif provider == 'ollama':
            # Get model from environment or use default
            model_name = model or os.getenv('OLLAMA_MODEL', 'llama3')
            base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
            
            # Remove 'ollama/' prefix if present (LangChain doesn't want it)
            if model_name.startswith('ollama/'):
                model_name = model_name[len('ollama/'):]
            
            logger.info(f"Using Ollama with model={model_name} at {base_url}")
            
            # For CrewAI compatibility
            os.environ["OPENAI_API_KEY"] = "sk-valid-ollama-key"
            os.environ["CREW_LLM_PROVIDER"] = "ollama"
            
            # Use the LangChain OllamaLLM implementation
            return OllamaLLM(
                model=model_name,
                base_url=base_url,
                **kwargs
            )
        
        elif provider == 'anthropic':
            # Direct Anthropic API integration
            try:
                from langchain_anthropic import ChatAnthropic
            except ImportError:
                raise ImportError("ChatAnthropic not available. Install with 'pip install langchain-anthropic'")
                
            # Get model from environment or use default
            model_name = model or os.getenv('ANTHROPIC_MODEL', 'claude-3-opus-20240229')
            api_key = os.getenv('ANTHROPIC_API_KEY')
            
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable is required for Anthropic provider")
                
            logger.info(f"Using Anthropic with model={model_name}")
            
            # For CrewAI compatibility
            os.environ["OPENAI_API_KEY"] = "sk-valid-anthropic-key"
            os.environ["CREW_LLM_PROVIDER"] = "anthropic"
            
            return ChatAnthropic(
                model=model_name,
                anthropic_api_key=api_key,
                **kwargs
            )
        
        elif provider == 'bedrock':
            # For AWS Bedrock (which hosts Anthropic models)
            model_name = model or os.getenv('BEDROCK_MODEL', 'anthropic.claude-3-sonnet-20240229-v1:0')
            region = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
            
            logger.info(f"Using AWS Bedrock with model={model_name}, region={region}")
            
            # Check if LiteLLM is available (required for CrewAI)
            if not LITELLM_AVAILABLE:
                raise ImportError("LiteLLM is required for Bedrock integration. Install with: pip install litellm crewai")
            
            # Set up credentials for LiteLLM
            aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
            aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
            
            # Set essential environment variables for LiteLLM
            os.environ["CREW_LLM_PROVIDER"] = "bedrock"
            os.environ["BEDROCK_MODEL_ID"] = model_name
            
            # Create a properly configured CrewAI-compatible LLM using LiteLLM
            # This uses the correct provider format for LiteLLM
            bedrock_model = f"bedrock/{model_name}"
            
            logger.info(f"Creating CrewAI-compatible LLM with provider bedrock, model={bedrock_model}")
            
            # Create kwargs dict for litellm
            litellm_kwargs = {
                "model": bedrock_model,
                "temperature": temperature,
                "aws_region_name": region,
                "verify_ssl": False,  # Disable SSL verification to fix certificate errors
            }
            
            # Add credentials if available
            if aws_access_key and aws_secret_key:
                litellm_kwargs["aws_access_key_id"] = aws_access_key
                litellm_kwargs["aws_secret_access_key"] = aws_secret_key
            
            # Return a CrewLLM instance
            return CrewLLM(
                provider="bedrock",
                model=model_name,
                config=litellm_kwargs
            )
            
        elif provider == 'snowflake' or provider == 'cortex':
            try:
                # First try to import directly from langchain
                from langchain.llms import SnowflakeConnection
                logger.info("Using SnowflakeConnection from langchain")
            except ImportError:
                try:
                    # Try from community module
                    from langchain_community.llms import SnowflakeConnection
                    logger.info("Using SnowflakeConnection from langchain_community")
                except ImportError:
                    raise ImportError("SnowflakeConnection not available. Install with: pip install snowflake-connector-python langchain langchain-community")
                
            # Get Snowflake connection parameters from environment
            account = os.getenv('SNOWFLAKE_ACCOUNT')
            user = os.getenv('SNOWFLAKE_USER')
            password = os.getenv('SNOWFLAKE_PASSWORD')
            database = os.getenv('SNOWFLAKE_DATABASE', 'CORTEX_DB')
            schema = os.getenv('SNOWFLAKE_SCHEMA', 'CORTEX_SCHEMA')
            warehouse = os.getenv('SNOWFLAKE_WAREHOUSE', 'CORTEX_WH')
            role = os.getenv('SNOWFLAKE_ROLE', 'CORTEX_ROLE')
            
            # Get model from environment or use default
            model_name = model or os.getenv('SNOWFLAKE_MODEL', 'llama-3-8b-instruct')
            
            logger.info(f"Using Snowflake Cortex AI with model={model_name}")
            
            # For CrewAI compatibility
            os.environ["OPENAI_API_KEY"] = "sk-valid-cortex-key"
            os.environ["CREW_LLM_PROVIDER"] = "snowflake"
            
            # Create connection parameters
            connection_kwargs = {
                "account": account,
                "user": user,
                "password": password,
                "database": database,
                "schema": schema,
                "warehouse": warehouse,
                "role": role,
            }
            
            # Create a Snowflake LLM using the connection
            return SnowflakeConnection(
                connection_kwargs=connection_kwargs,
                model_name=model_name, 
                temperature=temperature,
                **kwargs
            )
        
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}. Valid options are: openai, ollama, bedrock, anthropic, snowflake, cortex")
    
    @staticmethod
    def validate_environment(provider: str = None) -> bool:
        """
        Validate that required environment variables are set for a provider.
        
        Args:
            provider: The LLM provider to validate
            
        Returns:
            True if all required variables are set, False otherwise
        """
        # Get provider from environment if not specified
        provider = provider or os.getenv('LLM_PROVIDER', 'openai')
        
        # Clean the provider string - it might have comments
        provider = provider.split('#')[0].strip().lower()
        
        logger.info(f"Validating environment for LLM provider: {provider}")
        
        if provider == 'openai':
            required_vars = ["OPENAI_API_KEY"]
        elif provider == 'bedrock':
            required_vars = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION"]
        elif provider == 'anthropic':
            required_vars = ["ANTHROPIC_API_KEY"]
            # Check if langchain-anthropic is installed
            try:
                from langchain_anthropic import ChatAnthropic
            except ImportError:
                logger.error("Missing required package: langchain-anthropic. Install with: pip install langchain-anthropic")
                return False
        elif provider == 'ollama':
            # Ollama just needs a running server, no keys required
            required_vars = []
            logger.info(f"Using Ollama provider, no environment variables required")
            return True
        elif provider == 'snowflake' or provider == 'cortex':
            required_vars = ["SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD"]
            # Check if snowflake-connector-python is installed
            try:
                import snowflake.connector
            except ImportError:
                logger.error("Missing required package: snowflake-connector-python. Install with: pip install snowflake-connector-python")
                return False
        else:
            logger.error(f"Unknown provider: {provider}. Valid options are: openai, ollama, bedrock, anthropic, snowflake, cortex")
            return False
        
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            logger.error(f"Missing required environment variables for {provider}: {', '.join(missing_vars)}")
            logger.error("Please set these variables in your .env file")
            return False
        
        return True 