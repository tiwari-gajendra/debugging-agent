"""
Unified LLM Provider - Simplified interface for working with different LLM providers

This module provides a unified interface for working with different Language Model providers,
including OpenAI, Ollama, and Bedrock/Anthropic. It handles:
- Model registration and configuration
- Environment validation
- Direct API access for Ollama
- Compatible LLM instances for CrewAI
"""

import os
import sys
import json
import logging
import httpx
from typing import Dict, List, Any, Optional, Union, Type
from dotenv import load_dotenv

# LangChain imports
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM
from langchain_community.llms.bedrock import Bedrock

# Import boto3 for AWS services
import boto3

# Set up logger
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class ModelConfig:
    """Configuration for an LLM model"""
    
    def __init__(
        self, 
        provider: str,
        model_id: str,
        model_options: Optional[Dict[str, Any]] = None,
        provider_options: Optional[Dict[str, Any]] = None,
        requires_auth: bool = True,
        auth_env_var: Optional[str] = None,
        default_temp: float = 0.2
    ):
        """
        Initialize a model configuration.
        
        Args:
            provider: The provider name (e.g., 'openai', 'bedrock')
            model_id: The model identifier
            model_options: Model-specific options
            provider_options: Provider-specific options
            requires_auth: Whether authentication is required
            auth_env_var: Environment variable for authentication
            default_temp: Default temperature
        """
        self.provider = provider.lower()
        self.model_id = model_id
        self.model_options = model_options or {}
        self.provider_options = provider_options or {}
        self.requires_auth = requires_auth
        self.auth_env_var = auth_env_var
        self.default_temp = default_temp

class LLMProvider:
    """Unified interface for working with different LLM providers
    
    This class provides a single point of access for all LLM functionality:
    - Create LLM instances for different providers
    - Register and retrieve model configurations
    - Direct API access for special cases like Ollama
    - Environment validation
    """
    
    # Registry of available models
    _models = {}
    
    @classmethod
    def register_model(cls, name: str, config: ModelConfig) -> None:
        """
        Register a model in the registry
        
        Args:
            name: Name to register the model under
            config: Configuration for the model
        """
        cls._models[name.lower()] = config
        logger.debug(f"Registered model: {name}")
    
    @classmethod
    def get_model_config(cls, name: str) -> Optional[ModelConfig]:
        """
        Get a model configuration by name
        
        Args:
            name: Model name to look up
            
        Returns:
            Model configuration or None if not found
        """
        return cls._models.get(name.lower())
    
    @classmethod
    def list_models(cls) -> Dict[str, str]:
        """
        List all registered models
        
        Returns:
            Dictionary of model names and providers
        """
        return {name: config.provider for name, config in cls._models.items()}
    
    @classmethod
    def register_defaults(cls) -> None:
        """Register default LLM models"""
        # OpenAI models
        cls.register_model("gpt-4", ModelConfig(
            provider="openai",
            model_id="gpt-4",
            requires_auth=True,
            auth_env_var="OPENAI_API_KEY"
        ))
        
        cls.register_model("gpt-3.5-turbo", ModelConfig(
            provider="openai",
            model_id="gpt-3.5-turbo",
            requires_auth=True,
            auth_env_var="OPENAI_API_KEY"
        ))
        
        # Ollama models
        cls.register_model("llama3", ModelConfig(
            provider="ollama",
            model_id="llama3",
            provider_options={"base_url": "http://localhost:11434"},
            requires_auth=False
        ))
        
        cls.register_model("deepseek-r1", ModelConfig(
            provider="ollama",
            model_id="deepseek-r1:8b",
            provider_options={"base_url": "http://localhost:11434"},
            requires_auth=False
        ))
        
        # Bedrock/Anthropic models
        cls.register_model("claude", ModelConfig(
            provider="bedrock",
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            requires_auth=True,
            auth_env_var="AWS_ACCESS_KEY_ID"
        ))
    
    @classmethod
    def create_llm(cls, provider_or_model: str = None, model: str = None, **kwargs) -> BaseChatModel:
        """
        Create an LLM instance based on the provider and model.
        
        This method supports two calling patterns:
        1. By model name: create_llm("gpt-4")
        2. By provider/model: create_llm("openai", "gpt-4")
        
        Args:
            provider_or_model: The LLM provider (openai, anthropic, bedrock, ollama) or a model name
            model: The specific model to use (if provider is specified)
            **kwargs: Additional arguments for the LLM
            
        Returns:
            An instance of a chat model
        """
        # Initialize defaults
        if not provider_or_model:
            provider_or_model = os.getenv('LLM_PROVIDER', 'openai')
        
        # Clean the provider string (remove any comments)
        provider_or_model = provider_or_model.split('#')[0].strip().lower()
        
        # Check if this is a model name in our registry
        model_config = cls.get_model_config(provider_or_model)
        
        if model_config:
            # We were given a model name, use its configuration
            provider = model_config.provider
            model_name = model_config.model_id
            
            # Apply config options
            for k, v in model_config.provider_options.items():
                if k not in kwargs:
                    kwargs[k] = v
                    
            for k, v in model_config.model_options.items():
                if k not in kwargs:
                    kwargs[k] = v
                    
            temperature = kwargs.get('temperature', model_config.default_temp)
        else:
            # Assume provider_or_model is a provider name
            provider = provider_or_model
            
            # Default model by provider
            if not model:
                if provider == 'openai':
                    model_name = os.getenv('OPENAI_MODEL', 'gpt-4')
                elif provider == 'ollama':
                    model_name = os.getenv('OLLAMA_MODEL', 'llama3')
                elif provider == 'bedrock' or provider == 'anthropic':
                    model_name = os.getenv('BEDROCK_MODEL', 'anthropic.claude-3-sonnet-20240229-v1:0')
                else:
                    model_name = model or 'gpt-4'  # Safe default
            else:
                model_name = model
                
            temperature = kwargs.get('temperature', float(os.getenv('TEMPERATURE', 0.2)))
            
        logger.info(f"Creating LLM with provider={provider}, model={model_name}")
        
        # Ensure temperature is applied consistently
        if 'temperature' not in kwargs:
            kwargs['temperature'] = temperature
        
        # Create LLM based on provider
        if provider == 'openai':
            api_key = os.getenv('OPENAI_API_KEY')
            
            logger.info(f"Using OpenAI with model={model_name}")
            
            return ChatOpenAI(
                model=model_name,
                api_key=api_key,
                **kwargs
            )
        
        elif provider == 'ollama':
            base_url = kwargs.get('base_url', os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'))
            
            # Ensure model has ollama/ prefix for LiteLLM compatibility
            langchain_model = model_name
            if model_name.startswith('ollama/'):
                langchain_model = model_name[len('ollama/'):]
            
            logger.info(f"Using Ollama with model={model_name} at {base_url}")
            
            # For CrewAI compatibility
            os.environ["OPENAI_API_KEY"] = "sk-valid-ollama-key"
            os.environ["CREW_LLM_PROVIDER"] = "ollama"
            
            # Use the LangChain OllamaLLM implementation
            return OllamaLLM(
                model=langchain_model,  # Use the name without ollama/ prefix for LangChain
                base_url=base_url,
                **kwargs
            )
        
        elif provider == 'bedrock' or provider == 'anthropic':
            # For AWS Bedrock (which hosts Anthropic models)
            region = kwargs.get('region', os.getenv('AWS_DEFAULT_REGION', 'us-east-1'))
            
            logger.info(f"Using AWS Bedrock with model={model_name}, region={region}")
            
            # For CrewAI compatibility
            os.environ["OPENAI_API_KEY"] = "sk-valid-bedrock-key"
            os.environ["CREW_LLM_PROVIDER"] = "bedrock"
            
            # Set up AWS session
            aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
            aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
            
            session_kwargs = {}
            if aws_access_key and aws_secret_key:
                session_kwargs = {
                    'aws_access_key_id': aws_access_key,
                    'aws_secret_access_key': aws_secret_key,
                    'region_name': region
                }
            
            bedrock_session = boto3.Session(**session_kwargs)
            bedrock_client = bedrock_session.client('bedrock-runtime', region_name=region)
            
            # Set up model kwargs
            model_kwargs = kwargs.get('model_kwargs', {})
            if 'temperature' in kwargs and 'temperature' not in model_kwargs:
                model_kwargs['temperature'] = kwargs['temperature']
            
            return Bedrock(
                client=bedrock_client,
                model_id=model_name,
                model_kwargs=model_kwargs,
                **kwargs
            )
        
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
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
        raw_provider = provider or os.getenv('LLM_PROVIDER', 'openai')
        
        # Clean the provider string - it might have comments if read directly from .env
        # Extract just the provider name without any comments
        provider = raw_provider.split('#')[0].strip().lower()
        
        logger.info(f"Validating environment for LLM provider: {provider}")
        
        if provider == 'openai':
            required_vars = ["OPENAI_API_KEY"]
        elif provider == 'bedrock' or provider == 'anthropic':
            required_vars = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION"]
        elif provider == 'ollama':
            # Ollama just needs a running server, no keys required
            required_vars = []
            logger.info(f"Using Ollama provider, no environment variables required")
            return True
        else:
            logger.error(f"Unknown provider: {provider}. Valid options are: openai, ollama, bedrock, anthropic")
            return False
        
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            logger.error(f"Missing required environment variables for {provider}: {', '.join(missing_vars)}")
            logger.error("Please set these variables in your .env file")
            return False
        
        return True
        
    @staticmethod
    def ollama_chat_completion(
        messages: List[Dict[str, str]],
        model: str = None,
        temperature: float = 0.2,
        base_url: str = None
    ) -> str:
        """
        Use Ollama Chat API directly without going through LiteLLM or CrewAI
        
        Direct method to communicate with Ollama API for when the standard
        abstractions don't work well. This is used as a fallback mechanism.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            model: The model name to use
            temperature: Temperature setting (0-1)
            base_url: Base URL for Ollama server
            
        Returns:
            Generated response as string
        """
        # Default to environment variables if not provided
        model = model or os.getenv('OLLAMA_MODEL', 'llama3')
        
        # Clean the model name - remove ollama/ prefix if present
        if model.startswith('ollama/'):
            model = model[len('ollama/'):]
        
        # Make sure we use the correct model name format - Ollama is sensitive to this
        if model == 'deepseek-r1' or model == 'deepseek-r1:8b':
            model = 'deepseek-r1:8b'  # Use the exact format from Ollama
            
        base_url = base_url or os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        
        # Remove trailing slash from base_url if present
        if base_url.endswith('/'):
            base_url = base_url[:-1]
            
        # Format the request 
        api_url = f"{base_url}/api/generate"
        
        # Convert chat messages to a single prompt
        prompt = ""
        system_prompt = None
        
        for message in messages:
            role = message.get("role", "").lower()
            content = message.get("content", "")
            
            if role == "system":
                system_prompt = content
            elif role == "user":
                prompt += f"User: {content}\n\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n\n"
            else:
                prompt += f"{content}\n\n"
                
        # Add final assistant prompt
        prompt += "Assistant: "
        
        request_data = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
        }
        
        # Add system prompt if available
        if system_prompt:
            request_data["system"] = system_prompt
        
        try:
            logger.info(f"Calling Ollama Generate API directly with model={model}")
            
            # Make the API call
            with httpx.Client(timeout=120.0) as client:
                response = client.post(api_url, json=request_data)
                response.raise_for_status()
                
                # Ollama's generate endpoint returns streaming responses
                # We'll collect all the text from the response
                all_text = ""
                for line in response.iter_lines():
                    if not line:
                        continue
                    
                    try:
                        response_part = json.loads(line)
                        response_text = response_part.get("response", "")
                        all_text += response_text
                        
                        # Check for done flag
                        if response_part.get("done", False):
                            break
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to decode Ollama response line: {line}")
                
                logger.debug(f"Ollama response length: {len(all_text)} chars")
                return all_text
        
        except Exception as e:
            logger.error(f"Error calling Ollama API: {str(e)}")
            return f"Error: {str(e)}"

# Register default models on module import
LLMProvider.register_defaults() 