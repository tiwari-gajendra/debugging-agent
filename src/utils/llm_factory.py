"""
LLM Factory - Factory for creating LLM instances from different providers.
"""

import os
from typing import Dict, Any, Optional, Union, Type
from dotenv import load_dotenv
import logging

# LangChain imports
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM

# Import LiteLLM for direct bedrock access
import litellm
from litellm import completion
from openai import OpenAI

# Set up logger
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class DirectBedrockClient:
    """A direct Bedrock client that mimics the OpenAI client interface for CrewAI compatibility"""
    
    def __init__(self, model: str, region_name: str, temperature: float = 0.2, **kwargs):
        """
        Initialize a direct Bedrock client.
        
        Args:
            model: The Bedrock model ID
            region_name: AWS region name
            temperature: Temperature for response generation
            **kwargs: Additional model configuration
        """
        self.model = model
        self.region_name = region_name
        self.temperature = temperature
        self.model_kwargs = kwargs.get("model_kwargs", {})
        
        # Ensure model has bedrock/ prefix
        if not model.startswith("bedrock/"):
            self.bedrock_model = f"bedrock/{model}"
        else:
            self.bedrock_model = model
            
        # Set model kwargs if not provided
        if "temperature" not in self.model_kwargs:
            self.model_kwargs["temperature"] = temperature
            
        # For CrewAI compatibility
        self.api_key = "sk-valid-bedrock-key"
        
        # Configure litellm
        litellm.drop_params = True  # Don't pass invalid params
        litellm.set_verbose = True  # Enable detailed logging
        
        # Configure for CrewAI
        os.environ["OPENAI_API_KEY"] = self.api_key
        os.environ["CREW_LLM_PROVIDER"] = "bedrock"
        
        logger.info(f"Initialized DirectBedrockClient with model: {self.bedrock_model}")
    
    # This is needed for compatibility with CrewAI
    @property
    def api_type(self):
        return "open_ai"
        
    def chat(self):
        """Returns a completion interface compatible with OpenAI's client"""
        return self.ChatCompletions(self)
    
    # Add compatibility methods expected by CrewAI
    def __getattr__(self, name):
        if name == "api_base":
            return "https://bedrock.us-west-2.amazonaws.com"
        elif name == "api_version":
            return "2023-05-15"
        elif name == "api_type":
            return "open_ai"
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")
    
    class ChatCompletions:
        """
        An OpenAI-compatible chat completions interface for CrewAI.
        This matches what CrewAI expects from an OpenAI client.
        """
        
        def __init__(self, client):
            self.client = client
        
        def create(self, messages, **kwargs):
            """
            Create a chat completion using litellm's completion function.
            
            Args:
                messages: Chat messages in OpenAI format
                **kwargs: Additional parameters
                
            Returns:
                OpenAI-compatible response
            """
            try:
                # Merge model kwargs with any provided kwargs
                model_kwargs = self.client.model_kwargs.copy()
                if "model_kwargs" in kwargs:
                    model_kwargs.update(kwargs.pop("model_kwargs"))
                
                # Update with any directly provided args
                if "temperature" in kwargs:
                    model_kwargs["temperature"] = kwargs.pop("temperature")
                
                # Remove any OpenAI-specific parameters that might cause issues
                for param in ['function_call', 'functions', 'tools']:
                    if param in kwargs:
                        kwargs.pop(param)
                        
                logger.debug(f"Making LiteLLM call with model: {self.client.bedrock_model}")
                
                # Make the LiteLLM call with specific provider
                response = completion(
                    model=self.client.bedrock_model,
                    messages=messages,
                    api_key="not-needed-for-bedrock",  # LiteLLM will use boto credentials
                    model_kwargs=model_kwargs,
                    **kwargs
                )
                
                return response
                
            except Exception as e:
                logger.error(f"Error in DirectBedrockClient: {str(e)}")
                # Log more details for debugging
                logger.error(f"Messages: {messages}")
                logger.error(f"Model: {self.client.bedrock_model}")
                logger.error(f"Kwargs: {kwargs}")
                raise e

class ModelConfig:
    """Configuration class for LLM models"""
    
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

class ModelRegistry:
    """Registry for managing available LLM models"""
    
    # Dictionary to store registered models
    _models: Dict[str, ModelConfig] = {}
    
    @classmethod
    def register(cls, name: str, config: ModelConfig) -> None:
        """
        Register a model configuration.
        
        Args:
            name: The name to register the model under
            config: The model configuration
        """
        cls._models[name.lower()] = config
        logger.debug(f"Registered model: {name}")
    
    @classmethod
    def get(cls, name: str) -> Optional[ModelConfig]:
        """
        Get a model configuration by name.
        
        Args:
            name: The name of the model
            
        Returns:
            The model configuration or None if not found
        """
        return cls._models.get(name.lower())
    
    @classmethod
    def list_models(cls) -> Dict[str, str]:
        """
        List all registered models.
        
        Returns:
            Dictionary of model names and their providers
        """
        return {name: config.provider for name, config in cls._models.items()}

class LLMFactory:
    """Factory class to create LLM instances from different providers."""
    
    # Register default models
    @classmethod
    def register_defaults(cls) -> None:
        """Register default models in the registry"""
        # OpenAI models
        ModelRegistry.register("gpt-4", ModelConfig(
            provider="openai",
            model_id="gpt-4",
            requires_auth=True,
            auth_env_var="OPENAI_API_KEY"
        ))
        
        ModelRegistry.register("gpt-4-turbo", ModelConfig(
            provider="openai",
            model_id="gpt-4-turbo-preview", 
            requires_auth=True,
            auth_env_var="OPENAI_API_KEY"
        ))
        
        ModelRegistry.register("gpt-3.5-turbo", ModelConfig(
            provider="openai",
            model_id="gpt-3.5-turbo",
            requires_auth=True,
            auth_env_var="OPENAI_API_KEY"
        ))
        
        # Ollama models
        ModelRegistry.register("llama3", ModelConfig(
            provider="ollama",
            model_id="llama3",
            provider_options={"base_url": "http://localhost:11434"},
            requires_auth=False
        ))
        
        ModelRegistry.register("mistral", ModelConfig(
            provider="ollama",
            model_id="mistral",
            provider_options={"base_url": "http://localhost:11434"},
            requires_auth=False
        ))
        
        # Bedrock models
        ModelRegistry.register("claude-3-sonnet", ModelConfig(
            provider="bedrock",
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            provider_options={"region_name": os.getenv('AWS_DEFAULT_REGION', 'us-east-1')},
            requires_auth=True
        ))
        
        ModelRegistry.register("claude-3-haiku", ModelConfig(
            provider="bedrock",
            model_id="anthropic.claude-3-haiku-20240307-v1:0",
            provider_options={"region_name": os.getenv('AWS_DEFAULT_REGION', 'us-east-1')},
            requires_auth=True
        ))
        
        ModelRegistry.register("llama3-70b", ModelConfig(
            provider="bedrock",
            model_id="meta.llama3-70b-instruct-v1:0",
            provider_options={"region_name": os.getenv('AWS_DEFAULT_REGION', 'us-east-1')},
            requires_auth=True
        ))
    
    @staticmethod
    def create_llm(
        provider_or_model: str, 
        model_name: Optional[str] = None, 
        **kwargs
    ) -> Union[BaseChatModel, DirectBedrockClient]:
        """
        Create an LLM instance based on the specified provider or model name.
        
        Args:
            provider_or_model: The LLM provider or a registered model name
            model_name: The specific model name to use (if provider is specified)
            **kwargs: Additional arguments to pass to the LLM constructor
            
        Returns:
            An instance of a LangChain chat model or compatible client
        """
        # Ensure default models are registered
        if not ModelRegistry._models:
            LLMFactory.register_defaults()
            
        # Set default temperature if not provided
        temperature = kwargs.get('temperature', float(os.getenv('TEMPERATURE', 0.2)))
        
        # Check if the provided string is a registered model
        model_config = ModelRegistry.get(provider_or_model)
        
        if model_config:
            # If a registered model was specified, use its configuration
            provider = model_config.provider
            model = model_config.model_id
            
            # Merge options from config and kwargs
            merged_kwargs = {**model_config.provider_options, **kwargs}
            model_kwargs = {**model_config.model_options}
            if 'model_kwargs' in kwargs:
                model_kwargs.update(kwargs.pop('model_kwargs'))
                
            merged_kwargs['model_kwargs'] = model_kwargs
        else:
            # If not a registered model, treat as provider
            provider = provider_or_model.lower()
            # Use provided model name or get from env
            model = model_name
        
        # Create LLM based on provider
        if provider == 'openai':
            model = model or os.getenv('OPENAI_MODEL', 'gpt-4')
            api_key = os.getenv('OPENAI_API_KEY')
            
            # Import ChatOpenAI here to ensure it's always available
            from langchain_openai import ChatOpenAI
            
            return ChatOpenAI(
                model=model,
                temperature=temperature,
                api_key=api_key,
                **kwargs
            )
        
        elif provider == 'ollama':
            model = model or os.getenv('OLLAMA_MODEL', 'llama3')
            base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
            
            # Import OllamaLLM here to ensure it's always available
            from langchain_ollama import OllamaLLM
            
            return OllamaLLM(
                model=model,
                temperature=temperature,
                base_url=base_url,
                **kwargs
            )
        
        elif provider == 'bedrock':
            # For Bedrock, create our DirectBedrockClient
            model = model or os.getenv('BEDROCK_MODEL', 'anthropic.claude-3-sonnet-20240229-v1:0')
            region = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
            
            # Set up model kwargs
            model_kwargs = kwargs.get('model_kwargs', {})
            model_kwargs['temperature'] = temperature
            
            logger.info(f"Creating DirectBedrockClient with model={model}, region={region}")
            
            # Create a direct Bedrock client
            return DirectBedrockClient(
                model=model,
                region_name=region,
                temperature=temperature,
                model_kwargs=model_kwargs
            )
        
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    @staticmethod
    def register_custom_model(
        name: str,
        provider: str,
        model_id: str,
        options: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register a custom model configuration.
        
        Args:
            name: The name to register the model under
            provider: The provider name
            model_id: The model identifier
            options: Additional options for the model
        """
        provider = provider.lower()
        options = options or {}
        
        if provider == 'openai':
            config = ModelConfig(
                provider=provider,
                model_id=model_id,
                requires_auth=True,
                auth_env_var="OPENAI_API_KEY",
                model_options=options
            )
        elif provider == 'ollama':
            config = ModelConfig(
                provider=provider,
                model_id=model_id,
                provider_options={"base_url": os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')},
                requires_auth=False,
                model_options=options
            )
        elif provider == 'bedrock':
            config = ModelConfig(
                provider=provider,
                model_id=model_id,
                provider_options={"region_name": os.getenv('AWS_DEFAULT_REGION', 'us-east-1')},
                requires_auth=True,
                model_options=options
            )
        else:
            raise ValueError(f"Unsupported provider for custom model: {provider}")
        
        ModelRegistry.register(name, config)
        logger.info(f"Registered custom model: {name} (provider: {provider}, model_id: {model_id})")
    
    @staticmethod
    def list_available_models() -> Dict[str, str]:
        """
        List all available models.
        
        Returns:
            Dictionary of model names and their providers
        """
        # Ensure defaults are registered
        if not ModelRegistry._models:
            LLMFactory.register_defaults()
            
        return ModelRegistry.list_models() 