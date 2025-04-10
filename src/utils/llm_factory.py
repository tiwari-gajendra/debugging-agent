"""
LLM Factory - Factory for creating LLM instances from different providers.
"""

import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# LangChain imports
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM
from langchain_community.chat_models import BedrockChat

# Load environment variables
load_dotenv()

class LLMFactory:
    """Factory class to create LLM instances from different providers."""
    
    @staticmethod
    def create_llm(provider: str, model_name: Optional[str] = None, **kwargs) -> BaseChatModel:
        """
        Create an LLM instance based on the specified provider.
        
        Args:
            provider: The LLM provider ('openai', 'ollama', 'bedrock')
            model_name: The specific model name to use
            **kwargs: Additional arguments to pass to the LLM constructor
            
        Returns:
            An instance of a LangChain chat model
        """
        provider = provider.lower()
        
        # Set default temperature if not provided
        temperature = kwargs.get('temperature', float(os.getenv('TEMPERATURE', 0.2)))
        
        if provider == 'openai':
            model = model_name or os.getenv('OPENAI_MODEL', 'gpt-4')
            api_key = os.getenv('OPENAI_API_KEY')
            return ChatOpenAI(
                model=model,
                temperature=temperature,
                api_key=api_key,
                **kwargs
            )
        
        elif provider == 'ollama':
            model = model_name or os.getenv('OLLAMA_MODEL', 'llama3')
            base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
            return OllamaLLM(
                model=model,
                temperature=temperature,
                base_url=base_url,
                **kwargs
            )
        
        elif provider == 'bedrock':
            model = model_name or os.getenv('BEDROCK_MODEL', 'anthropic.claude-3-sonnet-20240229-v1:0')
            region = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
            return BedrockChat(
                model_id=model,
                region_name=region,
                temperature=temperature,
                **kwargs
            )
        
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}") 