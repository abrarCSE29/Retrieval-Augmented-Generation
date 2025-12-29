"""
LLM Factory for creating LLM instances based on provider configuration.
"""

from typing import Dict, Any, Optional
from .base_llm import BaseLLM
from .google_llm import GoogleGenerativeAI
from utils.logger import get_logger

logger = get_logger(__name__)


class LLMFactory:
    """
    Factory class for creating LLM instances.
    
    This factory provides a centralized way to instantiate different LLM providers
    based on configuration, making it easy to switch between providers and extend
    support for new ones.
    """
    
    # Supported providers and their corresponding classes
    _providers = {
        'google': GoogleGenerativeAI,
        # Future providers can be added here:
        # 'openai': OpenAI,
        # 'anthropic': Anthropic,
    }
    
    @classmethod
    def create_llm(cls, provider: str, config: Dict[str, Any]) -> BaseLLM:
        """
        Create an LLM instance based on the provider and configuration.
        
        Args:
            provider (str): The LLM provider (e.g., 'google', 'openai')
            config (Dict[str, Any]): Configuration dictionary for the provider
            
        Returns:
            BaseLLM: An instance of the appropriate LLM implementation
            
        Raises:
            ValueError: If the provider is not supported or configuration is invalid
        """
        provider = provider.lower()
        
        if provider not in cls._providers:
            supported_providers = list(cls._providers.keys())
            raise ValueError(
                f"Unsupported LLM provider: {provider}. "
                f"Supported providers: {supported_providers}"
            )
        
        llm_class = cls._providers[provider]
        
        try:
            # Create and return the LLM instance
            llm_instance = llm_class(config)
            logger.info(f"Created {llm_class.__name__} instance for provider: {provider}")
            return llm_instance
            
        except Exception as e:
            logger.error(f"Failed to create LLM instance for provider {provider}: {e}")
            raise ValueError(f"Failed to create LLM instance: {str(e)}")
    
    @classmethod
    def get_supported_providers(cls) -> list:
        """
        Get list of supported LLM providers.
        
        Returns:
            list: List of supported provider names
        """
        return list(cls._providers.keys())
    
    @classmethod
    def is_provider_supported(cls, provider: str) -> bool:
        """
        Check if a provider is supported.
        
        Args:
            provider (str): The provider name to check
            
        Returns:
            bool: True if the provider is supported, False otherwise
        """
        return provider.lower() in cls._providers
    
    @classmethod
    def create_from_env(cls, provider: Optional[str] = None) -> BaseLLM:
        """
        Create an LLM instance from environment variables.
        
        Args:
            provider (Optional[str]): The provider to use. If None, uses LLM_PROVIDER env var.
            
        Returns:
            BaseLLM: An LLM instance configured from environment variables
            
        Raises:
            ValueError: If provider is not specified and LLM_PROVIDER env var is not set
        """
        import os
        
        # Determine provider
        if provider is None:
            provider = os.getenv('LLM_PROVIDER')
            if not provider:
                raise ValueError("Provider must be specified either as parameter or via LLM_PROVIDER environment variable")
        
        # Build configuration from environment variables
        config = cls._build_config_from_env(provider)
        
        # Create and return LLM instance
        return cls.create_llm(provider, config)
    
    @classmethod
    def _build_config_from_env(cls, provider: str) -> Dict[str, Any]:
        """
        Build configuration dictionary from environment variables.
        
        Args:
            provider (str): The LLM provider
            
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        import os
        
        config = {}
        
        if provider == 'google':
            # Google-specific environment variables
            config['api_key'] = os.getenv('GOOGLE_API_KEY')
            config['model_name'] = os.getenv('GOOGLE_MODEL_NAME', 'gemini-2.5-flash')
            config['temperature'] = float(os.getenv('GOOGLE_TEMPERATURE', 0.7))
            
            # Validate required configuration
            if not config['api_key']:
                raise ValueError("GOOGLE_API_KEY environment variable is required for Google provider")
        
        elif provider == 'openai':
            # OpenAI-specific environment variables (future implementation)
            config['api_key'] = os.getenv('OPENAI_API_KEY')
            config['model_name'] = os.getenv('OPENAI_MODEL_NAME', 'gpt-3.5-turbo')
            config['organization'] = os.getenv('OPENAI_ORGANIZATION')
        
        # Add common configuration
        config['provider'] = provider
        
        return config
    
    @classmethod
    def get_provider_info(cls, provider: str) -> Dict[str, Any]:
        """
        Get information about a specific provider.
        
        Args:
            provider (str): The provider name
            
        Returns:
            Dict[str, Any]: Information about the provider including supported features
        """
        provider = provider.lower()
        
        if provider not in cls._providers:
            return {
                'supported': False,
                'error': f'Provider {provider} is not supported'
            }
        
        # Get the class and create a temporary instance to get info
        llm_class = cls._providers[provider]
        
        # Return provider information
        return {
            'supported': True,
            'provider': provider,
            'class': llm_class.__name__,
            'required_config': cls._get_required_config(provider),
            'default_config': cls._get_default_config(provider)
        }
    
    @classmethod
    def _get_required_config(cls, provider: str) -> list:
        """
        Get list of required configuration keys for a provider.
        
        Args:
            provider (str): The provider name
            
        Returns:
            list: List of required configuration keys
        """
        if provider == 'google':
            return ['api_key', 'model_name']
        elif provider == 'openai':
            return ['api_key']
        else:
            return []
    
    @classmethod
    def _get_default_config(cls, provider: str) -> Dict[str, Any]:
        """
        Get default configuration for a provider.
        
        Args:
            provider (str): The provider name
            
        Returns:
            Dict[str, Any]: Default configuration
        """
        if provider == 'google':
            return {
                'model_name': 'gemini-pro',
                'temperature': 0.7,
                'max_tokens': 1000
            }
        elif provider == 'openai':
            return {
                'model_name': 'gpt-3.5-turbo',
                'temperature': 0.7,
                'max_tokens': 1000
            }
        else:
            return {}
