"""
Base LLM class for abstracting different LLM providers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
from utils.logger import get_logger

logger = get_logger(__name__)


class BaseLLM(ABC):
    """
    Abstract base class for all LLM providers.
    
    This class defines the interface that all LLM implementations must follow,
    ensuring consistent behavior across different providers (Google, OpenAI, etc.).
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the base LLM with configuration.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary containing provider-specific settings
        """
        self.config = config
        self._validate_config()
        logger.info(f"Initialized {self.__class__.__name__} with configuration")
    
    @abstractmethod
    def _validate_config(self) -> None:
        """
        Validate provider-specific configuration.
        
        Raises:
            ValueError: If required configuration is missing or invalid
        """
        pass
    
    @abstractmethod
    async def generate_response(
        self, 
        prompt: str, 
        context: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt (str): The main prompt to send to the LLM
            context (Optional[str]): Additional context to include with the prompt
            **kwargs: Additional provider-specific parameters
            
        Returns:
            str: The generated response from the LLM
            
        Raises:
            Exception: If the LLM call fails
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model configuration.
        
        Returns:
            Dict[str, Any]: Model information including name, provider, and configuration
        """
        pass
    
    def _build_full_prompt(self, prompt: str, context: Optional[str] = None) -> str:
        """
        Build the full prompt by combining context and main prompt.
        
        Args:
            prompt (str): The main prompt
            context (Optional[str]): Additional context
            
        Returns:
            str: The combined prompt
        """
        if context and context.strip():
            return f"Context: {context}\n\nPrompt: {prompt}"
        return prompt
    
    def _get_default_generation_config(self) -> Dict[str, Any]:
        """
        Get default generation configuration.
        
        Returns:
            Dict[str, Any]: Default generation parameters
        """
        return {
            'temperature': 0.7,
            'max_tokens': 1000,
            'top_p': 1.0,
            'top_k': 32
        }
    
    def _merge_generation_config(self, **kwargs) -> Dict[str, Any]:
        """
        Merge user-provided generation config with defaults.
        
        Args:
            **kwargs: User-provided generation parameters
            
        Returns:
            Dict[str, Any]: Merged generation configuration
        """
        default_config = self._get_default_generation_config()
        # Override defaults with user-provided values
        for key, value in kwargs.items():
            if key in default_config:
                default_config[key] = value
        return default_config
    
    async def health_check(self) -> bool:
        """
        Perform a health check on the LLM provider.
        
        Returns:
            bool: True if the LLM is accessible and working, False otherwise
        """
        try:
            # Try a simple generation to test connectivity
            test_response = await self.generate_response(
                "Hello, this is a test.",
                max_tokens=10
            )
            return bool(test_response and len(test_response.strip()) > 0)
        except Exception as e:
            logger.error(f"Health check failed for {self.__class__.__name__}: {e}")
            return False
