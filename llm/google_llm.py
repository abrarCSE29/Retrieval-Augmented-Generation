"""
Google Generative AI implementation using LangChain for the LLM abstraction layer.
"""

import os
from typing import Dict, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .base_llm import BaseLLM
from utils.logger import get_logger

logger = get_logger(__name__)


class GoogleGenerativeAI(BaseLLM):
    """
    Google Generative AI implementation using LangChain.
    
    Uses LangChain's ChatGoogleGenerativeAI for text generation with support for
    context-aware responses, structured prompts, and advanced LangChain features.
    """
    
    def _validate_config(self) -> None:
        """
        Validate Google Generative AI specific configuration.
        
        Raises:
            ValueError: If required configuration is missing or invalid
        """
        required_keys = ['api_key', 'model_name']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required Google AI config: {key}")
        
        # Set the API key as environment variable for LangChain
        os.environ['GOOGLE_API_KEY'] = self.config['api_key']
        logger.info("Google Generative AI configured with LangChain")
    
    def _initialize_llm(self, **kwargs) -> ChatGoogleGenerativeAI:
        """
        Initialize the LangChain Google Generative AI model.
        
        Args:
            **kwargs: Additional generation parameters
            
        Returns:
            ChatGoogleGenerativeAI: Configured LLM instance
        """
        # Get generation configuration
        generation_config = self._get_generation_config(**kwargs)
        
        # Initialize LangChain LLM
        llm = ChatGoogleGenerativeAI(
            model=self.config['model_name'],
            temperature=generation_config['temperature'],
            max_output_tokens=generation_config['max_output_tokens'],
            top_p=generation_config['top_p'],
            top_k=generation_config['top_k'],
            convert_system_message_to_human=True  # Better compatibility
        )
        
        return llm
    
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """
        Create a structured prompt template using LangChain.
        
        Returns:
            ChatPromptTemplate: Configured prompt template
        """
        # Create a structured prompt template with proper variable handling
        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content="""
You are a helpful AI assistant that answers questions based on provided context.
Use the provided context to answer the question to the best of your ability.
If the context contains relevant information, incorporate it into your response.
If the context is insufficient, provide the best answer possible based on the available information.
Keep answers concise and to the point.
Do not make up information not present in the context.
            """),
            MessagesPlaceholder(variable_name="history"),  # For conversation history
            ("human", "Context: {context}\n\nQuestion: {question}")
        ])
        
        return prompt_template
    
    async def generate_response(
        self, 
        prompt: str, 
        context: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate a response using LangChain's Google Generative AI.
        
        Args:
            prompt (str): The main prompt to send to the LLM
            context (Optional[str]): Additional context to include with the prompt
            **kwargs: Additional generation parameters (temperature, max_tokens, etc.)
            
        Returns:
            str: The generated response from Google AI
            
        Raises:
            Exception: If the Google AI call fails
        """
        try:
            # Initialize LLM
            llm = self._initialize_llm(**kwargs)
            
            # Create prompt template
            prompt_template = self._create_prompt_template()
            
            # Create output parser
            output_parser = StrOutputParser()
            
            # Build the chain
            chain = prompt_template | llm | output_parser
            
            # Prepare context (handle empty context)
            context_text = context if context and context.strip() else "No relevant context found."
            # Generate response
            response = await chain.ainvoke({
                "context": context_text,
                "question": prompt,
                "history": []  # No conversation history for now
            })
            logger.info(f"Generated response using {self.config['model_name']}")
            return response.strip()
                
        except Exception as e:
            logger.error(f"Error generating response with Google AI: {e}", exc_info=True)
            raise Exception(f"Failed to generate response: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the Google AI model configuration.
        
        Returns:
            Dict[str, Any]: Model information including name, provider, and configuration
        """
        return {
            'provider': 'google',
            'model_name': self.config['model_name'],
            'api_key_set': bool(self.config.get('api_key')),
            'framework': 'langchain',
            'supported_features': ['text_generation', 'context_aware', 'structured_prompts', 'streaming']
        }
    
    def _get_generation_config(self, **kwargs) -> Dict[str, Any]:
        """
        Get generation configuration for Google AI.
        
        Args:
            **kwargs: User-provided generation parameters
            
        Returns:
            Dict[str, Any]: Generation configuration for Google AI
        """
        # Default configuration
        config = {
            'temperature': kwargs.get('temperature', 0.7),
            'max_output_tokens': kwargs.get('max_tokens', 1000),
            'top_p': kwargs.get('top_p', 1.0),
            'top_k': kwargs.get('top_k', 32)
        }
        
        # Filter out None values and convert to appropriate types
        filtered_config = {}
        for key, value in config.items():
            if value is not None:
                filtered_config[key] = value
        
        return filtered_config
    
    def get_supported_models(self) -> list:
        """
        Get list of supported Google AI models.
        
        Returns:
            list: List of supported model names
        """
        return [
            'gemini-pro',
            'gemini-pro-vision',
            'gemini-ultra'
        ]
    
    def get_prompt_template(self) -> ChatPromptTemplate:
        """
        Get the prompt template for inspection or customization.
        
        Returns:
            ChatPromptTemplate: The configured prompt template
        """
        return self._create_prompt_template()
    
    def create_streaming_chain(self, **kwargs):
        """
        Create a streaming chain for real-time response generation.
        
        Args:
            **kwargs: Generation parameters
            
        Returns:
            Runnable: Streaming chain
        """
        llm = self._initialize_llm(**kwargs)
        prompt_template = self._create_prompt_template()
        output_parser = StrOutputParser()
        
        return prompt_template | llm | output_parser
    
    def get_llm_instance(self, **kwargs) -> ChatGoogleGenerativeAI:
        """
        Get the raw LLM instance for advanced usage.
        
        Args:
            **kwargs: Generation parameters
            
        Returns:
            ChatGoogleGenerativeAI: The LLM instance
        """
        return self._initialize_llm(**kwargs)
