"""
Response generator for RAG system using LLM providers.
"""

import os
from typing import Dict, Any, Optional
from llm.llm_factory import LLMFactory
from prompts.rag_prompts import RAGPrompts
from utils.logger import get_logger

logger = get_logger(__name__)


class ResponseGenerator:
    """
    Generate responses using LLM with retrieved context from vector database.
    
    This class orchestrates the response generation process by:
    1. Using the LLM factory to create the appropriate LLM instance
    2. Formatting prompts with context using RAG prompt templates
    3. Generating responses from the LLM
    4. Handling errors and fallback scenarios
    """
    
    def __init__(
        self, 
        llm_provider: str = None, 
        llm_config: Dict[str, Any] = None,
        prompt_template: str = None
    ):
        """
        Initialize the response generator.
        
        Args:
            llm_provider (str, optional): LLM provider to use (e.g., 'google', 'openai')
            llm_config (Dict[str, Any], optional): Configuration for the LLM provider
            prompt_template (str, optional): Custom prompt template to use
        """
        # Determine provider from parameter or environment
        if llm_provider is None:
            llm_provider = os.getenv('LLM_PROVIDER', 'google')
        
        # Build configuration
        if llm_config is None:
            llm_config = self._build_llm_config(llm_provider)
        
        # Create LLM instance
        self.llm = LLMFactory.create_llm(llm_provider, llm_config)
        
        # Set prompt template
        self.prompt_template = prompt_template or RAGPrompts.get_rag_prompt_template()
        
        logger.info(f"Initialized ResponseGenerator with provider: {llm_provider}")
    
    async def generate_answer(
        self, 
        question: str, 
        context: str = None,
        **generation_kwargs
    ) -> Dict[str, Any]:
        """
        Generate an answer using the LLM and retrieved context.
        
        Args:
            question (str): The user's question
            context (str, optional): Retrieved context from vector database
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Dict[str, Any]: Response containing answer, model info, and metadata
        """
        try:
            # Handle case when no context is found
            if not context or context.strip() == "":
                return self._create_fallback_response(question)
            
            # Generate response using LLM (LangChain handles prompt formatting)
            answer = await self.llm.generate_response(
                prompt=question,
                context=context,
                **generation_kwargs
            )
            
            # Return structured response
            return {
                "answer": answer,
                "model_used": self.llm.get_model_info()['model_name'],
                "framework": self.llm.get_model_info()['framework'],
                "context_used": True,
                "context_length": len(context),
                "question_length": len(question),
                "generation_params": self._extract_generation_params(generation_kwargs)
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            return self._create_error_response(str(e))
    
    def _format_prompt(self, question: str, context: str) -> str:
        """
        Format the prompt using the RAG prompt template.
        
        Args:
            question (str): The user's question
            context (str): Retrieved context from vector database
            
        Returns:
            str: Formatted prompt for the LLM
        """
        return self.prompt_template.format(
            context=context,
            question=question
        )
    
    def _create_fallback_response(self, question: str) -> Dict[str, Any]:
        """
        Create a fallback response when no context is found.
        
        Args:
            question (str): The user's question
            
        Returns:
            Dict[str, Any]: Fallback response
        """
        logger.info(f"No context found for question: {question[:50]}...")
        
        return {
            "answer": "I don't have enough context to answer this question. Please provide more information or check if the relevant documents have been uploaded to the system.",
            "model_used": self.llm.get_model_info()['model_name'] if hasattr(self.llm, 'get_model_info') else "unknown",
            "context_used": False,
            "context_length": 0,
            "question_length": len(question)
        }
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """
        Create an error response when LLM generation fails.
        
        Args:
            error_message (str): The error message
            
        Returns:
            Dict[str, Any]: Error response
        """
        return {
            "answer": "Sorry, I encountered an error while generating a response. Please try again later.",
            "model_used": self.llm.get_model_info()['model_name'] if hasattr(self.llm, 'get_model_info') else "unknown",
            "context_used": False,
            "error": error_message,
            "context_length": 0,
            "question_length": 0
        }
    
    def _build_llm_config(self, provider: str) -> Dict[str, Any]:
        """
        Build LLM configuration from environment variables.
        
        Args:
            provider (str): The LLM provider
            
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        config = {}
        
        if provider == 'google':
            config['api_key'] = os.getenv('GOOGLE_API_KEY')
            config['model_name'] = os.getenv('GOOGLE_MODEL_NAME', 'gemini-pro')
            config['temperature'] = float(os.getenv('GOOGLE_TEMPERATURE', 0.7))
            
            if not config['api_key']:
                raise ValueError("GOOGLE_API_KEY environment variable is required for Google provider")
        
        elif provider == 'openai':
            config['api_key'] = os.getenv('OPENAI_API_KEY')
            config['model_name'] = os.getenv('OPENAI_MODEL_NAME', 'gpt-3.5-turbo')
            config['organization'] = os.getenv('OPENAI_ORGANIZATION')
        
        return config
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the response generator and underlying LLM.
        
        Returns:
            Dict[str, Any]: Health check results
        """
        try:
            # Check if LLM is healthy
            llm_healthy = await self.llm.health_check()
            
            return {
                "status": "healthy" if llm_healthy else "unhealthy",
                "llm_provider": self.llm.get_model_info()['provider'],
                "llm_model": self.llm.get_model_info()['model_name'],
                "llm_healthy": llm_healthy,
                "prompt_template_set": bool(self.prompt_template)
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "llm_provider": "unknown",
                "llm_model": "unknown",
                "llm_healthy": False,
                "prompt_template_set": bool(self.prompt_template)
            }
    
    def get_provider_info(self) -> Dict[str, Any]:
        """
        Get information about the current LLM provider.
        
        Returns:
            Dict[str, Any]: Provider information
        """
        return self.llm.get_model_info()
    
    def update_prompt_template(self, new_template: str):
        """
        Update the prompt template used for response generation.
        
        Args:
            new_template (str): New prompt template
        """
        self.prompt_template = new_template
        logger.info("Updated prompt template")
    
    def _extract_generation_params(self, generation_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract generation parameters for logging and metadata.
        
        Args:
            generation_kwargs: Generation parameters
            
        Returns:
            Dict[str, Any]: Extracted parameters
        """
        return {
            'temperature': generation_kwargs.get('temperature', 0.7),
            'max_tokens': generation_kwargs.get('max_tokens', 1000),
            'top_p': generation_kwargs.get('top_p', 1.0),
            'top_k': generation_kwargs.get('top_k', 32)
        }
