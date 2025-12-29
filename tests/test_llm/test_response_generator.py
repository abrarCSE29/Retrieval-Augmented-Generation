"""
Tests for Response Generator functionality.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from core.generate_response import ResponseGenerator
from utils.logger import get_logger

logger = get_logger(__name__)


class TestResponseGenerator:
    """Test cases for ResponseGenerator class."""
    
    @patch('core.generate_response.LLMFactory')
    def test_init_with_default_provider(self, mock_factory):
        """Test initialization with default provider."""
        mock_llm = MagicMock()
        mock_factory.create_llm.return_value = mock_llm
        
        generator = ResponseGenerator()
        
        mock_factory.create_llm.assert_called_once()
        assert generator.llm == mock_llm
        assert generator.prompt_template is not None
    
    @patch('core.generate_response.LLMFactory')
    def test_init_with_custom_provider(self, mock_factory):
        """Test initialization with custom provider."""
        mock_llm = MagicMock()
        mock_factory.create_llm.return_value = mock_llm
        
        generator = ResponseGenerator(llm_provider='google', llm_config={'api_key': 'test'})
        
        mock_factory.create_llm.assert_called_once_with('google', {'api_key': 'test'})
    
    @patch('core.generate_response.LLMFactory')
    async def test_generate_answer_with_context(self, mock_factory):
        """Test generating answer with context."""
        mock_llm = MagicMock()
        mock_llm.generate_response = AsyncMock(return_value="Generated answer")
        mock_llm.get_model_info.return_value = {
            'model_name': 'gemini-pro',
            'framework': 'langchain'
        }
        mock_factory.create_llm.return_value = mock_llm
        
        generator = ResponseGenerator()
        result = await generator.generate_answer(
            question="What is AI?",
            context="AI is artificial intelligence..."
        )
        
        assert result['answer'] == "Generated answer"
        assert result['model_used'] == 'gemini-pro'
        assert result['framework'] == 'langchain'
        assert result['context_used'] is True
        assert result['context_length'] > 0
        assert 'generation_params' in result
    
    @patch('core.generate_response.LLMFactory')
    async def test_generate_answer_no_context(self, mock_factory):
        """Test generating answer when no context is found."""
        mock_llm = MagicMock()
        mock_llm.get_model_info.return_value = {'model_name': 'gemini-pro'}
        mock_factory.create_llm.return_value = mock_llm
        
        generator = ResponseGenerator()
        result = await generator.generate_answer(question="What is AI?")
        
        assert "don't have enough context" in result['answer']
        assert result['model_used'] == 'gemini-pro'
        assert result['context_used'] is False
        assert result['context_length'] == 0
    
    @patch('core.generate_response.LLMFactory')
    async def test_generate_answer_llm_error(self, mock_factory):
        """Test error handling when LLM generation fails."""
        mock_llm = MagicMock()
        mock_llm.generate_response = AsyncMock(side_effect=Exception("LLM Error"))
        mock_llm.get_model_info.return_value = {'model_name': 'gemini-pro'}
        mock_factory.create_llm.return_value = mock_llm
        
        generator = ResponseGenerator()
        result = await generator.generate_answer(
            question="What is AI?",
            context="AI is artificial intelligence..."
        )
        
        assert "error while generating" in result['answer']
        assert result['model_used'] == 'gemini-pro'
        assert result['context_used'] is False
        assert 'error' in result
    
    @patch('core.generate_response.LLMFactory')
    async def test_health_check_healthy(self, mock_factory):
        """Test health check when LLM is healthy."""
        mock_llm = MagicMock()
        mock_llm.health_check = AsyncMock(return_value=True)
        mock_llm.get_model_info.return_value = {
            'provider': 'google',
            'model_name': 'gemini-pro'
        }
        mock_factory.create_llm.return_value = mock_llm
        
        generator = ResponseGenerator()
        result = await generator.health_check()
        
        assert result['status'] == 'healthy'
        assert result['llm_provider'] == 'google'
        assert result['llm_model'] == 'gemini-pro'
        assert result['llm_healthy'] is True
    
    @patch('core.generate_response.LLMFactory')
    async def test_health_check_unhealthy(self, mock_factory):
        """Test health check when LLM is unhealthy."""
        mock_llm = MagicMock()
        mock_llm.health_check = AsyncMock(return_value=False)
        mock_llm.get_model_info.return_value = {
            'provider': 'google',
            'model_name': 'gemini-pro'
        }
        mock_factory.create_llm.return_value = mock_llm
        
        generator = ResponseGenerator()
        result = await generator.health_check()
        
        assert result['status'] == 'unhealthy'
        assert result['llm_healthy'] is False
    
    @patch('core.generate_response.LLMFactory')
    async def test_health_check_exception(self, mock_factory):
        """Test health check when exception occurs."""
        mock_factory.create_llm.side_effect = Exception("Factory Error")
        
        generator = ResponseGenerator()
        result = await generator.health_check()
        
        assert result['status'] == 'unhealthy'
        assert 'Factory Error' in result['error']
    
    @patch('core.generate_response.LLMFactory')
    def test_get_provider_info(self, mock_factory):
        """Test getting provider information."""
        mock_llm = MagicMock()
        mock_llm.get_model_info.return_value = {
            'provider': 'google',
            'model_name': 'gemini-pro'
        }
        mock_factory.create_llm.return_value = mock_llm
        
        generator = ResponseGenerator()
        info = generator.get_provider_info()
        
        assert info['provider'] == 'google'
        assert info['model_name'] == 'gemini-pro'
    
    @patch('core.generate_response.LLMFactory')
    def test_update_prompt_template(self, mock_factory):
        """Test updating prompt template."""
        mock_llm = MagicMock()
        mock_factory.create_llm.return_value = mock_llm
        
        generator = ResponseGenerator()
        new_template = "Custom template: {context} {question}"
        
        generator.update_prompt_template(new_template)
        
        assert generator.prompt_template == new_template
    
    @patch('core.generate_response.LLMFactory')
    def test_build_llm_config_google(self, mock_factory):
        """Test building Google LLM configuration from environment."""
        mock_llm = MagicMock()
        mock_factory.create_llm.return_value = mock_llm
        
        with patch('core.generate_response.os.getenv') as mock_getenv:
            mock_getenv.side_effect = lambda key: {
                'GOOGLE_API_KEY': 'test-key',
                'GOOGLE_MODEL_NAME': 'gemini-pro',
                'GOOGLE_TEMPERATURE': '0.8'
            }.get(key)
            
            generator = ResponseGenerator(llm_provider='google')
            
            # Verify the configuration was built correctly
            mock_factory.create_llm.assert_called_once()
            call_args = mock_factory.create_llm.call_args
            config = call_args[0][1]  # Second positional argument
            
            assert config['api_key'] == 'test-key'
            assert config['model_name'] == 'gemini-pro'
            assert config['temperature'] == 0.8
