"""
Tests for LLM Factory functionality.
"""

import pytest
from unittest.mock import patch, MagicMock
from llm.llm_factory import LLMFactory
from llm.google_llm import GoogleGenerativeAI
from utils.logger import get_logger

logger = get_logger(__name__)


class TestLLMFactory:
    """Test cases for LLMFactory class."""
    
    def test_get_supported_providers(self):
        """Test that supported providers are returned correctly."""
        providers = LLMFactory.get_supported_providers()
        assert 'google' in providers
        assert isinstance(providers, list)
    
    def test_is_provider_supported(self):
        """Test provider support checking."""
        assert LLMFactory.is_provider_supported('google') is True
        assert LLMFactory.is_provider_supported('GOOGLE') is True  # Case insensitive
        assert LLMFactory.is_provider_supported('nonexistent') is False
    
    def test_create_google_llm(self):
        """Test creating Google LLM instance."""
        config = {
            'api_key': 'test-api-key',
            'model_name': 'gemini-pro'
        }
        
        with patch('google.generativeai.configure'):
            llm = LLMFactory.create_llm('google', config)
            assert isinstance(llm, GoogleGenerativeAI)
            assert llm.config['api_key'] == 'test-api-key'
            assert llm.config['model_name'] == 'gemini-pro'
    
    def test_create_unsupported_provider(self):
        """Test error when creating unsupported provider."""
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            LLMFactory.create_llm('unsupported', {})
    
    def test_create_llm_invalid_config(self):
        """Test error when configuration is invalid."""
        with pytest.raises(ValueError, match="Missing required Google AI config"):
            LLMFactory.create_llm('google', {'model_name': 'gemini-pro'})  # Missing api_key
    
    @patch('os.getenv')
    def test_create_from_env_google(self, mock_getenv):
        """Test creating LLM from environment variables."""
        mock_getenv.side_effect = lambda key: {
            'LLM_PROVIDER': 'google',
            'GOOGLE_API_KEY': 'env-test-key',
            'GOOGLE_MODEL_NAME': 'gemini-pro',
            'GOOGLE_TEMPERATURE': '0.8'
        }.get(key)
        
        with patch('google.generativeai.configure'):
            llm = LLMFactory.create_from_env()
            assert isinstance(llm, GoogleGenerativeAI)
            assert llm.config['api_key'] == 'env-test-key'
            assert llm.config['model_name'] == 'gemini-pro'
            assert llm.config['temperature'] == 0.8
    
    def test_create_from_env_no_provider(self):
        """Test error when no provider is specified in environment."""
        with patch('os.getenv', return_value=None):
            with pytest.raises(ValueError, match="Provider must be specified"):
                LLMFactory.create_from_env()
    
    def test_get_provider_info(self):
        """Test getting provider information."""
        info = LLMFactory.get_provider_info('google')
        
        assert info['supported'] is True
        assert info['provider'] == 'google'
        assert info['class'] == 'GoogleGenerativeAI'
        assert 'api_key' in info['required_config']
        assert 'model_name' in info['required_config']
    
    def test_get_provider_info_unsupported(self):
        """Test getting info for unsupported provider."""
        info = LLMFactory.get_provider_info('unsupported')
        
        assert info['supported'] is False
        assert 'error' in info
    
    def test_get_required_config(self):
        """Test getting required configuration keys."""
        required = LLMFactory._get_required_config('google')
        assert 'api_key' in required
        assert 'model_name' in required
    
    def test_get_default_config(self):
        """Test getting default configuration."""
        default = LLMFactory._get_default_config('google')
        assert default['model_name'] == 'gemini-pro'
        assert default['temperature'] == 0.7
        assert default['max_tokens'] == 1000
