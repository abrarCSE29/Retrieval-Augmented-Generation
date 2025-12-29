"""
Tests for enhanced query processing with LLM integration.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from core.execute_query import execute_query, execute_query_sync
from utils.logger import get_logger

logger = get_logger(__name__)


class TestEnhancedQuery:
    """Test cases for enhanced query processing."""
    
    @patch('core.execute_query.response_generator')
    @patch('core.execute_query.retrive_related_vector_embedding')
    async def test_execute_query_with_context(self, mock_retrieve, mock_response_gen):
        """Test query execution with retrieved context."""
        # Mock the vector retrieval
        mock_retrieve.return_value = "Retrieved context from vector database"
        
        # Mock the response generator
        mock_response_gen.generate_answer = AsyncMock(return_value={
            "answer": "Generated answer based on context",
            "model_used": "gemini-pro",
            "context_used": True
        })
        
        # Execute the query
        result = await execute_query("What is AI?")
        
        # Verify the result
        assert result['status'] == 'success'
        assert result['message'] == 'Query processed successfully'
        assert result['context'] == "Retrieved context from vector database"
        assert result['answer'] == "Generated answer based on context"
        assert result['model_used'] == 'gemini-pro'
        assert result['context_used'] is True
        
        # Verify mocks were called
        mock_retrieve.assert_called_once_with(user_query="What is AI?")
        mock_response_gen.generate_answer.assert_called_once_with(
            question="What is AI?",
            context="Retrieved context from vector database",
            temperature=0.7
        )
    
    @patch('core.execute_query.response_generator')
    @patch('core.execute_query.retrive_related_vector_embedding')
    async def test_execute_query_no_context(self, mock_retrieve, mock_response_gen):
        """Test query execution when no context is found."""
        # Mock the vector retrieval returning empty context
        mock_retrieve.return_value = ""
        
        # Mock the response generator
        mock_response_gen.generate_answer = AsyncMock(return_value={
            "answer": "I don't have enough context to answer this question.",
            "model_used": "gemini-pro",
            "context_used": False
        })
        
        # Execute the query
        result = await execute_query("What is AI?")
        
        # Verify the result
        assert result['status'] == 'success'
        assert result['context'] == ""
        assert "don't have enough context" in result['answer']
        assert result['context_used'] is False
    
    @patch('core.execute_query.response_generator')
    @patch('core.execute_query.retrive_related_vector_embedding')
    async def test_execute_query_with_user_id(self, mock_retrieve, mock_response_gen):
        """Test query execution with user ID."""
        mock_retrieve.return_value = "Retrieved context"
        mock_response_gen.generate_answer = AsyncMock(return_value={
            "answer": "Generated answer",
            "model_used": "gemini-pro",
            "context_used": True
        })
        
        result = await execute_query("What is AI?", user_id="user123")
        
        assert result['user_id'] == "user123"
        assert result['status'] == 'success'
    
    @patch('core.execute_query.response_generator')
    @patch('core.execute_query.retrive_related_vector_embedding')
    async def test_execute_query_llm_error(self, mock_retrieve, mock_response_gen):
        """Test query execution when LLM generation fails."""
        mock_retrieve.return_value = "Retrieved context"
        mock_response_gen.generate_answer = AsyncMock(side_effect=Exception("LLM Error"))
        
        result = await execute_query("What is AI?")
        
        assert result['status'] == 'error'
        assert "Error processing query" in result['message']
        assert result['answer'] == ""
        assert result['context_used'] is False
    
    @patch('core.execute_query.response_generator')
    @patch('core.execute_query.retrive_related_vector_embedding')
    async def test_execute_query_vector_error(self, mock_retrieve, mock_response_gen):
        """Test query execution when vector retrieval fails."""
        mock_retrieve.side_effect = Exception("Vector DB Error")
        
        result = await execute_query("What is AI?")
        
        assert result['status'] == 'error'
        assert "Error executing query" in result['message']
        assert result['context'] == ""
    
    @patch('core.execute_query.execute_query')
    def test_execute_query_sync(self, mock_async_execute):
        """Test synchronous wrapper for execute_query."""
        mock_async_execute.return_value = {
            'status': 'success',
            'message': 'Query processed successfully',
            'context': 'test context',
            'answer': 'test answer'
        }
        
        result = execute_query_sync("What is AI?")
        
        assert result['status'] == 'success'
        assert result['message'] == 'Query processed successfully'
        assert result['context'] == 'test context'
        assert result['answer'] == 'test answer'
        mock_async_execute.assert_called_once_with("What is AI?", None)
    
    @patch('core.execute_query.execute_query')
    def test_execute_query_sync_with_user_id(self, mock_async_execute):
        """Test synchronous wrapper with user ID."""
        mock_async_execute.return_value = {
            'status': 'success',
            'user_id': 'user123'
        }
        
        result = execute_query_sync("What is AI?", user_id="user123")
        
        assert result['user_id'] == 'user123'
        mock_async_execute.assert_called_once_with("What is AI?", "user123")
    
    @patch('core.execute_query.response_generator')
    @patch('core.execute_query.retrive_related_vector_embedding')
    async def test_execute_query_logging(self, mock_retrieve, mock_response_gen, caplog):
        """Test that query execution logs appropriately."""
        mock_retrieve.return_value = "Retrieved context"
        mock_response_gen.generate_answer = AsyncMock(return_value={
            "answer": "Generated answer",
            "model_used": "gemini-pro",
            "context_used": True
        })
        
        with caplog.at_level('INFO'):
            await execute_query("What is AI?", user_id="user123")
        
        # Check that appropriate logs were generated
        assert "Processing query" in caplog.text
        assert "Retrieved" in caplog.text
    
    @patch('core.execute_query.response_generator')
    @patch('core.execute_query.retrive_related_vector_embedding')
    async def test_execute_query_temperature_from_env(self, mock_retrieve, mock_response_gen):
        """Test that temperature is passed from environment variables."""
        mock_retrieve.return_value = "Retrieved context"
        mock_response_gen.generate_answer = AsyncMock(return_value={
            "answer": "Generated answer",
            "model_used": "gemini-pro",
            "context_used": True
        })
        
        with patch('core.execute_query.os.getenv', return_value='0.8'):
            await execute_query("What is AI?")
        
        # Verify that the temperature was passed correctly
        mock_response_gen.generate_answer.assert_called_once_with(
            question="What is AI?",
            context="Retrieved context",
            temperature=0.8
        )
