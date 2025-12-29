from core.retrieve_vector_embeddings import retrive_related_vector_embedding
from core.generate_response import ResponseGenerator
from utils.logger import get_logger
import os

logger = get_logger(__name__)

# Initialize response generator with configuration from environment
response_generator = ResponseGenerator(
    llm_provider=os.getenv('LLM_PROVIDER', 'google'),
    llm_config={
        'api_key': os.getenv('GOOGLE_API_KEY'),
        'model_name': os.getenv('GOOGLE_MODEL_NAME', 'gemini-pro'),
        'temperature': float(os.getenv('GOOGLE_TEMPERATURE', 0.7))
    }
)


async def execute_query(user_query: str, user_id: str = None):
    """
    Execute query with enhanced response generation using LLM.
    
    Retrieves relevant context from vector database and generates a coherent
    answer using the configured LLM provider.

    Args:
        user_query (str): The user's query string
        user_id (str, optional): User identifier if needed for filtering

    Returns:
        Dict[str, Any]: Enhanced response with answer, context, and metadata
    """
    try:
        logger.info(f"Processing query: '{user_query[:100]}...'" + (f" for user {user_id}" if user_id else ""))
        
        # Retrieve relevant context from vector database
        context = retrive_related_vector_embedding(user_query=user_query)
        logger.info(f"Retrieved {len(context) if context else 0} characters of context")
        
        # Generate enhanced response using LLM
        response = await response_generator.generate_answer(
            question=user_query,
            context=context,
            temperature=float(os.getenv('GOOGLE_TEMPERATURE', 0.7))
        )
        
        # Return enhanced response with both context and generated answer
        return {
            "status": "success",
            "message": "Query processed successfully",
            "context": context,
            "answer": response["answer"],
            "model_used": response["model_used"],
            "context_used": response["context_used"],
            "user_id": user_id
        }
        
    except Exception as e:
        logger.error(f"Error executing query: {e}", exc_info=True)
        return {
            "status": "error",
            "message": f"Error processing query: {str(e)}",
            "context": "",
            "answer": "",
            "model_used": "",
            "context_used": False,
            "user_id": user_id
        }


def execute_query_sync(user_query: str, user_id: str = None):
    """
    Synchronous wrapper for execute_query to maintain backward compatibility.
    
    This function can be used by existing code that expects synchronous behavior.

    Args:
        user_query (str): The user's query string
        user_id (str, optional): User identifier if needed for filtering

    Returns:
        Dict[str, Any]: Enhanced response with answer, context, and metadata
    """
    import asyncio
    
    # Run the async function synchronously
    return asyncio.run(execute_query(user_query, user_id))
