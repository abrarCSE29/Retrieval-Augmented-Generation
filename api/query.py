from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from core.execute_query import execute_query
from utils.api_response_format import create_api_response
from utils.logger import get_logger

logger = get_logger(__name__)

query_ep = APIRouter(prefix="/query", tags=["query"])

class QueryRequest(BaseModel):
    query: str
    user_id: str = None

@query_ep.post("")
async def process_query(request: QueryRequest):
    """
    Process a user query and return enhanced response with LLM-generated answer.

    Args:
        request: QueryRequest with query string and optional user_id

    Returns:
        JSON response with status, context, generated answer, and metadata
    """
    logger.info(f"Processing query: '{request.query[:100]}...'" + (f" for user {request.user_id}" if request.user_id else ""))

    try:
        if not request.query.strip():
            logger.warning("Empty query received")
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        # Execute the enhanced RAG query with LLM response generation
        logger.debug("Executing enhanced RAG query with LLM")
        response_data = await execute_query(
            user_query=request.query,
            user_id=request.user_id
        )

        # Check if the query execution was successful
        if response_data.get("status") == "error":
            logger.error(f"Query execution failed: {response_data.get('message')}")
            raise HTTPException(status_code=500, detail=response_data.get("message"))

        # Return the enhanced response with all fields
        return create_api_response(
            status=response_data["status"],
            message=response_data["message"],
            context=response_data["context"],
            answer=response_data["answer"],
            model_used=response_data["model_used"],
            context_used=response_data["context_used"],
            user_id=response_data["user_id"]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
