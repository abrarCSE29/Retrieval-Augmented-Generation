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
    Process a user query and return relevant context from the RAG system.

    Args:
        request: QueryRequest with query string and optional user_id

    Returns:
        JSON response with status and relevant text content
    """
    logger.info(f"Processing query: '{request.query[:100]}...'" + (f" for user {request.user_id}" if request.user_id else ""))

    try:
        if not request.query.strip():
            logger.warning("Empty query received")
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        # Execute the RAG query
        logger.debug("Executing RAG query")
        relevant_content = execute_query(
            user_query=request.query,
            user_id=request.user_id
        )

        logger.info(f"Query processed, returned {len(relevant_content)} characters of context")
        return create_api_response(
            status="success",
            message="Query processed successfully",
            context=relevant_content
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
