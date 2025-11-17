from vector_db.qdrant import Qdrant_VDB
from utils.logger import get_logger

logger = get_logger(__name__)

vdb = Qdrant_VDB(
    collection_name="test_collection_384",
    vector_size=384
)

def retrive_related_vector_embedding(user_query : str):
    """
    Retrieves related chunks against user query from a qdrant vector db

    Args:
        user_query(str):Users query

    Returns:
        related_embeddings(str): concatenated text chunks related to user question
    """
    try:
        logger.info(f"Retrieving related text content for query: '{user_query[:100]}...'")
        vector_embeddings = vdb.retrieve_document_embeddings(user_query=user_query)
        logger.info(f"Retrieved {len(vector_embeddings)} characters of relevant content")
        return vector_embeddings
    except Exception as e:
        logger.error(f"Failed to retrieve embeddings for query: {e}", exc_info=True)
        raise
