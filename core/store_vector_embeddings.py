from vector_db.qdrant import Qdrant_VDB
from utils.logger import get_logger

logger = get_logger(__name__)

vdb = Qdrant_VDB(
    collection_name="test_collection_384",
    vector_size=384
)


def store_embedding_vector(embedding_vec : list[list[float]], document_id : str, chunk_texts : list[str]):
    """
    Store embedding vectors for a document in the vector database.

    Args:
        embedding_vec: List of embedding vectors
        document_id: Unique identifier for the document
        chunk_texts: Corresponding text chunks
    """
    try:
        logger.info(f"Storing embeddings for document {document_id} with {len(chunk_texts)} chunks")
        result = vdb.store_document_embeddings(
            document_id=document_id,
            embeddings=embedding_vec,
            chunk_texts=chunk_texts
        )
        logger.info(f"Successfully stored document {document_id}")
        return result
    except Exception as e:
        logger.error(f"Failed to store embeddings for document {document_id}: {e}", exc_info=True)
        raise
