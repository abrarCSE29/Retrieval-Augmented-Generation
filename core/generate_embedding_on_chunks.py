from utils.logger import get_logger

logger = get_logger(__name__)


def generate_embedding(text_chunks: list, model_name: str = 'all-MiniLM-L6-v2'):
    """
    Generate embeddings for text chunks using sentence transformers.
    """
    from sentence_transformers import SentenceTransformer

    logger.debug(f"Generating embeddings with model '{model_name}' for {len(text_chunks)} chunks")

    # Handle empty input
    if not text_chunks:
        logger.warning("No text chunks provided for embedding")
        return []

    try:
        # Load model
        logger.debug(f"Loading sentence transformer model: {model_name}")
        model = SentenceTransformer(model_name)

        # Generate embeddings
        vector_embeddings = model.encode(
            text_chunks,
            convert_to_numpy=True,
            show_progress_bar=len(text_chunks) > 100,
            batch_size=32
        )

        logger.info(f"Generated {len(vector_embeddings)} embeddings with shape {vector_embeddings.shape}")
        return vector_embeddings

    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}", exc_info=True)
        raise
