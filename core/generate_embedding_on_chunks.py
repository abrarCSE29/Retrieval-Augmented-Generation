def generate_embedding(text_chunks: list, model_name: str = 'all-MiniLM-L6-v2'):
    """
    Enhanced version with model selection
    """
    from sentence_transformers import SentenceTransformer
    
    # Load model
    model = SentenceTransformer(model_name)
    
    # Handle empty input
    if not text_chunks:
        return []
    
    vector_embeddings = model.encode(
        text_chunks,
        convert_to_numpy=True,
        show_progress_bar=len(text_chunks) > 100,
        batch_size=32
    )
    print(vector_embeddings.shape)
    return vector_embeddings
