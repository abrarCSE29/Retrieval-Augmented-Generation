from core.retrieve_vector_embeddings import retrive_related_vector_embedding

def execute_query(user_query : str ,user_id : str = None ):
    """
    Executes query on user question by retrieving necessary text_chunks from vector db

    Args:
        user_query (str): The user's query string
        user_id (str, optional): User identifier if needed for filtering

    Returns:
        str: Concatenated relevant text chunks from the vector database
    """
    related_text_chunks = retrive_related_vector_embedding(user_query=user_query)
    return related_text_chunks
