import re

def convert_text_into_chunks(text: str, chunk_len: int = 500, overlap: int = 100) -> list:
    """
    Convert input text into overlapping chunks for embedding or RAG purposes.

    Args:
        texts (str): The text to be converted into chunks.
        chunk_len (int): Maximum number of characters per chunk.
        overlap (int): Number of overlapping characters between consecutive chunks.

    Returns:
        list[str]: A list of text chunks.
    """
    # Normalize spaces and clean text
    text = re.sub(r'\s+', ' ', text).strip()

    if len(text) <= chunk_len:
        return [text]

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_len

        # Avoid cutting mid-sentence if possible
        if end < text_length:
            next_period = text.rfind('.', start, end)
            if next_period != -1 and next_period > start + chunk_len * 0.5:
                end = next_period + 1

        chunks.append(text[start:end].strip())
        start = end - overlap  # create overlap for context

    return chunks
