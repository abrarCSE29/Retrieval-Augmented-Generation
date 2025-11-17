from fastapi import APIRouter, File, UploadFile, HTTPException
from core.text_extractor import extract_text_from_pdf
from core.convert_to_chunks import convert_text_into_chunks
from core.generate_embedding_on_chunks import generate_embedding
from core.store_vector_embeddings import store_embedding_vector
from utils.api_response_format import create_api_response
from utils.logger import get_logger
import uuid

logger = get_logger(__name__)

documents_ep = APIRouter(prefix="/documents", tags=["documents"])

@documents_ep.post("")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF document, process it, and store it in the vector database.

    Args:
        file: The PDF file to upload

    Returns:
        JSON response with status and document_id
    """
    logger.info(f"Processing document upload: {file.filename}")

    if not file.filename.lower().endswith('.pdf'):
        logger.warning(f"Invalid file type uploaded: {file.filename}")
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        # Extract text from PDF
        logger.debug("Extracting text from PDF")
        pdf_text = extract_text_from_pdf(file.file)
        if not pdf_text.strip():
            logger.error("No text extracted from PDF")
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")

        # Generate unique document ID
        document_id = str(uuid.uuid4())
        logger.info(f"Generated document ID: {document_id}")

        # Chunk the text
        chunks = convert_text_into_chunks(pdf_text)

        # Generate embeddings
        embeddings = generate_embedding(chunks)

        # Store in vector database
        store_embedding_vector(
            embedding_vec=embeddings,
            document_id=document_id,
            chunk_texts=chunks
        )

        logger.info(f"Successfully processed document {document_id} with {len(chunks)} chunks")
        return create_api_response(
            status="success",
            message="Document processed and stored successfully",
            document_id=document_id,
            chunks_count=len(chunks)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing document: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
