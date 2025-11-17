


from utils.logger import get_logger

logger = get_logger(__name__)


def extract_text_from_pdf(pdf_file):
    """
    The function receives a pdf pdf_file path as input, extracts text and returns the extracted text.
    Args:
        pdf_file(str) : path of a pdf file
    Returns:
        extracted_text : text extracted from the pdf_file
    """
    import fitz

    extracted_text = ""

    try:
        logger.info(f"Starting text extraction from PDF: {pdf_file}")
        # Open PDF
        if isinstance(pdf_file, str):
            doc = fitz.open(pdf_file)
            logger.debug(f"Opened PDF file: {pdf_file}")
        else:
            # For file objects, read bytes and open
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
            logger.debug("Opened PDF from file stream")

        # Extract text from each page
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if text:
                extracted_text += text + "\n"

        logger.info(f"Successfully extracted {len(extracted_text)} characters from {len(doc)} pages")
        doc.close()

    except FileNotFoundError:
        logger.error(f"PDF file not found at path: {pdf_file}", exc_info=True)
        return ""
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}", exc_info=True)
        return ""

    return extracted_text.strip()
