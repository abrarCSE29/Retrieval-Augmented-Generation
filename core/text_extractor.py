


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
        # Open PDF
        if isinstance(pdf_file, str):
            doc = fitz.open(pdf_file)
        else:
            # For file objects, read bytes and open
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        
        # Extract text from each page
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if text:
                extracted_text += text + "\n"
        
        doc.close()
    
    except FileNotFoundError:
        print(f"Error: File not found at path: {pdf_file}")
        return ""
    except Exception as e:
        print(f"Error extracting text from PDF: {str(e)}")
        return ""
    
    return extracted_text.strip()