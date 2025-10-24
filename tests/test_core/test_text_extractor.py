import pytest
from unittest.mock import patch, MagicMock
from core.text_extractor import extract_text_from_pdf

import warnings

def mock_page(text):
    """Helper to create a mock page with get_text()"""
    page = MagicMock()
    page.get_text.return_value = text
    return page

def test_extract_text_from_valid_file_path():
    # Ignore deprication warning
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    # Mock document with 2 pages
    mock_doc = MagicMock()
    mock_doc.__len__.return_value = 2
    mock_doc.__getitem__.side_effect = [mock_page("Page 1 text"), mock_page("Page 2 text")]

    # Patch fitz.open to return the mock document
    with patch("fitz.open", return_value=mock_doc):
        result = extract_text_from_pdf("dummy.pdf")
        assert "Page 1 text" in result
        assert "Page 2 text" in result

def test_file_not_found():
    non_existent_file = "this_file_does_not_exist.pdf"
    result = extract_text_from_pdf(non_existent_file)
    assert result == ""