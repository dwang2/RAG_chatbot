from typing import List, Tuple, Optional, Any, Union
import os
from pathlib import Path
from pypdf import PdfReader
from rag_chatbot.utils.logging_config import get_logger

# Initialize logger
logger = get_logger(__name__)

def validate_pdf_file(file: Any, max_size_mb: int = 10) -> Tuple[bool, str]:
    """Validate the uploaded PDF file."""
    try:
        if file is None:
            return False, "No file uploaded"
        
        if file.size > max_size_mb * 1024 * 1024:
            return False, f"File size exceeds {max_size_mb}MB limit"
        
        if not file.name.lower().endswith('.pdf'):
            return False, "Only PDF files are supported"
        
        return True, ""
    except Exception as e:
        logger.error(f"Error validating PDF file: {str(e)}")
        return False, str(e)

def extract_text_from_pdf(pdf_path: str) -> List[str]:
    """
    Extract text from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of text strings, one per page
    """
    try:
        logger.info(f"Extracting text from PDF: {pdf_path}")
        reader = PdfReader(pdf_path)
        texts = []
        
        for page in reader.pages:
            text = page.extract_text()
            if text.strip():  # Only add non-empty pages
                texts.append(text)
        
        logger.info(f"Extracted {len(texts)} pages from PDF")
        return texts
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise 