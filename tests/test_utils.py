import pytest
from rag_chatbot.utils.text_processor import split_text
from rag_chatbot.utils.pdf_processor import extract_text_from_pdf, validate_pdf_file
from rag_chatbot.utils.vector_store import VectorStore
from rag_chatbot.utils.logging_config import get_logger
from unittest.mock import patch, MagicMock
import numpy as np
import os
import tempfile
from reportlab.pdfgen import canvas
from io import BytesIO
import warnings

# Initialize logger
logger = get_logger(__name__)

def create_test_pdf(path):
    """Create a test PDF file with some content."""
    c = canvas.Canvas(path)
    c.drawString(100, 750, "Sample text")
    c.save()

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_split_text():
    """Test text splitting functionality."""
    # Test single string input
    text = "This is a test sentence. This is another sentence."
    chunks = split_text(text)
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    assert all(isinstance(chunk, str) for chunk in chunks)

    # Test list input
    texts = ["First paragraph.", "Second paragraph."]
    chunks = split_text(texts)
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    assert all(isinstance(chunk, str) for chunk in chunks)

    # Test empty input
    empty_result = split_text("")
    assert isinstance(empty_result, list)
    assert len(empty_result) == 0

    empty_list_result = split_text([])
    assert isinstance(empty_list_result, list)
    assert len(empty_list_result) == 0

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_extract_text_from_pdf():
    """Test PDF text extraction."""
    # Create a temporary PDF file
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
        tmp_path = tmp.name
        create_test_pdf(tmp_path)
    
    try:
        # Test extraction
        texts = extract_text_from_pdf(tmp_path)
        assert isinstance(texts, list)
        assert len(texts) > 0
        assert any("Sample text" in text for text in texts)
    finally:
        # Clean up
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_validate_pdf_file():
    """Test PDF file validation."""
    # Create a temporary PDF file
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
        tmp_path = tmp.name
        create_test_pdf(tmp_path)
    
    try:
        # Test valid file
        mock_file = MagicMock()
        mock_file.name = tmp_path
        mock_file.size = 5 * 1024 * 1024  # 5MB
        
        is_valid, message = validate_pdf_file(mock_file)
        assert is_valid
        assert message == ""

        # Test invalid file type
        mock_file.name = "test.txt"
        is_valid, message = validate_pdf_file(mock_file)
        assert not is_valid
        assert "Only PDF files are supported" in message

        # Test file size limit
        mock_file.name = tmp_path
        mock_file.size = 11 * 1024 * 1024  # 11MB
        is_valid, message = validate_pdf_file(mock_file)
        assert not is_valid
        assert "File size exceeds" in message
    finally:
        # Clean up
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_vector_store_operations():
    """Test vector store operations."""
    # Mock Ollama client with correct embedding size
    mock_embeddings = np.zeros((1, 384))  # Create zero vector with correct size
    with patch('rag_chatbot.models.ollama_client.get_ollama_client') as mock_client:
        mock_client.return_value.embeddings.return_value = {'embedding': mock_embeddings[0].tolist()}
        
        # Test embedding creation
        vector_store = VectorStore()
        embeddings = vector_store.create_embeddings(["test chunk"])
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (1, 384)  # Updated to match actual embedding size

        # Test search with k=1
        results, scores = vector_store.search("test query", k=1)
        assert len(results) == 1
        assert len(scores) == 1
        assert all(isinstance(score, float) for score in scores)

        # Test save and load
        vector_store.save_vector_store("test_store")
        loaded_store = VectorStore()
        loaded_store.load_vector_store("test_store")
        
        # Verify loaded data with k=1
        results, _ = loaded_store.search("test query", k=1)
        assert len(results) == 1

        # Clean up
        if os.path.exists("test_store"):
            os.remove("test_store") 