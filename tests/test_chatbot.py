import pytest
from rag_chatbot.chatbot import Chatbot
from rag_chatbot.utils.pdf_processor import extract_text_from_pdf
from rag_chatbot.utils.vector_store import VectorStore
from unittest.mock import patch, MagicMock
import numpy as np
import os
import tempfile
from reportlab.pdfgen import canvas
import warnings

def create_test_pdf(path):
    """Create a test PDF file with some content."""
    c = canvas.Canvas(path)
    c.drawString(100, 750, "Sample text")
    c.save()

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_chatbot_initialization():
    """Test chatbot initialization."""
    chatbot = Chatbot()
    assert isinstance(chatbot.vector_store, VectorStore)

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_extract_text_from_pdf():
    """Test PDF text extraction functionality."""
    # Create a temporary PDF file
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
        tmp_path = tmp.name
        create_test_pdf(tmp_path)
    
    try:
        # Test extraction
        text = extract_text_from_pdf(tmp_path)
        assert isinstance(text, list)
        assert len(text) > 0
        assert any("Sample text" in t for t in text)
    finally:
        # Clean up
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_vector_store_embeddings():
    """Test vector store embedding creation."""
    # Mock Ollama client with correct embedding size
    mock_embeddings = np.zeros((1, 384))  # Create zero vector with correct size
    with patch('rag_chatbot.models.ollama_client.get_ollama_client') as mock_client:
        mock_client.return_value.embeddings.return_value = {'embedding': mock_embeddings[0].tolist()}
        
        vector_store = VectorStore()
        embeddings = vector_store.create_embeddings(["test chunk"])
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (1, 384)  # Updated to match actual embedding size

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_vector_store_search():
    """Test vector store search functionality."""
    # Create test data
    test_chunks = ["test chunk 1", "test chunk 2"]
    test_embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    
    with patch('rag_chatbot.models.ollama_client.get_ollama_client') as mock_client:
        mock_client.return_value.embeddings.return_value = {'embedding': test_embeddings[0].tolist()}
        
        vector_store = VectorStore()
        vector_store.create_embeddings(test_chunks)
        
        # Test search
        results, scores = vector_store.search("test query", k=2)
        assert len(results) == 2
        assert len(scores) == 2
        assert all(isinstance(score, float) for score in scores)

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_vector_store_persistence():
    """Test vector store save and load functionality."""
    # Create test data
    test_chunks = ["test chunk"]
    mock_embeddings = np.zeros((1, 384))  # Create zero vector with correct size
    
    with patch('rag_chatbot.models.ollama_client.get_ollama_client') as mock_client:
        mock_client.return_value.embeddings.return_value = {'embedding': mock_embeddings[0].tolist()}
        
        # Test save
        vector_store = VectorStore()
        vector_store.create_embeddings(test_chunks)
        vector_store.save_vector_store("test_store")
        
        # Test load
        loaded_store = VectorStore()
        loaded_store.load_vector_store("test_store")
        
        # Verify loaded data with k=1
        results, _ = loaded_store.search("test query", k=1)
        assert len(results) == 1

        # Clean up
        if os.path.exists("test_store"):
            os.remove("test_store") 