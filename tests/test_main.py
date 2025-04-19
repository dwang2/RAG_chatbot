import pytest
from rag_chatbot.chatbot import Chatbot
from unittest.mock import patch, MagicMock
import tempfile
import os
from reportlab.pdfgen import canvas
from rag_chatbot.utils.vector_store import VectorStore

def create_test_pdf(path):
    """Create a test PDF file with some content."""
    c = canvas.Canvas(path)
    c.drawString(100, 750, "Sample text")
    c.save()

def test_chatbot_initialization():
    """Test that the chatbot initializes correctly."""
    chatbot = Chatbot()
    assert isinstance(chatbot.vector_store, VectorStore)

@patch('rag_chatbot.chatbot.extract_text_from_pdf')
@patch('rag_chatbot.chatbot.split_text')
def test_chatbot_process_document(mock_split_text, mock_extract_text):
    """Test document processing functionality."""
    # Setup mocks
    mock_extract_text.return_value = ["Sample text"]
    mock_split_text.return_value = ["Chunk 1", "Chunk 2"]
    
    chatbot = Chatbot()
    chatbot.vector_store.create_embeddings = MagicMock()
    
    # Process document
    chatbot.process_document("test.pdf")
    
    # Verify mocks were called correctly
    mock_extract_text.assert_called_once_with("test.pdf")
    mock_split_text.assert_called_once_with("Sample text")
    chatbot.vector_store.create_embeddings.assert_called_once_with(["Chunk 1", "Chunk 2"])

@patch('rag_chatbot.chatbot.create_rag_graph')
@patch('rag_chatbot.chatbot.get_answer')
def test_chatbot_query(mock_get_answer, mock_create_rag_graph):
    """Test query functionality."""
    # Setup mocks
    mock_rag_chain = MagicMock()
    mock_create_rag_graph.return_value = mock_rag_chain
    mock_get_answer.return_value = "This is the answer"
    
    chatbot = Chatbot()
    
    # Query chatbot
    result = chatbot.query("What is the meaning of life?")
    
    # Verify mocks were called correctly
    mock_create_rag_graph.assert_called_once()
    mock_get_answer.assert_called_once_with(mock_rag_chain, "What is the meaning of life?")
    assert isinstance(result, str)
    assert result == "This is the answer" 