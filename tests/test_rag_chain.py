import pytest
from rag_chatbot.rag_chain import (
    AgentState,
    format_docs,
    retrieve,
    generate,
    create_rag_graph,
    get_answer
)
from rag_chatbot.utils.vector_store import VectorStore
from unittest.mock import patch, MagicMock
import numpy as np
import warnings

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_agent_state():
    """Test AgentState initialization and type checking."""
    state = AgentState(
        question="test question",
        context=[],
        answer=""
    )
    assert isinstance(state, dict)
    assert state["question"] == "test question"
    assert isinstance(state["context"], list)
    assert state["answer"] == ""

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_format_docs():
    """Test document formatting."""
    docs = ["doc1", "doc2"]
    formatted = format_docs(docs)
    assert isinstance(formatted, str)
    assert "doc1" in formatted
    assert "doc2" in formatted

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_retrieve():
    """Test retrieval function."""
    # Mock vector store
    mock_store = MagicMock(spec=VectorStore)
    mock_store.search.return_value = (["relevant doc"], [0.8])
    
    # Test state
    state = AgentState(
        question="test question",
        context=[],
        answer=""
    )
    
    # Test retrieval
    new_state = retrieve(state, mock_store)
    assert isinstance(new_state, dict)
    assert len(new_state["context"]) == 1
    assert new_state["context"][0] == "relevant doc"
    mock_store.search.assert_called_once()

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_generate():
    """Test answer generation."""
    # Mock LLM
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = "Based on the provided context, Paris is the capital of France."
    
    # Test state with more realistic context
    state = AgentState(
        question="What is the capital of France?",
        context=["Paris is the capital of France."],
        answer=""
    )
    
    with patch('rag_chatbot.models.ollama_client.get_ollama_client', return_value=mock_llm):
        new_state = generate(state)
        assert isinstance(new_state, dict)
        assert "Paris" in new_state["answer"]
        assert "capital of France" in new_state["answer"]

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_create_rag_graph():
    """Test RAG graph creation."""
    # Mock vector store
    mock_store = MagicMock(spec=VectorStore)
    
    # Create graph
    graph = create_rag_graph(mock_store)
    assert graph is not None
    assert hasattr(graph, 'invoke')

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_get_answer():
    """Test end-to-end answer generation."""
    # Mock components
    mock_store = MagicMock(spec=VectorStore)
    mock_store.search.return_value = (["relevant doc"], [0.8])
    
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = "test answer"
    
    with patch('rag_chatbot.models.ollama_client.get_ollama_client', return_value=mock_llm), \
         patch('rag_chatbot.rag_chain.create_rag_graph') as mock_create_graph:
        
        # Mock graph
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "question": "test question",
            "context": ["relevant doc"],
            "answer": "test answer"
        }
        mock_create_graph.return_value = mock_graph
        
        # Test answer generation
        answer = get_answer(mock_graph, "test question")
        assert answer == "test answer"
        mock_graph.invoke.assert_called_once() 