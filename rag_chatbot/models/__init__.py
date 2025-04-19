"""
Models package for the RAG Chatbot.
"""

from rag_chatbot.models.ollama_client import get_ollama_client, check_ollama_connection

__all__ = ["get_ollama_client", "check_ollama_connection"]
