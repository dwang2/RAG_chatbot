import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langgraph.graph import Graph
from typing import Optional, Tuple, List, Callable
import numpy as np

from rag_chatbot.utils.text_processor import split_text
from rag_chatbot.utils.vector_store import VectorStore
from rag_chatbot.models.ollama_client import get_ollama_client
from rag_chatbot.utils.logging_config import get_logger
from rag_chatbot.utils.pdf_processor import extract_text_from_pdf, validate_pdf_file
from rag_chatbot.rag_chain import create_rag_graph, get_answer

# Load environment variables
load_dotenv()

# Constants
MAX_FILE_SIZE_MB = 10
SUPPORTED_FILE_TYPES = ["pdf"]

# Initialize logger
logger = get_logger(__name__)

def validate_pdf_file(file) -> Tuple[bool, str]:
    """Validate the uploaded PDF file."""
    if file is None:
        return False, "No file uploaded"
    
    if file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        return False, f"File size exceeds {MAX_FILE_SIZE_MB}MB limit"
    
    if not file.name.lower().endswith('.pdf'):
        return False, "Only PDF files are supported"
    
    return True, ""

def process_pdf_file(pdf_file) -> None:
    """Process a PDF file and create embeddings."""
    try:
        # Step 1: Extract text from PDF
        logger.info("Extracting text from PDF")
        texts = extract_text_from_pdf(pdf_file)
        if not texts:
            raise ValueError("No text could be extracted from the PDF")
        logger.info(f"Extracted {len(texts)} pages")

        # Step 2: Split text into chunks
        logger.info("Splitting text into chunks")
        text_chunks = split_text(texts)
        if not text_chunks:
            raise ValueError("No text chunks could be created")
        logger.info(f"Created {len(text_chunks)} chunks")

        # Step 3: Create embeddings
        logger.info("Creating embeddings")
        vector_store = VectorStore()
        embeddings = vector_store.create_embeddings(text_chunks)
        if embeddings is None:
            raise ValueError("Failed to create embeddings")
        logger.info(f"Created {len(embeddings)} embeddings")

        # Step 4: Save vector store
        logger.info("Saving vector store")
        vector_store.save_vector_store("data/vector_store")
        st.session_state.vector_store = vector_store
        logger.info("Vector store saved successfully")

    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise

def create_rag_chain():
    """Create a RAG chain using LangGraph."""
    try:
        # Initialize vector store
        vector_store = st.session_state.vector_store
        if vector_store is None:
            raise ValueError("Vector store not initialized")
        
        # Create the RAG graph
        graph = create_rag_graph(vector_store)
        
        return graph
    except Exception as e:
        logger.error(f"Failed to create RAG chain: {str(e)}")
        raise

def update_splitting_progress(progress: float) -> None:
    """Update progress for text splitting step."""
    try:
        # Get or create progress elements in session state
        if 'progress_bar' not in st.session_state:
            st.session_state.progress_bar = st.progress(0)
        if 'details_text' not in st.session_state:
            st.session_state.details_text = st.empty()
            
        # Update progress (scale to 25-50 range for this step)
        st.session_state.progress = 25 + (progress * 25)
        st.session_state.progress_bar.progress(st.session_state.progress / 100)
        st.session_state.details_text.text(f"Processing text segments... {progress*100:.1f}%")
    except Exception as e:
        logger.error(f"Error updating splitting progress: {str(e)}")

class Chatbot:
    def __init__(self):
        """Initialize the chatbot with necessary components."""
        self.vector_store = VectorStore()
        logger.info("Chatbot initialized")

    def process_document(self, file_path: str) -> None:
        """Process a document and add it to the vector store."""
        try:
            logger.info(f"Processing document: {file_path}")
            
            # Extract text from PDF
            texts = extract_text_from_pdf(file_path)
            logger.info(f"Extracted {len(texts)} pages from PDF")
            
            # Split text into chunks
            chunks = []
            for text in texts:
                chunks.extend(split_text(text))
            logger.info(f"Split text into {len(chunks)} chunks")
            
            # Create embeddings and add to vector store
            self.vector_store.create_embeddings(chunks)
            logger.info("Document processed and added to vector store")
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise

    def query(self, question: str) -> str:
        """Query the chatbot with a question and get an answer."""
        try:
            logger.info(f"Processing query: {question}")
            
            # Create RAG chain with vector store
            rag_chain = create_rag_graph(self.vector_store)
            
            # Get answer using RAG chain
            answer = get_answer(rag_chain, question)
            logger.info("Generated answer successfully")
            
            return answer
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise

    def save_state(self, path: str) -> None:
        """Save the chatbot's state to disk."""
        try:
            self.vector_store.save_vector_store(path)
            logger.info(f"Chatbot state saved to {path}")
        except Exception as e:
            logger.error(f"Error saving chatbot state: {str(e)}")
            raise

    def load_state(self, path: str) -> None:
        """Load the chatbot's state from disk."""
        try:
            self.vector_store.load_vector_store(path)
            logger.info(f"Chatbot state loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading chatbot state: {str(e)}")
            raise

def main():
    st.set_page_config(
        page_title="RAG Chatbot",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/yourusername/rag-chatbot',
            'Report a bug': "https://github.com/yourusername/rag-chatbot/issues",
            'About': "# RAG-based PDF Question Answering Chatbot"
        }
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .chat-message {
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            display: flex;
        }
        .chat-message.user {
            background-color: #2b313e;
        }
        .chat-message.assistant {
            background-color: #475063;
        }
        .chat-message .avatar {
            width: 20%;
        }
        .chat-message .message {
            width: 80%;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("📚 RAG-based PDF Question Answering Chatbot")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "progress" not in st.session_state:
        st.session_state.progress = 0
    if "progress_bar" not in st.session_state:
        st.session_state.progress_bar = None
    if "details_text" not in st.session_state:
        st.session_state.details_text = None
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = Chatbot()
    
    # Sidebar for PDF upload
    with st.sidebar:
        st.title("📄 Document Upload")
        st.markdown("""
            Upload a PDF document to start asking questions about its content.
            The document will be processed and stored for future questions.
        """)
        
        pdf_file = st.file_uploader(
            "Choose a PDF file",
            type=SUPPORTED_FILE_TYPES,
            help=f"Maximum file size: {MAX_FILE_SIZE_MB}MB"
        )
        
        if pdf_file and not st.session_state.processing:
            is_valid, error_msg = validate_pdf_file(pdf_file)
            if not is_valid:
                st.error(error_msg)
            else:
                st.session_state.processing = True
                try:
                    # Create progress bar and status text
                    st.session_state.progress_bar = st.progress(0)
                    st.session_state.details_text = st.empty()
                    
                    # Step 1: Extract text
                    st.session_state.details_text.text("Extracting text from PDF...")
                    text = extract_text_from_pdf(pdf_file)
                    if not text:
                        raise Exception("No text could be extracted from the PDF")
                    st.session_state.progress = 25
                    st.session_state.progress_bar.progress(st.session_state.progress)
                    st.session_state.details_text.text(f"Extracted {len(text)} characters")
                    
                    # Step 2: Split text into chunks
                    st.session_state.details_text.text("Splitting text into chunks...")
                    text_chunks = split_text(text, progress_callback=update_splitting_progress)
                    if not text_chunks:
                        raise Exception("No text chunks could be created")
                    st.session_state.progress = 50
                    st.session_state.progress_bar.progress(st.session_state.progress)
                    st.session_state.details_text.text(f"Created {len(text_chunks)} chunks")
                    
                    # Step 3: Process document using chatbot
                    st.session_state.details_text.text("Processing document...")
                    st.session_state.chatbot.process_document(pdf_file)
                    st.session_state.vector_store = st.session_state.chatbot.vector_store
                    st.session_state.progress = 75
                    st.session_state.progress_bar.progress(st.session_state.progress)
                    st.session_state.details_text.text("Document processed successfully")
                    
                    # Step 4: Save vector store
                    st.session_state.details_text.text("Saving vector store...")
                    st.session_state.vector_store.save_vector_store("data/vector_store")
                    st.session_state.progress = 100
                    st.session_state.progress_bar.progress(st.session_state.progress)
                    
                    st.session_state.details_text.text("✅ PDF processed successfully!")
                    st.session_state.details_text.text("Document is ready for questions!")
                    st.success("Document is ready for questions!")
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
                    st.session_state.vector_store = None
                finally:
                    st.session_state.processing = False
                    st.session_state.progress = 0
    
    # Main chat interface
    st.markdown("### 💬 Chat")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if st.session_state.vector_store is not None:
        if prompt := st.chat_input("Ask a question about your document"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get answer
            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    try:
                        answer = st.session_state.chatbot.query(prompt)
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
    else:
        st.info("📝 Please upload a PDF document to start chatting.")

if __name__ == "__main__":
    main() 