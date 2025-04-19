from typing import Optional
from langchain_community.llms import Ollama
from rag_chatbot.utils.logging_config import get_logger
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize logger
logger = get_logger(__name__)

# Ollama configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2")
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.7"))
OLLAMA_TOP_P = float(os.getenv("OLLAMA_TOP_P", "0.9"))
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "2048"))
OLLAMA_NUM_THREAD = int(os.getenv("OLLAMA_NUM_THREAD", "4"))

def check_ollama_connection() -> bool:
    """Check if the Ollama server is running and accessible."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/version")
        if response.status_code == 200:
            logger.info("Ollama server is running and accessible")
            return True
        else:
            logger.warning(f"Ollama server returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        logger.error(f"Could not connect to Ollama server at {OLLAMA_BASE_URL}")
        return False
    except Exception as e:
        logger.error(f"Error checking Ollama connection: {str(e)}")
        return False

def check_ollama_server() -> bool:
    """Check if the Ollama server is running."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/version")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False

def get_ollama_client(model_name: Optional[str] = None) -> Ollama:
    """Get an Ollama client instance."""
    try:
        # Check if server is running
        if not check_ollama_server():
            raise ConnectionError(f"Ollama server is not running at {OLLAMA_BASE_URL}")
        
        # Use provided model name or fall back to environment variable
        model = model_name or OLLAMA_MODEL
        
        logger.info(f"Connecting to Ollama server at {OLLAMA_BASE_URL}")
        logger.info(f"Using model: {model}")
        logger.info(f"Configuration: temperature={OLLAMA_TEMPERATURE}, top_p={OLLAMA_TOP_P}, num_ctx={OLLAMA_NUM_CTX}, num_thread={OLLAMA_NUM_THREAD}")
        
        # Create Ollama client with specific settings
        client = Ollama(
            base_url=OLLAMA_BASE_URL,
            model=model,
            temperature=OLLAMA_TEMPERATURE,
            top_p=OLLAMA_TOP_P,
            num_ctx=OLLAMA_NUM_CTX,
            num_thread=OLLAMA_NUM_THREAD
        )
        
        # Test the connection
        try:
            client.invoke("Hello")
            logger.info("Successfully connected to Ollama server")
        except Exception as e:
            logger.error(f"Failed to connect to Ollama server: {str(e)}")
            raise
        
        return client
    except Exception as e:
        logger.error(f"Error creating Ollama client: {str(e)}")
        raise 