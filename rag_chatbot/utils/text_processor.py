import re
import logging
import time
import textwrap
from typing import List, Callable, Optional, Union

# Import the logger from a separate module
from rag_chatbot.utils.logging_config import get_logger

# Get the logger
logger = get_logger(__name__)

# Constants
CHUNK_SIZE = 500  # Maximum chunk size
CHUNK_OVERLAP = 100  # Overlap between chunks
MAX_PROCESSING_TIME = 30  # Maximum time in seconds for processing
PROGRESS_UPDATE_INTERVAL = 0.5  # Update progress every 0.5 seconds

def find_break_point(text: str, start: int, end: int) -> int:
    """Find the best break point in the text between start and end indices."""
    # First try to find a paragraph break
    para_break = text.rfind('\n\n', start, end)
    if para_break != -1:
        return para_break + 2
    
    # Then try to find a sentence end
    sentence_end = re.search(r'[.!?]\s', text[start:end])
    if sentence_end:
        return start + sentence_end.end()
    
    # If no good break point found, return the original end
    return end

def split_text(text: Union[str, List[str]], chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP, 
              progress_callback: Optional[Callable[[float], None]] = None) -> List[str]:
    """
    Split text into chunks using a simple sliding window approach.
    
    Args:
        text: Either a single string or a list of strings to split
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between chunks
        progress_callback: Optional callback function for progress updates
        
    Returns:
        List of text chunks
    """
    try:
        start_time = time.time()
        
        # Handle list of strings
        if isinstance(text, list):
            logger.info(f"Starting text splitting for {len(text)} texts")
            all_chunks = []
            total_texts = len(text)
            
            for i, t in enumerate(text):
                if not t or not t.strip():
                    continue
                    
                # Update progress for each text
                if time.time() - start_time > MAX_PROCESSING_TIME:
                    logger.warning(f"Text splitting timed out after {MAX_PROCESSING_TIME} seconds")
                    break
                    
                # Process single text
                chunks = split_text(t, chunk_size, chunk_overlap)
                all_chunks.extend(chunks)
                
                # Update progress
                progress = (i + 1) / total_texts
                if progress_callback:
                    progress_callback(progress)
                logger.info(f"Processed text {i + 1}/{total_texts}")
            
            logger.info(f"Split {total_texts} texts into {len(all_chunks)} chunks")
            return all_chunks
            
        # Handle single string
        if not text or not text.strip():
            logger.error("Empty text provided for splitting")
            return []
            
        logger.info(f"Starting text splitting. Text length: {len(text)}")
        text = text.strip()
        
        chunks = []
        current_pos = 0
        last_progress_update = time.time()
        
        while current_pos < len(text):
            # Check for timeout
            if time.time() - start_time > MAX_PROCESSING_TIME:
                logger.warning(f"Text splitting timed out after {MAX_PROCESSING_TIME} seconds")
                break
                
            # Update progress periodically
            if time.time() - last_progress_update > PROGRESS_UPDATE_INTERVAL:
                progress = (current_pos / len(text))
                logger.info(f"Text splitting progress: {progress*100:.1f}%")
                if progress_callback:
                    progress_callback(progress)
                last_progress_update = time.time()
            
            # Calculate the end position for this chunk
            end_pos = min(current_pos + chunk_size, len(text))
            
            # If we're not at the end, try to find a good breaking point
            if end_pos < len(text):
                # Look for the last space in the chunk
                last_space = text.rfind(' ', current_pos, end_pos)
                if last_space != -1:
                    end_pos = last_space + 1
            
            # Extract the chunk
            chunk = text[current_pos:end_pos].strip()
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)
            
            # Move to the next position with overlap
            if end_pos < len(text):
                current_pos = max(current_pos + 1, end_pos - chunk_overlap)
            else:
                current_pos = end_pos
        
        processing_time = time.time() - start_time
        logger.info(f"Text splitting complete in {processing_time:.2f} seconds. Created {len(chunks)} chunks")
        if progress_callback:
            progress_callback(1.0)
        return chunks
    except Exception as e:
        logger.error(f"Error in split_text: {str(e)}")
        raise 