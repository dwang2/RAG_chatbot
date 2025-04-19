import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Optional, Callable
from rag_chatbot.utils.logging_config import get_logger

# Initialize logger
logger = get_logger(__name__)

class VectorStore:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the vector store with FAISS index and sentence transformer."""
        try:
            logger.info(f"Loading SentenceTransformer model: {model_name}")
            
            # Disable all warnings
            import warnings
            warnings.filterwarnings("ignore")
            
            # Set environment variables for stability
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
            os.environ['OMP_NUM_THREADS'] = '1'
            
            # Create models directory if it doesn't exist
            os.makedirs('models', exist_ok=True)
            
            # Load model with minimal settings
            self.model = SentenceTransformer(
                model_name,
                device='cpu',
                cache_folder='models',
                use_auth_token=False
            )
            
            # Initialize FAISS index
            self.dimension = self.model.get_sentence_embedding_dimension()
            self.index = faiss.IndexFlatL2(self.dimension)
            self.texts = []
            
            logger.info("VectorStore initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing VectorStore: {str(e)}")
            raise

    def create_embeddings(self, texts: List[str], progress_callback: Optional[Callable[[float], None]] = None) -> np.ndarray:
        """Create embeddings for a list of texts and add them to the FAISS index."""
        try:
            if not texts:
                logger.warning("No texts provided for embedding creation")
                return np.array([])
                
            logger.info(f"Creating embeddings for {len(texts)} texts")
            
            # Process in smaller batches to avoid memory issues
            batch_size = 16  # Reduced batch size
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                try:
                    batch_embeddings = self.model.encode(
                        batch,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                        batch_size=1  # Process one at a time
                    )
                    all_embeddings.append(batch_embeddings)
                    
                    # Update progress
                    if progress_callback:
                        progress = (i + len(batch)) / len(texts)
                        progress_callback(progress)
                except Exception as e:
                    logger.error(f"Error processing batch {i//batch_size}: {str(e)}")
                    continue
            
            if not all_embeddings:
                raise Exception("No embeddings could be created")
                
            # Combine all embeddings
            embeddings = np.vstack(all_embeddings)
            
            # Add to FAISS index
            self.index.add(embeddings)
            self.texts.extend(texts)
            
            logger.info(f"Successfully created and added {len(texts)} embeddings to FAISS index")
            return embeddings
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            raise

    def save_vector_store(self, store_path: str) -> None:
        """Save the FAISS index and associated texts to disk."""
        try:
            if not store_path:
                raise ValueError("Store path cannot be empty")
                
            # Ensure store_path is absolute
            store_path = os.path.abspath(store_path)
            
            # Create directory if it doesn't exist
            store_dir = os.path.dirname(store_path)
            if store_dir:
                os.makedirs(store_dir, exist_ok=True)
            
            # Save FAISS index
            index_path = f"{store_path}.index"
            faiss.write_index(self.index, index_path)
            
            # Save texts
            texts_path = f"{store_path}.pkl"
            with open(texts_path, 'wb') as f:
                pickle.dump(self.texts, f)
            
            logger.info(f"Vector store saved to {store_path}")
            logger.info(f"Index file: {index_path}")
            logger.info(f"Texts file: {texts_path}")
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            raise

    def load_vector_store(self, store_path: str) -> None:
        """Load the FAISS index and associated texts from disk."""
        try:
            if not store_path:
                raise ValueError("Store path cannot be empty")
                
            # Ensure store_path is absolute
            store_path = os.path.abspath(store_path)
            
            # Check if files exist
            index_path = f"{store_path}.index"
            texts_path = f"{store_path}.pkl"
            
            if not os.path.exists(index_path):
                raise FileNotFoundError(f"Index file not found: {index_path}")
            if not os.path.exists(texts_path):
                raise FileNotFoundError(f"Texts file not found: {texts_path}")
            
            # Load FAISS index
            self.index = faiss.read_index(index_path)
            
            # Load texts
            with open(texts_path, 'rb') as f:
                self.texts = pickle.load(f)
            
            logger.info(f"Vector store loaded from {store_path}")
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            raise

    def search(self, query: str, k: int = 5) -> Tuple[List[str], List[float]]:
        """Search for similar texts using the FAISS index."""
        try:
            # Ensure query is a string
            if isinstance(query, dict):
                query = query.get('input', '')  # Handle dictionary input from LangGraph
            elif not isinstance(query, str):
                query = str(query)
            
            if not query or not query.strip():
                logger.warning("Empty query provided for search")
                return [], []
                
            # Create embedding for query
            query_embedding = self.model.encode(
                [query],
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=1
            )
            
            # Search in FAISS index
            distances, indices = self.index.search(query_embedding, k)
            
            # Get results and scores
            results = [self.texts[i] for i in indices[0]]
            scores = [float(1 / (1 + d)) for d in distances[0]]  # Convert distances to similarity scores
            
            logger.info(f"Search completed with {k} results")
            return results, scores
        except Exception as e:
            logger.error(f"Error in search: {str(e)}")
            raise 