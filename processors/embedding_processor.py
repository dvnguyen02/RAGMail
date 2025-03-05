"""Vector embedding generation and storage in FAISS index."""
import numpy as np
from typing import Dict, Any, List, Optional
import logging
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmbeddingProcessor:
    """
    Generate vector embeddings for emails.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding processor.
        
        Args:
            model_name: Name of the sentence-transformers model to use
        """
        try:
            logger.info(f"Loading embedding model: {model_name}")
            self.model = SentenceTransformer(model_name)
            logger.info(f"Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
            raise
    
    def generate_embedding(self, email: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Generate a vector embedding for an email.
        
        Args:
            email: Email dictionary
            
        Returns:
            Vector embedding (numpy array) or None if error
        """
        try:
            # Extract text to embed (prioritize processed content if available)
            if 'body_cleaned' in email:
                text = f"{email.get('subject', '')} {email['body_cleaned']}"
            elif 'body' in email:
                text = f"{email.get('subject', '')} {email['body']}"
            else:
                logger.warning(f"Email has no body content for embedding")
                text = email.get('subject', '')
            
            # Truncate long texts to avoid excessive processing time
            # Most models work best with < 512 tokens
            if len(text) > 5000:
                text = text[:5000]
            
            # Generate embedding
            embedding = self.model.encode(text)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return None
    
    def generate_query_embedding(self, query: str) -> Optional[np.ndarray]:
        """
        Generate a vector embedding for a search query.
        
        Args:
            query: Search query string
            
        Returns:
            Vector embedding (numpy array) or None if error
        """
        try:
            embedding = self.model.encode(query)
            return embedding
        except Exception as e:
            logger.error(f"Error generating query embedding: {str(e)}")
            return None
    
    def batch_generate_embeddings(self, emails: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for a batch of emails.
        
        Args:
            emails: List of email dictionaries
            
        Returns:
            Dictionary mapping email IDs to embeddings
        """
        embeddings = {}
        
        for email in emails:
            if 'id' not in email:
                logger.warning(f"Email missing ID, skipping embedding generation")
                continue
                
            embedding = self.generate_embedding(email)
            if embedding is not None:
                embeddings[email['id']] = embedding
        
        logger.info(f"Generated {len(embeddings)} embeddings")
        return embeddings