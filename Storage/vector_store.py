import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
import logging
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorStore:
    """
    Vector store for email embeddings.
    Stores vector embeddings for emails and provides functions for similarity search.
    """
    
    def __init__(self, storage_dir: str = "vector_data"):
        """
        Initialize the vector store.
        
        Args:
            storage_dir: Directory to store persisted vectors
        """
        self.storage_dir = storage_dir
        self.vectors = {}  # In-memory storage: {email_id: embedding_vector}
        self.persist_enabled = True
        
        # Create the storage directory if it doesn't exist
        if self.persist_enabled and not os.path.exists(storage_dir):
            os.makedirs(storage_dir)
            logger.info(f"Created vector storage directory: {storage_dir}")
    
    def add(self, email_id: str, vector: np.ndarray) -> bool:
        """
        Add a vector embedding for an email.
        
        Args:
            email_id: ID of the email
            vector: Vector embedding (numpy array)
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Store in memory
        self.vectors[email_id] = vector
        
        # Persist to disk if enabled
        if self.persist_enabled:
            self._save_to_disk(email_id, vector)
        
        return True
    
    def get(self, email_id: str) -> Optional[np.ndarray]:
        """
        Get the vector embedding for an email.
        
        Args:
            email_id: ID of the email
            
        Returns:
            Vector embedding or None if not found
        """
        # Try to get from memory
        if email_id in self.vectors:
            return self.vectors[email_id]
        
        # If not in memory and persistence is enabled, try to load from disk
        if self.persist_enabled:
            vector = self._load_from_disk(email_id)
            if vector is not None:
                # Cache in memory for future access
                self.vectors[email_id] = vector
                return vector
        
        return None
    
    def delete(self, email_id: str) -> bool:
        """
        Delete the vector embedding for an email.
        
        Args:
            email_id: ID of the email
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Remove from memory
        if email_id in self.vectors:
            del self.vectors[email_id]
        
        # Remove from disk if enabled
        if self.persist_enabled:
            file_path = os.path.join(self.storage_dir, f"{email_id}.json")
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    return True
                except Exception as e:
                    logger.error(f"Error deleting vector file {file_path}: {str(e)}")
                    return False
        
        return True
    
    def get_ids(self) -> Set[str]:
        """
        Get all email IDs in the store.
        
        Returns:
            Set of email IDs
        """
        return set(self.vectors.keys())
    
    def find_similar(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find emails with similar vector embeddings.
        
        Args:
            query_vector: Query vector embedding
            top_k: Number of top results to return
            
        Returns:
            List of (email_id, similarity_score) tuples
        """
        if not self.vectors:
            return []
        
        # Reshape to ensure consistent dimensions
        query_vector = query_vector.reshape(1, -1)
        
        # Calculate similarities with all vectors
        similarities = []
        
        for email_id, vector in self.vectors.items():
            # Reshape to ensure consistent dimensions
            vector = vector.reshape(1, -1)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(query_vector, vector)[0][0]
            similarities.append((email_id, similarity))
        
        # Sort by similarity (descending) and take top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def clear(self) -> None:
        """Clear all vectors from memory and optionally disk."""
        # Clear memory
        self.vectors = {}
        
        # Clear disk if enabled
        if self.persist_enabled:
            for filename in os.listdir(self.storage_dir):
                if filename.endswith('.json'):
                    file_path = os.path.join(self.storage_dir, filename)
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        logger.error(f"Error removing file {file_path}: {str(e)}")
    
    def count(self) -> int:
        """
        Get the number of vectors in the store.
        
        Returns:
            int: Number of vectors
        """
        return len(self.vectors)
    
    def _save_to_disk(self, email_id: str, vector: np.ndarray) -> bool:
        """
        Save a vector to disk.
        
        Args:
            email_id: ID of the email
            vector: Vector embedding
            
        Returns:
            bool: True if successful, False otherwise
        """
        file_path = os.path.join(self.storage_dir, f"{email_id}.json")
        
        try:
            # Convert numpy array to list for JSON serialization
            vector_list = vector.tolist()
            
            with open(file_path, 'w') as f:
                json.dump(vector_list, f)
            
            return True
        except Exception as e:
            logger.error(f"Error saving vector to {file_path}: {str(e)}")
            return False
    
    def _load_from_disk(self, email_id: str) -> Optional[np.ndarray]:
        """
        Load a vector from disk.
        
        Args:
            email_id: ID of the email
            
        Returns:
            Vector embedding or None if not found or error
        """
        file_path = os.path.join(self.storage_dir, f"{email_id}.json")
        
        if not os.path.exists(file_path):
            return None
        
        try:
            with open(file_path, 'r') as f:
                vector_list = json.load(f)
            
            # Convert list back to numpy array
            return np.array(vector_list)
        except Exception as e:
            logger.error(f"Error loading vector from {file_path}: {str(e)}")
            return None
    
    def load_all_from_disk(self) -> int:
        """
        Load all vectors from disk into memory.
        
        Returns:
            int: Number of vectors loaded
        """
        if not self.persist_enabled:
            return 0
            
        count = 0
        for filename in os.listdir(self.storage_dir):
            if filename.endswith('.json'):
                email_id = filename[:-5]  # Remove the .json extension
                vector = self._load_from_disk(email_id)
                if vector is not None:
                    self.vectors[email_id] = vector
                    count += 1
        
        logger.info(f"Loaded {count} vectors from disk")
        return count