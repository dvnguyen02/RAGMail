from typing import Dict, Any, List, Tuple, Optional
import logging

from processors.embedding_processor import EmbeddingProcessor
from storage.document_store import DocumentStore
from storage.vector_store import VectorStore

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QueryService:
    """
    Service for querying emails.
    Provides methods for semantic search and keyword search.
    """
    
    def __init__(
        self, 
        document_store: DocumentStore,
        vector_store: VectorStore,
        embedding_processor: EmbeddingProcessor
    ):
        """
        Initialize the query service.
        
        Args:
            document_store: Document store for retrieving emails
            vector_store: Vector store for searching embeddings
            embedding_processor: Processor to generate embeddings for queries
        """
        self.document_store = document_store
        self.vector_store = vector_store
        self.embedding_processor = embedding_processor
    
    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform semantic search using vector embeddings.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            
        Returns:
            List of matching email dictionaries
        """
        logger.info(f"Performing semantic search for: '{query}'")
        
        # Generate embedding for the query
        query_embedding = self.embedding_processor.generate_query_embedding(query)
        if query_embedding is None:
            logger.warning("Failed to generate query embedding, falling back to keyword search")
            return self.keyword_search(query, top_k)
        
        # Find similar vectors
        similar_results = self.vector_store.find_similar(query_embedding, top_k=top_k)
        
        # Retrieve the emails for the similar vectors
        emails = []
        for email_id, similarity in similar_results:
            email = self.document_store.get(email_id)
            if email:
                # Add the similarity score to the email
                email['similarity_score'] = float(similarity)
                emails.append(email)
        
        logger.info(f"Semantic search returned {len(emails)} results")
        return emails
    
    def keyword_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform keyword-based search.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            
        Returns:
            List of matching email dictionaries
        """
        logger.info(f"Performing keyword search for: '{query}'")
        
        # Use the document store's search function
        results = self.document_store.search(query)
        
        # Limit to top_k results
        if len(results) > top_k:
            results = results[:top_k]
        
        logger.info(f"Keyword search returned {len(results)} results")
        return results
    
    def hybrid_search(self, query: str, top_k: int = 5, semantic_weight: float = 0.7) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining semantic and keyword search.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            semantic_weight: Weight for semantic search results (0.0 to 1.0)
            
        Returns:
            List of matching email dictionaries
        """
        logger.info(f"Performing hybrid search for: '{query}'")
        
        # Perform both types of search
        semantic_results = self.semantic_search(query, top_k=top_k*2)  # Get more results for blending
        keyword_results = self.keyword_search(query, top_k=top_k*2)
        
        # Create a dictionary to blend results with scores
        blended_results = {}
        
        # Add semantic results with their weights
        for email in semantic_results:
            email_id = email['id']
            similarity = email.get('similarity_score', 0.5)  # Default if missing
            blended_results[email_id] = {
                'email': email,
                'score': similarity * semantic_weight
            }
        
        # Add keyword results with their weights
        for i, email in enumerate(keyword_results):
            email_id = email['id']
            # For keyword results, we assign decreasing scores based on position
            keyword_score = 1.0 - (i / len(keyword_results))
            
            if email_id in blended_results:
                # If already in results, add the keyword score
                blended_results[email_id]['score'] += keyword_score * (1 - semantic_weight)
            else:
                blended_results[email_id] = {
                    'email': email,
                    'score': keyword_score * (1 - semantic_weight)
                }
        
        # Sort by combined score and take top_k
        sorted_results = sorted(
            blended_results.values(),
            key=lambda x: x['score'],
            reverse=True
        )[:top_k]
        
        # Extract just the emails
        emails = [item['email'] for item in sorted_results]
        
        logger.info(f"Hybrid search returned {len(emails)} results")
        return emails
    
    def search(self, query: str, search_type: str = 'hybrid', top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search emails using the specified search type.
        
        Args:
            query: Search query string
            search_type: Type of search ('semantic', 'keyword', or 'hybrid')
            top_k: Number of top results to return
            
        Returns:
            List of matching email dictionaries
        """
        if search_type == 'semantic':
            return self.semantic_search(query, top_k)
        elif search_type == 'keyword':
            return self.keyword_search(query, top_k)
        else:  # Default to hybrid
            return self.hybrid_search(query, top_k)
    
    def get_email_by_id(self, email_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific email by ID.
        
        Args:
            email_id: ID of the email to retrieve
            
        Returns:
            Email dictionary or None if not found
        """
        return self.document_store.get(email_id)