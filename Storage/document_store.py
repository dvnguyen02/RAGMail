import os
import json
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

class DocumentStore:
    """
    Simple document store that stores emails as JSON files.
    Each email is stored as a separate JSON file in the specified directory.
    """
    
    def __init__(self, storage_path: str):
        """
        Initialize the document store.
        
        Args:
            storage_path: Path to the JSON file that stores the documents
        """
        self.storage_path = storage_path
        self.documents = {}
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        
        # Load existing documents if any
        self._load_documents()
    
    def _load_documents(self):
        """Load existing documents from the storage file."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    self.documents = json.load(f)
                logger.info(f"Loaded {len(self.documents)} documents from {self.storage_path}")
            except json.JSONDecodeError:
                logger.error(f"Error decoding JSON from {self.storage_path}")
                self.documents = {}
            except Exception as e:
                logger.error(f"Error loading documents: {e}")
                self.documents = {}
        else:
            logger.info(f"No document store found at {self.storage_path}, creating new store")
            self.documents = {}
    
    def _save_documents(self):
        """Save documents to the storage file."""
        try:
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(self.documents, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved {len(self.documents)} documents to {self.storage_path}")
        except Exception as e:
            logger.error(f"Error saving documents: {e}")
    
    def put(self, document: Dict[str, Any]) -> str:
        """
        Store a document.
        
        Args:
            document: Document to store
            
        Returns:
            ID of the stored document
        """
        # Generate a unique ID based on email fields
        email_id = document.get("id")
        
        if not email_id:
            # Generate an ID if none exists
            subject = document.get("Subject", "")
            sender = document.get("From", "")
            date = document.get("Date", "")
            email_id = f"{hash(subject + sender + date)}"
            document["id"] = email_id
        
        # Store the document
        self.documents[email_id] = document
        
        # Save to disk
        self._save_documents()
        
        return email_id
    
    def get(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a document by ID.
        
        Args:
            document_id: ID of the document to retrieve
            
        Returns:
            Document dictionary or None if not found
        """
        return self.documents.get(document_id)
    
    def delete(self, document_id: str) -> bool:
        """
        Delete a document by ID.
        
        Args:
            document_id: ID of the document to delete
            
        Returns:
            True if document was deleted, False otherwise
        """
        if document_id in self.documents:
            del self.documents[document_id]
            self._save_documents()
            return True
        return False
    
    def get_all(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all documents.
        
        Returns:
            Dictionary of all documents with their IDs as keys
        """
        return self.documents
    
    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Simple keyword-based search.
        
        Args:
            query: Search query string
            
        Returns:
            List of matching documents
        """
        results = []
        query_lower = query.lower()
        
        for document in self.documents.values():
            # Check subject
            subject = document.get("Subject", "").lower()
            if query_lower in subject:
                results.append(document)
                continue
            
            # Check sender
            sender = document.get("From", "").lower()
            if query_lower in sender:
                results.append(document)
                continue
            
            # Check body
            body = document.get("Body", "").lower()
            if query_lower in body:
                results.append(document)
                continue
        
        return results
    
    def clear(self):
        """Clear all documents."""
        self.documents = {}
        self._save_documents()
    
    def count(self) -> int:
        """
        Get the number of documents.
        
        Returns:
            Number of documents
        """
        return len(self.documents)