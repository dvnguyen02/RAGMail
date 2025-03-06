import os
import json
import time
from typing import Dict, Any, List, Optional, Set
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentStore:
    """
    Simple document store for emails.
    This implementation stores emails in memory with an option to persist to disk.
    """
    
    def __init__(self, storage_dir: str = "email_data"):
        """
        Initialize the document store.
        
        Args:
            storage_dir: Directory to store persisted emails
        """
        self.storage_dir = storage_dir
        self.emails = {}  # In-memory storage: {email_id: email_dict}
        self.persist_enabled = True
        
        # Create the storage directory if it doesn't exist
        if self.persist_enabled and not os.path.exists(storage_dir):
            os.makedirs(storage_dir)
            logger.info(f"Created storage directory: {storage_dir}")
    
    def add(self, email: Dict[str, Any]) -> bool:
        """
        Add an email to the store.
        
        Args:
            email: Email dictionary to store
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Ensure the email has an ID
        if 'id' not in email:
            logger.error("Cannot add email without an ID")
            return False
        
        email_id = email['id']
        
        # Store in memory
        self.emails[email_id] = email
        
        # Persist to disk if enabled
        if self.persist_enabled:
            self._save_to_disk(email_id, email)
        
        return True
    
    def get(self, email_id: str) -> Optional[Dict[str, Any]]:
        """
        Get an email by ID.
        
        Args:
            email_id: ID of the email to retrieve
            
        Returns:
            Email dictionary or None if not found
        """
        # Try to get from memory
        if email_id in self.emails:
            return self.emails[email_id]
        
        # If not in memory and persistence is enabled, try to load from disk
        if self.persist_enabled:
            email = self._load_from_disk(email_id)
            if email:
                # Cache in memory for future access
                self.emails[email_id] = email
                return email
        
        return None
    
    def get_all(self) -> List[Dict[str, Any]]:
        """
        Get all emails in the store.
        
        Returns:
            List of all email dictionaries
        """
        # Return all emails in memory
        return list(self.emails.values())
    
    def get_ids(self) -> Set[str]:
        """
        Get all email IDs in the store.
        
        Returns:
            Set of email IDs
        """
        return set(self.emails.keys())
    
    def delete(self, email_id: str) -> bool:
        """
        Delete an email from the store.
        
        Args:
            email_id: ID of the email to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Remove from memory
        if email_id in self.emails:
            del self.emails[email_id]
        
        # Remove from disk if enabled
        if self.persist_enabled:
            file_path = os.path.join(self.storage_dir, f"{email_id}.json")
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    return True
                except Exception as e:
                    logger.error(f"Error deleting email file {file_path}: {str(e)}")
                    return False
        
        return True
    
    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Search emails by keyword.
        Basic implementation that just looks for the query string in email fields.
        
        Args:
            query: Search query string
            
        Returns:
            List of matching emails
        """
        results = []
        
        for email_id, email in self.emails.items():
            # Check if query exists in subject or body
            subject = email.get('Subject', '').lower()
            body = email.get('body', '').lower() 
            
            if query.lower() in subject or query.lower() in body:
                results.append(email)
        
        return results
    
    def clear(self) -> None:
        """Clear all emails from memory and optionally disk."""
        # Clear memory
        self.emails = {}
        
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
        Get the number of emails in the store.
        
        Returns:
            int: Number of emails
        """
        return len(self.emails)
    
    def _save_to_disk(self, email_id: str, email: Dict[str, Any]) -> bool:
        """
        Save an email to disk.
        
        Args:
            email_id: ID of the email
            email: Email dictionary
            
        Returns:
            bool: True if successful, False otherwise
        """
        file_path = os.path.join(self.storage_dir, f"{email_id}.json")
        
        try:
            with open(file_path, 'w') as f:
                json.dump(email, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving email to {file_path}: {str(e)}")
            return False
    
    def _load_from_disk(self, email_id: str) -> Optional[Dict[str, Any]]:
        """
        Load an email from disk.
        
        Args:
            email_id: ID of the email
            
        Returns:
            Email dictionary or None if not found or error
        """
        file_path = os.path.join(self.storage_dir, f"{email_id}.json")
        
        if not os.path.exists(file_path):
            return None
        
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading email from {file_path}: {str(e)}")
            return None
    
    def load_all_from_disk(self) -> int:
        """
        Load all emails from disk into memory.
        
        Returns:
            int: Number of emails loaded
        """
        if not self.persist_enabled:
            return 0
            
        count = 0
        for filename in os.listdir(self.storage_dir):
            if filename.endswith('.json'):
                email_id = filename[:-5]  # Remove the .json extension
                email = self._load_from_disk(email_id)
                if email:
                    self.emails[email_id] = email
                    count += 1
        
        logger.info(f"Loaded {count} emails from disk")
        return count