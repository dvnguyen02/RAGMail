"""
RAGMail: Email Management with RAG
Main application entry point

Author: dvnguyen02
Date: 2025-03-05
"""

import os
import sys
import logging
import argparse
from typing import Dict, Any, List, Optional
import json
from datetime import datetime, timedelta

from dotenv import load_dotenv

# Import RAGMail modules
from connectors import GmailConnector
from processors.email_processor import EmailProcessor
from processors.embedding_processor import EmbeddingProcessor
from storage.document_store import DocumentStore
from storage.vector_store import VectorStore
from service.query_service import QueryService
from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("ragmail.log")
    ]
)
logger = logging.getLogger("RAGMail")

class RAGMailApp:
    """Main RAGMail application class."""
    
    def __init__(self):
        """Initialize the RAGMail application."""
        logger.info("Initializing RAGMail application")
        
        # Load environment variables
        load_dotenv(override=True)
        
        # Initialize components
        try:
            self.initialize_components()
            self.is_initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize RAGMail application: {str(e)}")
            self.is_initialized = False
    
    def initialize_components(self):
        """Initialize all components of the RAGMail system."""
        # Create storage directories if needed
        storage_path = Config.STORAGE_PATH or "ragmail_data"
        email_data_dir = os.path.join(storage_path, "email_data")
        vector_data_dir = os.path.join(storage_path, "vector_data")
        
        # Initialize storage components
        self.document_store = DocumentStore(storage_dir=email_data_dir)
        self.vector_store = VectorStore(storage_dir=vector_data_dir)
        
        # Initialize processors
        self.email_processor = EmailProcessor()
        embedding_model = Config.EMBEDDING_MODEL or "all-MiniLM-L6-v2"
        self.embedding_processor = EmbeddingProcessor(model_name=embedding_model)
        
        # Initialize query service
        self.query_service = QueryService(
            document_store=self.document_store,
            vector_store=self.vector_store,
            embedding_processor=self.embedding_processor
        )
        
        # Initialize email connectors (on demand)
        self.gmail_connector = None
        
        logger.info("RAGMail components initialized successfully")
    
    def connect_to_gmail(self) -> bool:
        """Connect to Gmail using credentials from the config."""
        try:
            username = Config.GMAIL_USERNAME
            password = Config.GMAIL_PASSWORD
            
            if not username or not password:
                logger.error("Gmail credentials not found in environment variables")
                return False
            
            self.gmail_connector = GmailConnector(username, password)
            success = self.gmail_connector.connect()
            
            if success:
                logger.info("Successfully connected to Gmail")
                return True
            else:
                logger.error("Failed to connect to Gmail")
                return False
                
        except Exception as e:
            logger.error(f"Error connecting to Gmail: {str(e)}")
            return False
    
    def fetch_and_process_emails(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Fetch emails from Gmail, process them, and store them in the document store.
        
        Args:
            limit: Maximum number of emails to fetch
            
        Returns:
            List of processed emails
        """
        if not self.gmail_connector:
            if not self.connect_to_gmail():
                logger.error("Cannot fetch emails: not connected to Gmail")
                return []
        
        try:
            # Fetch emails from Gmail
            logger.info(f"Fetching up to {limit} emails from Gmail")
            raw_emails = self.gmail_connector.get_emails(limit)
            logger.info(f"Fetched {len(raw_emails)} emails")
            
            # Process emails
            logger.info("Processing emails")
            processed_emails = []
            for raw_email in raw_emails:
                # Ensure email has an ID
                if "id" not in raw_email:
                    # Use a hash of subject, date, and from as ID
                    id_source = f"{raw_email.get('Subject', '')}-{raw_email.get('Date', '')}-{raw_email.get('From', '')}"
                    raw_email["id"] = f"email_{abs(hash(id_source)) % 10000000:07d}"
                
                # Process email
                processed = self.email_processor.process_email(raw_email)
                processed_emails.append(processed)
                
                # Store in document store
                self.document_store.add(processed)
                
                # Generate and store embedding
                try:
                    embedding = self.embedding_processor.generate_embedding(processed)
                    if embedding is not None:
                        self.vector_store.add(raw_email["id"], embedding)
                    else:
                        logger.warning(f"Failed to generate embedding for email {raw_email['id']}")
                except Exception as e:
                    logger.error(f"Error generating embedding for email {raw_email['id']}: {str(e)}")
            
            logger.info(f"Processed and stored {len(processed_emails)} emails")
            return processed_emails
            
        except Exception as e:
            logger.error(f"Error fetching and processing emails: {str(e)}")
            return []
        finally:
            # No need to close connection yet as we may need it again
            pass
    
    def search_emails(self, query: str, top_k: int = 5, search_type: str = "semantic") -> List[Dict[str, Any]]:
        """
        Search for emails matching a query.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            search_type: Type of search (semantic, keyword, or hybrid)
            
        Returns:
            List of matching email dictionaries
        """
        logger.info(f"Searching emails with query: '{query}' (type: {search_type})")
        
        try:
            if search_type == "semantic":
                results = self.query_service.semantic_search(query, top_k=top_k)
            elif search_type == "keyword":
                results = self.query_service.keyword_search(query, top_k=top_k)
            elif search_type == "hybrid":
                results = self.query_service.hybrid_search(query, top_k=top_k)
            else:
                logger.warning(f"Unknown search type '{search_type}', falling back to semantic search")
                results = self.query_service.semantic_search(query, top_k=top_k)
            
            logger.info(f"Search returned {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Error searching emails: {str(e)}")
            return []
    
    def summarize_emails(self, emails: List[Dict[str, Any]]) -> str:
        """
        Summarize a list of emails.
        
        Args:
            emails: List of email dictionaries
            
        Returns:
            Summary text
        """
        # TODO: Implement email summarization using OpenAI API or other LLM
        # This is a placeholder that returns a basic summary
        if not emails:
            return "No emails to summarize."
        
        summary = f"Summary of {len(emails)} emails:\n\n"
        
        for i, email in enumerate(emails, 1):
            subject = email.get("Subject", "No subject")
            sender = email.get("From", "Unknown sender")
            date = email.get("Date_iso", email.get("Date", "Unknown date"))
            body_summary = email.get("body_summary", "No body summary available")
            
            summary += f"{i}. {subject} - from {sender} on {date}\n"
            summary += f"   {body_summary}\n\n"
        
        return summary
    
    def sync_recent_emails(self, days_back: int = None) -> int:
        """
        Sync recent emails from the past N days.
        
        Args:
            days_back: Number of days to go back (default: use Config.DEFAULT_DAY_BACKS)
            
        Returns:
            Number of synced emails
        """
        if days_back is None:
            days_back = Config.DEFAULT_DAY_BACKS
            
        logger.info(f"Syncing emails from the past {days_back} days")
        
        # For demonstration purposes, we'll just fetch a limited number
        # In a real implementation, you would filter by date
        processed_emails = self.fetch_and_process_emails(limit=20)
        
        return len(processed_emails)
    
    def close_connections(self):
        """Close all open connections."""
        if self.gmail_connector:
            self.gmail_connector.close()
            logger.info("Gmail connection closed")

def main():
    """Main entry point for the RAGMail application."""
    parser = argparse.ArgumentParser(description="RAGMail: Email Management with RAG")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Sync command
    sync_parser = subparsers.add_parser("sync", help="Sync recent emails")
    sync_parser.add_argument("--days", type=int, help="Number of days to go back")
    sync_parser.add_argument("--limit", type=int, default=10, help="Maximum number of emails to fetch")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search emails")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--type", choices=["semantic", "keyword", "hybrid"], default="semantic", help="Search type")
    search_parser.add_argument("--limit", type=int, default=5, help="Maximum number of results to return")
    
    # Summarize command
    summarize_parser = subparsers.add_parser("summarize", help="Summarize recent emails")
    summarize_parser.add_argument("--query", help="Filter emails by query before summarizing")
    summarize_parser.add_argument("--limit", type=int, default=5, help="Maximum number of emails to summarize")
    
    args = parser.parse_args()
    
    # Initialize app
    app = RAGMailApp()
    if not app.is_initialized:
        logger.error("Failed to initialize RAGMail application")
        return 1
    
    try:
        # Execute command
        if args.command == "sync":
            num_synced = app.sync_recent_emails(days_back=args.days)
            print(f"Synced {num_synced} emails")
            
        elif args.command == "search":
            results = app.search_emails(args.query, top_k=args.limit, search_type=args.type)
            
            if not results:
                print("No matching emails found.")
            else:
                print(f"Found {len(results)} matching emails:\n")
                for i, email in enumerate(results, 1):
                    subject = email.get("Subject", "No subject")
                    sender = email.get("From", "Unknown sender")
                    date = email.get("Date_iso", email.get("Date", "Unknown date"))
                    
                    print(f"{i}. {subject}")
                    print(f"   From: {sender}")
                    print(f"   Date: {date}")
                    
                    # Print similarity score if present
                    if "similarity_score" in email:
                        print(f"   Similarity: {email['similarity_score']:.2f}")
                    
                    print()
                    
        elif args.command == "summarize":
            # First, get emails to summarize (either by query or recent)
            if args.query:
                emails = app.search_emails(args.query, top_k=args.limit)
            else:
                # Fetch and process recent emails
                emails = app.fetch_and_process_emails(limit=args.limit)
            
            # Generate and print summary
            summary = app.summarize_emails(emails)
            print(summary)
            
        else:
            # No command specified, show help
            parser.print_help()
            
    except Exception as e:
        logger.error(f"Error executing command: {str(e)}")
        return 1
    finally:
        app.close_connections()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())