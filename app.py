import os
import sys
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import email.utils
import argparse
import json
from pathlib import Path
from dotenv import load_dotenv

# Import necessary components
from connectors import GmailConnector
from storage.document_store import DocumentStore
from storage.vector_store import VectorStore
from processors.embedding_processor import EmbeddingProcessor
from service.RAGQuery import RAGQueryService

# Load environment variables
load_dotenv(override=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("RAGMail.log")
    ]
)
logger = logging.getLogger("RAGMail")

# ANSI color codes for better terminal output
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"
BOLD = "\033[1m"

class SimpleRAGMail:
    """Simplified RAGMail application for email summaries and search."""
    
    def __init__(self):
        """Initialize the RAGMail application."""
        print(f"{CYAN}{BOLD}Initializing RAGMail...{RESET}")
        logger.info("Initializing RAGMail application")
        
        self.email_connector = None
        self.document_store = None
        self.query_service = None
        self.is_initialized = False
        
        try:
            self.initialize_components()
            self.is_initialized = True
            print(f"{GREEN}✓ RAGMail initialized successfully{RESET}")
        except Exception as e:
            logger.error(f"Error initializing RAGMail: {e}")
            print(f"{RED}Error initializing RAGMail: {e}{RESET}")
            import traceback
            traceback.print_exc()
    
    def initialize_components(self):
        """Initialize all components of the RAGMail system."""
        # Initialize email connector
        username = os.getenv("GMAIL_USERNAME")
        password = os.getenv("GMAIL_PASSWORD")
        
        if not username or not password:
            raise ValueError("Gmail credentials not found. Set GMAIL_USERNAME and GMAIL_PASSWORD in .env file.")
        
        self.email_connector = GmailConnector(username, password)
        
        # Initialize storage components
        home_dir = str(Path.home())
        storage_path = os.path.join(home_dir, "RAGMail_data")
        os.makedirs(storage_path, exist_ok=True)
        
        # Initialize document store
        self.document_store = DocumentStore(os.path.join(storage_path, "document_store.json"))
        
        # Initialize query service components
        vector_store_dir = os.path.join(storage_path, "vector_store")
        self.vector_store = VectorStore(vector_store_dir)
        
        model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.embedding_processor = EmbeddingProcessor(model_name)
        
        openai_api_key = os.getenv("OPENAI_API_KEY")
        model_name = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        
        if not openai_api_key:
            logger.warning("OpenAI API key not found. Set OPENAI_API_KEY in .env file for LLM features.")
            print(f"{YELLOW}Warning: OpenAI API key not found. Some features may not work properly.{RESET}")
        
        # Create RAG query service
        self.query_service = RAGQueryService(
            document_store=self.document_store,
            vector_store=self.vector_store,
            embedding_processor=self.embedding_processor,
            openai_api_key=openai_api_key,
            model_name=model_name
        )
        
        logger.info("RAGMail components initialized successfully")
    
    def close_connections(self):
        """Close all connections."""
        if self.email_connector:
            self.email_connector.close()
        logger.info("All connections closed")
    
    def sync_recent_emails(self, limit: int = 30) -> int:
        """
        Sync recent emails to the local storage.
        
        Args:
            limit: Maximum number of emails to fetch
            
        Returns:
            Number of emails synced
        """
        try:
            # Connect to email service
            print(f"{CYAN}Connecting to email service...{RESET}")
            if not self.email_connector.connect():
                logger.error("Failed to connect to email service")
                print(f"{RED}Failed to connect to email service{RESET}")
                return 0
            
            # Fetch and process emails
            print(f"{CYAN}Fetching and processing emails...{RESET}")
            raw_emails = self.email_connector.get_emails(limit)
            
            # Store each email and generate embeddings
            synced_count = 0
            for email_dict in raw_emails:
                # Store email in document store
                email_id = self.document_store.put(email_dict)
                
                # Generate embedding for the email
                embedding = self.embedding_processor.generate_embedding(email_dict)
                
                # Store embedding in vector store
                if embedding is not None:
                    self.vector_store.add(email_id, embedding)
                    synced_count += 1
            
            logger.info(f"Synced {synced_count} emails")
            print(f"{GREEN}✓ Synced {synced_count} emails{RESET}")
            return synced_count
            
        except Exception as e:
            logger.error(f"Error syncing emails: {e}")
            print(f"{RED}Error syncing emails: {e}{RESET}")
            return 0
    
    def search_emails(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for emails using semantic search.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            
        Returns:
            List of matching email dictionaries
        """
        logger.info(f"Searching emails for: '{query}'")
        print(f"{CYAN}Searching emails...{RESET}")
        
        try:
            # Use the query service's semantic search
            results = self.query_service.search(query, search_type="semantic", top_k=top_k)
            logger.info(f"Search returned {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Error searching emails: {e}")
            print(f"{RED}Error searching emails: {e}{RESET}")
            return []
    
    def get_recent_emails(self, since_days: int = 1) -> List[Dict[str, Any]]:
        """
        Get emails received since a specified number of days ago.
        
        Args:
            since_days: Number of days to look back
            
        Returns:
            List of email dictionaries received in the time period
        """
        logger.info(f"Fetching emails from the past {since_days} days")
        
        # Calculate the date threshold - make it timezone-aware with UTC
        from datetime import timezone
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=since_days)
        
        # Get all emails from the document store
        all_emails = self.document_store.get_all()
        if not all_emails:
            logger.info("No emails found in document store")
            return []
        
        # Filter emails by date
        recent_emails = []
        for email_id, email_dict in all_emails.items():
            # Parse the email date
            date_str = email_dict.get("Date")
            if not date_str:
                continue
                
            try:
                # Parse the email date string using the imported email.utils module
                # email.utils.parsedate_to_datetime() returns timezone-aware datetime objects
                email_date = email.utils.parsedate_to_datetime(date_str)
                
                # Check if the email is after the cutoff date
                # Both datetimes are now timezone-aware, so comparison will work
                if email_date > cutoff_date:
                    email_dict["id"] = email_id  # Add ID to the email dict
                    recent_emails.append(email_dict)
            except Exception as e:
                logger.warning(f"Error parsing date for email {email_id}: {e}")
                continue
        
        logger.info(f"Found {len(recent_emails)} emails from the past {since_days} days")
        return recent_emails
    
    def generate_daily_summary(self, emails: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a summary of the provided emails.
        
        Args:
            emails: List of email dictionaries to summarize
            
        Returns:
            Dictionary with summary information
        """
        if not emails:
            return {
                "summary": "No new emails received since yesterday.",
                "email_count": 0,
                "categories": [],
                "important": []
            }
        
        if not self.query_service.client:
            return {
                "summary": f"You have received {len(emails)} new emails since yesterday. LLM summary not available (API key missing).",
                "email_count": len(emails),
                "categories": [],
                "important": []
            }
        
        # Prepare email contexts for the LLM
        email_contexts = []
        for email in emails:
            email_ctx = {
                "id": email.get("id", ""),
                "subject": email.get("Subject", "No subject"),
                "from": email.get("From", "Unknown sender"),
                "date": email.get("Date", "Unknown date"),
                "preview": email.get("Body", "")[:500] if "Body" in email else ""  # Limit for token count
            }
            email_contexts.append(email_ctx)
        
        # Prepare messages for the LLM
        system_message = """
        You are an email summary assistant. You will receive a list of recent emails 
        and create a concise daily summary of them.
        
        Your summary should include:
        1. A brief overview of new emails, including total count
        2. Categorization of emails (e.g., work, personal, newsletters, etc.)
        3. Highlight of important or time-sensitive emails
        4. Any patterns or notable points
        
        Format your response as a JSON object with these fields:
        - summary: Overall text summary of the emails
        - email_count: Number of emails
        - categories: Array of category objects with name and count
        - important: Array of important email IDs with brief reason why they're important
        
        Keep your summary concise, focusing on information that would be most 
        useful to someone starting their day.
        """
        
        user_message = f"""
        Generate a daily summary of these {len(emails)} emails received since yesterday:
        
        ```
        {json.dumps(email_contexts, indent=2)}
        ```
        """
        
        try:
            # Generate the summary using the LLM
            response = self.query_service._call_llm_api(
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.2,
                max_tokens=1000,
                json_mode=True
            )
            
            # Parse the response
            try:
                summary_data = json.loads(response)
                return summary_data
            except json.JSONDecodeError:
                logger.warning("Failed to parse LLM response as JSON")
                return {
                    "summary": f"You have received {len(emails)} new emails since yesterday.\n\n{response}",
                    "email_count": len(emails),
                    "categories": [],
                    "important": []
                }
                
        except Exception as e:
            logger.error(f"Error generating email summary: {e}")
            return {
                "summary": f"You have received {len(emails)} new emails since yesterday. Error generating detailed summary: {str(e)}",
                "email_count": len(emails),
                "categories": [],
                "important": []
            }
    
    def display_email_summary(self, summary: Dict[str, Any]):
        """
        Display an email summary in a user-friendly format.
        
        Args:
            summary: Dictionary with summary information
        """
        print(f"\n{BOLD}{CYAN}📬 Your Email Summary{RESET}")
        print(f"{BOLD}{'=' * 40}{RESET}")
        
        # Print the main summary
        print(f"\n{summary.get('summary', 'No summary available.')}\n")
        
        # Print categories if available
        categories = summary.get('categories', [])
        if categories:
            print(f"{BOLD}Email Categories:{RESET}")
            for category in categories:
                name = category.get('name', 'Unknown')
                count = category.get('count', 0)
                print(f"  • {name}: {count} emails")
            print()
        
        # Print important emails if available
        important = summary.get('important', [])
        if important:
            print(f"{BOLD}Important Emails:{RESET}")
            for item in important:
                if isinstance(item, dict):
                    email_id = item.get('id', 'Unknown')
                    reason = item.get('reason', 'No reason provided')
                    
                    # Try to get the email details
                    email = self.document_store.get(email_id)
                    if email:
                        subject = email.get('Subject', 'No subject')
                        sender = email.get('From', 'Unknown sender')
                        print(f"  • {BOLD}{subject}{RESET} from {sender}")
                        print(f"    Reason: {reason}")
                else:
                    print(f"  • {item}")
            print()
        
        print(f"{BOLD}{'=' * 40}{RESET}\n")
    
    def display_search_results(self, results: List[Dict[str, Any]]):
        """
        Display search results in a user-friendly format.
        
        Args:
            results: List of matching email dictionaries
        """
        if not results:
            print(f"{YELLOW}No matching emails found.{RESET}")
            return
        
        print(f"{GREEN}Found {len(results)} matching emails:{RESET}\n")
        for i, email in enumerate(results, 1):
            subject = email.get("Subject", "No subject")
            sender = email.get("From", "Unknown sender")
            date = email.get("Date", "Unknown date")
            body_preview = email.get("Body", "")[:100] + "..." if len(email.get("Body", "")) > 100 else email.get("Body", "")
            
            print(f"{BOLD}{i}. {subject}{RESET}")
            print(f"   From: {sender}")
            print(f"   Date: {date}")
            
            # Show a preview of the body
            print(f"   Preview: {body_preview}")
            
            # Print relevance info if present
            if "similarity_score" in email:
                print(f"   Relevance: {email['similarity_score']:.2f}")
            
            print()
    
    def run(self):
        """Run the RAGMail application."""
        if not self.is_initialized:
            print(f"{RED}RAGMail failed to initialize. Check the logs for details.{RESET}")
            return 1
        
        try:
            # Check if we need to sync emails
            email_count = self.document_store.count()
            if email_count == 0:
                print(f"{YELLOW}No emails found in the database. Syncing recent emails...{RESET}")
                self.sync_recent_emails(limit=30)
            else:
                print(f"{GREEN}Found {email_count} emails in the database.{RESET}")
            
            # Generate and display the daily summary
            print(f"{CYAN}Generating daily email summary...{RESET}")
            recent_emails = self.get_recent_emails(since_days=1)
            if recent_emails:
                summary = self.generate_daily_summary(recent_emails)
                self.display_email_summary(summary)
            else:
                print(f"{YELLOW}No emails from the past day found.{RESET}")
            
            # Simple command loop for searches
            print(f"{BOLD}Simple RAGMail Commands:{RESET}")
            print(f"  {BOLD}search [query]{RESET} - Search for emails matching your query")
            print(f"  {BOLD}exit{RESET} - Exit the application")
            print()
            
            while True:
                command = input(f"{BOLD}RAGMail> {RESET}").strip()
                
                if not command:
                    continue
                
                # Check for exit command
                if command.lower() in ["exit", "quit", "q"]:
                    print("Exiting RAGMail...")
                    break
                
                # Check for search command
                if command.lower().startswith("search "):
                    query = command[7:].strip()
                    if not query:
                        print(f"{YELLOW}Please provide a search query.{RESET}")
                        continue
                    
                    results = self.search_emails(query)
                    self.display_search_results(results)
                else:
                    # Treat as search query
                    results = self.search_emails(command)
                    self.display_search_results(results)
            
            return 0
            
        except KeyboardInterrupt:
            print("\nExiting RAGMail...")
            return 0
        except Exception as e:
            logger.error(f"Error running RAGMail: {e}")
            print(f"{RED}Error: {str(e)}{RESET}")
            import traceback
            traceback.print_exc()
            return 1
        finally:
            self.close_connections()

def main():
    """Main entry point for RAGMail."""
    print(f"{BOLD}{CYAN}RAGMail - Email Summary and Search Tool{RESET}")
    print("A lightweight tool to summarize your recent emails and search your inbox.\n")
    
    # Initialize the application
    app = SimpleRAGMail()
    
    # Run the application
    return app.run()

if __name__ == "__main__":
    sys.exit(main())