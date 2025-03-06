import os
import sys
import logging
import argparse
import json
import re
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import tempfile
from pathlib import Path
from dotenv import load_dotenv

# Import components
from connectors import GmailConnector
from storage.document_store import DocumentStore
from service.llm_only_query_service import LLMOnlyQueryService

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

# ANSI color codes
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"
BOLD = "\033[1m"

class RAGMailInteractive:
    """RAGMail application class with LLM-only search and interactive mode."""
    
    def __init__(self):
        """Initialize the RAGMail application."""
        print(f"{CYAN}{BOLD}Initializing RAGMail Interactive...{RESET}")
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
        # Use home directory to avoid permission issues
        home_dir = str(Path.home())
        storage_path = os.path.join(home_dir, "RAGMail_data")
        os.makedirs(storage_path, exist_ok=True)
        
        self.document_store = DocumentStore(os.path.join(storage_path, "document_store.json"))
        
        # Initialize LLM query service
        openai_api_key = os.getenv("OPENAI_API_KEY")
        model_name = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        
        if not openai_api_key:
            logger.warning("OpenAI API key not found. Set OPENAI_API_KEY in .env file for LLM features.")
            print(f"{YELLOW}Warning: OpenAI API key not found. Set OPENAI_API_KEY in .env file for LLM features.{RESET}")
        
        self.query_service = LLMOnlyQueryService(
            document_store=self.document_store,
            openai_api_key=openai_api_key,
            model_name=model_name
        )
        
        logger.info("RAGMail components initialized successfully")
    
    def close_connections(self):
        """Close all connections."""
        if self.email_connector:
            self.email_connector.close()
        logger.info("All connections closed")
    
    def sync_recent_emails(self, days_back: int = None, limit: int = 10) -> int:
        """
        Sync recent emails to the local storage.
        
        Args:
            days_back: Number of days to go back (default: from config)
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
            emails = self.fetch_and_process_emails(limit)
            
            logger.info(f"Synced {len(emails)} emails")
            print(f"{GREEN}✓ Synced {len(emails)} emails{RESET}")
            return len(emails)
            
        except Exception as e:
            logger.error(f"Error syncing emails: {e}")
            print(f"{RED}Error syncing emails: {e}{RESET}")
            return 0
        
    def fetch_and_process_emails(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Fetch emails and store them in the document store.
        No embedding generation needed for LLM-only approach.
        
        Args:
            limit: Maximum number of emails to fetch
            
        Returns:
            List of processed emails
        """
        try:
            # Fetch emails
            logger.info(f"Fetching up to {limit} emails")
            raw_emails = self.email_connector.get_emails(limit)
            
            # Store each email in document store
            processed_emails = []
            
            for email_dict in raw_emails:
                # Store email in document store
                email_id = self.document_store.put(email_dict)
                processed_emails.append(email_dict)
            
            logger.info(f"Processed {len(processed_emails)} emails")
            return processed_emails
            
        except Exception as e:
            logger.error(f"Error fetching and processing emails: {e}")
            return []
    
    def search_emails(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for emails using LLM-based search.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            
        Returns:
            List of matching email dictionaries
        """
        logger.info(f"Searching emails with LLM for query: '{query}'")
        print(f"{CYAN}Searching emails with LLM...{RESET}")
        
        try:
            results = self.query_service.search(query, top_k)
            logger.info(f"Search returned {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Error searching emails: {e}")
            print(f"{RED}Error searching emails: {e}{RESET}")
            return []
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Answer a natural language question about emails.
        
        Args:
            question: Question to answer
            
        Returns:
            Dictionary with answer and supporting information
        """
        logger.info(f"Answering question: '{question}'")
        print(f"{CYAN}Analyzing your question...{RESET}")
        
        try:
            # First perform a search to get relevant emails
            context_emails = self.search_emails(question, top_k=5)
            
            if not context_emails:
                return {
                    "answer": "I couldn't find any relevant emails to answer your question.",
                    "supporting_emails": [],
                    "confidence": "low"
                }
            
            print(f"{CYAN}Found {len(context_emails)} relevant emails. Generating answer...{RESET}")
            
            # Then answer the question using these emails
            answer = self.query_service.natural_language_query(question, context_emails)
            return answer
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return {
                "answer": f"Error processing your question: {str(e)}",
                "error": str(e)
            }
    
    def run_interactive_mode(self):
        """
        Run an interactive shell for querying emails.
        """
        email_count = self.document_store.count()
        
        print(f"{BOLD}{CYAN}Welcome to RAGMail Interactive Mode!{RESET}")
        print(f"You have {email_count} emails in your database.")
        print("")
        print("Available commands:")
        print(f"  {BOLD}search [query]{RESET} - Search for emails matching your query")
        print(f"  {BOLD}ask [question]{RESET} - Ask a question about your emails")
        print(f"  {BOLD}sync [limit]{RESET} - Sync more emails (default limit: 10)")
        print(f"  {BOLD}help{RESET} - Show this help message")
        print(f"  {BOLD}exit{RESET} - Exit the interactive mode")
        print("")
        print("You can also simply type your question directly.")
        print("")
        
        while True:
            try:
                command = input(f"{BOLD}RAGMail> {RESET}").strip()
                
                if not command:
                    continue
                
                # Check for exit command
                if command.lower() in ["exit", "quit", "q"]:
                    print("Exiting interactive mode...")
                    break
                
                # Check for help command
                if command.lower() in ["help", "h", "?"]:
                    print("Available commands:")
                    print(f"  {BOLD}search [query]{RESET} - Search for emails matching your query")
                    print(f"  {BOLD}ask [question]{RESET} - Ask a question about your emails")
                    print(f"  {BOLD}sync [limit]{RESET} - Sync more emails (default limit: 10)")
                    print(f"  {BOLD}help{RESET} - Show this help message")
                    print(f"  {BOLD}exit{RESET} - Exit the interactive mode")
                    continue
                
                # Check for sync command
                if command.lower().startswith("sync"):
                    parts = command.split(maxsplit=1)
                    limit = 10  # Default limit
                    
                    if len(parts) > 1:
                        try:
                            limit = int(parts[1])
                        except ValueError:
                            print(f"{YELLOW}Invalid limit. Using default limit of 10.{RESET}")
                    
                    print(f"Syncing {limit} emails...")
                    self.sync_recent_emails(limit=limit)
                    continue
                
                # Check for search command
                if command.lower().startswith("search "):
                    query = command[7:].strip()
                    if not query:
                        print(f"{YELLOW}Please provide a search query.{RESET}")
                        continue
                    
                    results = self.search_emails(query)
                    
                    if not results:
                        print(f"{YELLOW}No matching emails found.{RESET}")
                    else:
                        print(f"{GREEN}Found {len(results)} matching emails:{RESET}\n")
                        for i, email in enumerate(results, 1):
                            subject = email.get("Subject", "No subject")
                            sender = email.get("From", "Unknown sender")
                            date = email.get("Date", "Unknown date")
                            
                            print(f"{BOLD}{i}. {subject}{RESET}")
                            print(f"   From: {sender}")
                            print(f"   Date: {date}")
                            
                            # Print relevance info if present
                            if "relevance_score" in email:
                                print(f"   Relevance: {email['relevance_score']}/10")
                            
                            if "relevance_reason" in email:
                                print(f"   Reason: {email['relevance_reason']}")
                            
                            print()
                    
                    continue
                
                # Check for ask command or treat as a direct question
                question = command
                if command.lower().startswith("ask "):
                    question = command[4:].strip()
                
                if not question:
                    print(f"{YELLOW}Please provide a question.{RESET}")
                    continue
                
                # Answer the question
                answer = self.answer_question(question)
                
                print(f"\n{BOLD}Q: {question}{RESET}\n")
                print(f"{GREEN}{answer.get('answer', 'No answer available')}{RESET}\n")
                
                if "supporting_emails" in answer and answer["supporting_emails"]:
                    print(f"{YELLOW}Based on these emails:{RESET}")
                    for email_id in answer["supporting_emails"]:
                        email = self.query_service.get_email_by_id(email_id)
                        if email:
                            print(f"- {email.get('Subject', 'No subject')} (from: {email.get('From', 'Unknown')})")
                    print()
                
                if "confidence" in answer:
                    print(f"Confidence: {answer['confidence']}")
                
            except KeyboardInterrupt:
                print("\nExiting interactive mode...")
                break
            except Exception as e:
                print(f"{RED}Error: {str(e)}{RESET}")
                logger.error(f"Error in interactive mode: {e}")


def main():
    """Main entry point for the RAGMail Interactive application."""
    parser = argparse.ArgumentParser(description="RAGMail: Email Management with LLM-only Search")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Sync command
    sync_parser = subparsers.add_parser("sync", help="Sync recent emails")
    sync_parser.add_argument("--days", type=int, help="Number of days to go back")
    sync_parser.add_argument("--limit", type=int, default=10, help="Maximum number of emails to fetch")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search emails using LLM")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--limit", type=int, default=5, help="Maximum number of results to return")
    
    # Ask command
    ask_parser = subparsers.add_parser("ask", help="Ask questions about your emails")
    ask_parser.add_argument("question", help="Question to answer")
    
    # Interactive command
    interactive_parser = subparsers.add_parser("interactive", help="Start interactive mode")
    
    args = parser.parse_args()
    
    # Initialize app
    app = RAGMailInteractive()
    if not app.is_initialized:
        logger.error("Failed to initialize RAGMail application")
        return 1
    
    try:
        # Execute command
        if args.command == "sync":
            num_synced = app.sync_recent_emails(days_back=args.days, limit=args.limit)
            print(f"Synced {num_synced} emails")
            
            # Automatically start interactive mode after sync
            print("\nStarting interactive mode...")
            app.run_interactive_mode()
        elif args.command == "search":
            results = app.search_emails(args.query, top_k=args.limit)
            
            if not results:
                print("No matching emails found.")
            else:
                print(f"Found {len(results)} matching emails:\n")
                for i, email in enumerate(results, 1):
                    subject = email.get("Subject", "No subject")
                    sender = email.get("From", "Unknown sender")
                    date = email.get("Date", "Unknown date")
                    
                    print(f"{BOLD}{i}. {subject}{RESET}")
                    print(f"   From: {sender}")
                    print(f"   Date: {date}")
                    
                    # Print relevance info if present
                    if "relevance_score" in email:
                        print(f"   Relevance: {email['relevance_score']}/10")
                    
                    if "relevance_reason" in email:
                        print(f"   Reason: {email['relevance_reason']}")
                    
                    print()
                
                # Ask if user wants to enter interactive mode
                response = input("Would you like to enter interactive mode? (y/n): ")
                if response.lower() in ["y", "yes"]:
                    app.run_interactive_mode()
                    
        elif args.command == "ask":
            answer = app.answer_question(args.question)
            
            print(f"\n{BOLD}Q: {args.question}{RESET}\n")
            print(f"{GREEN}{answer.get('answer', 'No answer available')}{RESET}\n")
            
            if "supporting_emails" in answer and answer["supporting_emails"]:
                print(f"{YELLOW}Based on these emails:{RESET}")
                for email_id in answer["supporting_emails"]:
                    email = app.query_service.get_email_by_id(email_id)
                    if email:
                        print(f"- {email.get('Subject', 'No subject')} (from: {email.get('From', 'Unknown')})")
                print()
            
            if "confidence" in answer:
                print(f"Confidence: {answer['confidence']}")
            
            # Ask if user wants to enter interactive mode
            response = input("Would you like to enter interactive mode? (y/n): ")
            if response.lower() in ["y", "yes"]:
                app.run_interactive_mode()
                
        elif args.command == "interactive" or not args.command:
            # If interactive command or no command specified, go into interactive mode
            app.run_interactive_mode()
            
        else:
            # No command specified, show help
            parser.print_help()
            
    except Exception as e:
        logger.error(f"Error executing command: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        app.close_connections()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())