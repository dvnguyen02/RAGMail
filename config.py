import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

class Config:
    """Configuration constants for the RAGMail application."""
    
    # General settings
    APP_NAME = "RAGMail"
    VERSION = "0.1.0"
    
    # Storage settings
    STORAGE_PATH = os.getenv("STORAGE_PATH", "ragmail_data")
    
    # Email settings
    GMAIL_USERNAME = os.getenv("GMAIL_USERNAME")
    GMAIL_PASSWORD = os.getenv("GMAIL_PASSWORD")
    DEFAULT_DAY_BACKS = 7  # Default number of days to go back
    
    # Embedding model settings
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    
    # OpenAI settings for LLM features
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")  # or "gpt-4" for better quality
    
    # Search settings
    DEFAULT_SEARCH_TYPE = "semantic"
    DEFAULT_SEARCH_LIMIT = 5
    
    # API rate limiting
    API_RATE_LIMIT = int(os.getenv("API_RATE_LIMIT", "10"))