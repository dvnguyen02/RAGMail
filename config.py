"""
Configs for the system
Handles the configuration setting and env variables.
"""

import os 
from dotenv import load_dotenv

load_dotenv()

class Config: 
    """Configuration for the RAG system
        Note that all the password used here are app passwords"""

    # Gmail and Outlook credentials
    GMAIL_USERNAME = os.getenv("GMAIL_USERNAME")
    GMAIL_PASSWORD = os.getenv("GMAIL_PASSWORD")

    OUTLOOK_USERNAME = os.getenv("OUTLOOK_USERNAME")
    OUTLOOK_PASSWORD = os.getenv("OUTLOOK_PASSWORD")

    # OPENAI API key
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL")

    # Embedding model - here I use all-MiniLM-L6-V2 since it's free and popular
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

    # Storage configs
    STORAGE_TYPE = os.getenv("STORAGE_TYPE")
    STORAGE_PATH = os.getenv("STORAGE_PATH")

    # CHATGPT model settings
    DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
    DEFAULT_TOP_P = float(os.getenv("DEFAULT_TOP_P", "0.9"))
    DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.6"))

    # App settings
    DEFAULT_DAY_BACKS = int(os.getenv("DEFAULT_DAY_BACKS", "30"))

    @classmethod
    def validate_connection(cls) -> None:
        """Validate the connection to the email servers"""
        if not (cls.GMAIL_USERNAME and cls.GMAIL_PASSWORD) and not (cls.OUTLOOK_USERNAME and cls.OUTLOOK_PASSWORD):
            print("No emails credentials found in the environment variables")
        if not cls.OPENAI_API_KEY:
            print("No OpenAI API key found in the environment variables")

# Validate the connection
Config.validate_connection()