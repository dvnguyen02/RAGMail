"""
Storage components for RAGMail.
"""

from storage.document_store import DocumentStore
from storage.vector_store import VectorStore

__all__ = ['DocumentStore', 'VectorStore']