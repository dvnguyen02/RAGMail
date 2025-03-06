import os
import sys
import json
import numpy as np
import shutil

# Add the parent directory to the Python path so we can import our module
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import our modules
from storage import DocumentStore, VectorStore

# Sample emails and vectors for testing
SAMPLE_EMAILS = [
    {
        "id": "email1",
        "subject": "Meeting Tomorrow",
        "from": "john@example.com",
        "body": "Let's meet tomorrow to discuss the project."
    },
    {
        "id": "email2",
        "subject": "Project Update",
        "from": "sarah@example.com",
        "body": "Here's the latest update on the project. We've made great progress."
    },
    {
        "id": "email3",
        "subject": "Lunch Invitation",
        "from": "mike@example.com",
        "body": "Would you like to grab lunch tomorrow?"
    }
]

# Create some random vectors (simulating embeddings)
SAMPLE_VECTORS = {
    "email1": np.random.rand(384),  # 384 is a common embedding dimension
    "email2": np.random.rand(384),
    "email3": np.random.rand(384)
}

def test_document_store():
    """Test the DocumentStore component."""
    print("\n=== Testing DocumentStore ===")
    
    # Create a clean test directory
    test_dir = "test_email_data"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    
    # Initialize the document store
    doc_store = DocumentStore(storage_dir=test_dir)
    
    # Add emails
    print(f"Adding {len(SAMPLE_EMAILS)} sample emails...")
    for email in SAMPLE_EMAILS:
        doc_store.add(email)
    
    # Verify count
    count = doc_store.count()
    print(f"Email count: {count}")
    assert count == len(SAMPLE_EMAILS), f"Expected {len(SAMPLE_EMAILS)} emails, got {count}"
    
    # Get one email
    email_id = SAMPLE_EMAILS[0]["id"]
    email = doc_store.get(email_id)
    print(f"Retrieved email with ID {email_id}: {email['subject']}")
    
    # Search for emails
    query = "project"
    results = doc_store.search(query)
    print(f"Search for '{query}' returned {len(results)} results:")
    for email in results:
        print(f"  - {email['subject']}")
    
    # Get all email IDs
    ids = doc_store.get_ids()
    print(f"All email IDs: {ids}")
    
    # Delete one email
    doc_store.delete(email_id)
    print(f"Deleted email with ID {email_id}")
    
    # Verify count after deletion
    count = doc_store.count()
    print(f"Email count after deletion: {count}")
    assert count == len(SAMPLE_EMAILS) - 1, f"Expected {len(SAMPLE_EMAILS) - 1} emails, got {count}"
    
    # Try loading from disk
    new_doc_store = DocumentStore(storage_dir=test_dir)
    new_doc_store.load_all_from_disk()
    count = new_doc_store.count()
    print(f"Loaded {count} emails from disk")
    
    # Clean up
    doc_store.clear()
    print("Cleared all emails")
    
    print("DocumentStore tests completed successfully!")
    
    return test_dir

def test_vector_store():
    """Test the VectorStore component."""
    print("\n=== Testing VectorStore ===")
    
    # Create a clean test directory
    test_dir = "test_vector_data"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    
    # Initialize the vector store
    vector_store = VectorStore(storage_dir=test_dir)
    
    # Add vectors
    print(f"Adding {len(SAMPLE_VECTORS)} sample vectors...")
    for email_id, vector in SAMPLE_VECTORS.items():
        vector_store.add(email_id, vector)
    
    # Verify count
    count = vector_store.count()
    print(f"Vector count: {count}")
    assert count == len(SAMPLE_VECTORS), f"Expected {len(SAMPLE_VECTORS)} vectors, got {count}"
    
    # Get one vector
    email_id = list(SAMPLE_VECTORS.keys())[0]
    vector = vector_store.get(email_id)
    print(f"Retrieved vector for email ID {email_id}: shape={vector.shape}")
    
    # Test similarity search
    query_vector = np.random.rand(384)  # Random query vector
    similar_emails = vector_store.find_similar(query_vector, top_k=2)
    print(f"Top 2 similar emails to query vector:")
    for email_id, similarity in similar_emails:
        print(f"  - {email_id}: similarity={similarity:.4f}")
    
    # Get all email IDs
    ids = vector_store.get_ids()
    print(f"All email IDs: {ids}")
    
    # Delete one vector
    vector_store.delete(email_id)
    print(f"Deleted vector for email ID {email_id}")
    
    # Verify count after deletion
    count = vector_store.count()
    print(f"Vector count after deletion: {count}")
    assert count == len(SAMPLE_VECTORS) - 1, f"Expected {len(SAMPLE_VECTORS) - 1} vectors, got {count}"
    
    # Try loading from disk
    new_vector_store = VectorStore(storage_dir=test_dir)
    new_vector_store.load_all_from_disk()
    count = new_vector_store.count()
    print(f"Loaded {count} vectors from disk")
    
    # Clean up
    vector_store.clear()
    print("Cleared all vectors")
    
    print("VectorStore tests completed successfully!")
    
    return test_dir

def main():
    """Run all storage tests."""
    print("Testing RAGMail storage components...")
    
    # Test document store
    doc_test_dir = test_document_store()
    
    # Test vector store
    vec_test_dir = test_vector_store()
    
    # Clean up test directories
    print("\n=== Cleaning up test directories ===")
    if os.path.exists(doc_test_dir):
        shutil.rmtree(doc_test_dir)
        print(f"Removed {doc_test_dir}")
    
    if os.path.exists(vec_test_dir):
        shutil.rmtree(vec_test_dir)
        print(f"Removed {vec_test_dir}")
    
    print("\nAll storage tests completed successfully!")

if __name__ == "__main__":
    main()