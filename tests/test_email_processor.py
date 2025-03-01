import os
import json
from dotenv import load_dotenv

# Import directly from project modules
from processors.email_processor import EmailProcessor
from connectors import GmailConnector

def main():
    # Load environment variables
    load_dotenv()
    
    # Get credentials
    username = os.getenv('GMAIL_USERNAME')
    password = os.getenv('GMAIL_PASSWORD')
    
    if not username or not password:
        print("Error: Gmail credentials not found in environment variables.")
        print("Please set GMAIL_USERNAME and GMAIL_PASSWORD in your .env file.")
        return
    
    # Create the connector and processor
    gmail = GmailConnector(username, password)
    processor = EmailProcessor()
    
    try:
        # Connect to Gmail
        if not gmail.connect():
            print("Failed to connect to Gmail. Please check your credentials.")
            return
        
        # Fetch emails
        print("Fetching emails...")
        raw_emails = gmail.get_emails(limit=4)
        print(f"Fetched {len(raw_emails)} emails.")
        
        # Process the emails
        print("\nProcessing emails...")
        processed_emails = processor.process_batch(raw_emails)
        
        # Display before and after for comparison
        for i, (raw, processed) in enumerate(zip(raw_emails, processed_emails), 1):
            print(f"\n=== Email {i} ===")
            
            # Print original email information
            print("\nORIGINAL EMAIL:")
            print(f"Subject: {raw['Subject']}")
            print(f"From: {raw['From']}")
            print(f"Date: {raw['Date']}")
            print(f"Body (first 100 chars): {raw['Body'][:100]}...")
            
            # Print processed email information
            print("\nPROCESSED EMAIL:")
            print(f"Subject: {processed['Subject']}")
            print(f"From: {processed['From']}")
            print(f"Date: {processed.get('Date_iso', 'Not parsed')}")
            print(f"Clean Body (first 100 chars): {processed.get('body_cleaned', '')[:100]}...")
            
            # Show extracted information
            if 'extracted_emails' in processed and processed['extracted_emails']:
                print(f"Extracted Emails: {', '.join(processed['extracted_emails'][:3])}...")
            
            if 'extracted_urls' in processed and processed['extracted_urls']:
                print(f"Extracted URLs: {', '.join(processed['extracted_urls'][:3])}...")
            
            if 'signature' in processed:
                print(f"Signature detected: {processed['signature'][:50]}...")
        
        # Save a sample processed email to a file for inspection
        if processed_emails:
            with open('sample_processed_email.json', 'w') as f:
                # Convert datetime objects to strings for serialization
                sample_email = processed_emails[0]
                json.dump(sample_email, f, indent=2)
            
            print("\nSaved sample processed email to sample_processed_email.json")
    
    finally:
        # Always close the connection
        gmail.close()
        print("\nDone!")

if __name__ == "__main__":
    main()