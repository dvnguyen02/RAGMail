import os
import sys
from dotenv import load_dotenv

# This is the critical fix - we need to add the parent directory of the 'tests' folder
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Import our connector
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
    
    # Create the connector
    gmail = GmailConnector(username, password)
    
    try:
        # Connect to Gmail
        if not gmail.connect():
            print("Failed to connect to Gmail. Please check your credentials.")
            return
        
        # Fetch emails
        emails = gmail.get_emails(limit=3)  # Note: Changed to match your connector's method name
        print(f"Fetched {len(emails)} number of emails.")
        
        # Display email info
        for i, email_dict in enumerate(emails, 1):
            print(f"\n--- Email {i} ---")
            print(f"Subject: {email_dict.get('Subject', 'No Subject')}")  # Using get() to handle missing keys
            print(f"From: {email_dict.get('From', 'No From')}")
            print(f"Date: {email_dict.get('Date', 'No Date')}")
            
            # Check if body exists and is not empty
            if 'Body' in email_dict and email_dict['Body']:
                preview = email_dict['Body'][:100]
                print(f"Body preview: {preview}...")
                if input("Do you want to see the full email? (y/n): ").lower() == 'y':
                    print(email_dict['Body'])
            else:
                print("No body content available.")
    
    finally:
        # Always close the connection
        gmail.close()
        print("Done!")

if __name__ == "__main__":
    main()