import os
import sys
from dotenv import load_dotenv

# Add the parent directory to the Python path so we can import our module
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import our connector
from gmail_connector import GmailConnector

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
        emails = gmail.get_emails(limit=3)
        print(f"Fetched {len(emails)} number of emails.")
        
        # Display email info
        for i, email_dict in enumerate(emails, 1):
            print(f"\n--- Email {i} ---")
            print(f"Subject: {email_dict['Subject']}")
            print(f"From: {email_dict['From']}")
            print(f"Date: {email_dict['Date']}")
            print(f"Body preview: {email_dict['Body'][:100]}...")
            if input("Do you want to see the full email? (y/n): ").lower() == 'y':
                print(email_dict['Body'])
    
    finally:
        # Always close the connection
        gmail.close()
        print("Done!")

if __name__ == "__main__":
    main()