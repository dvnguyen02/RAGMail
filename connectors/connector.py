import imaplib
import email
from email.header import decode_header
from dotenv import load_dotenv
import os
load_dotenv()

def connect_to_gmail(username, password) -> object: 
    """
    Connect to Gmail and return IMAP connection object
    
    Params: 
        user username
        user password

    Returns: 
        IMAP connection object
    """

    try: 
        # Connect to Gmail's IMAP server
        connection = imaplib.IMAP4_SSL("imap.gmail.com")
        # Login to Gmail
        connection.login(username, password) 
        print ("Sucessfully logged in to email account")
        return connection
    except Exception as e: 
        print ("Error connecting to Gmail: ", e)
        return None

if __name__ =="__main__":
    connect_to_gmail(os.getenv("GMAIL_USERNAME"), os.getenv("GMAIL_PASSWORD"))