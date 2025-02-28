import imaplib
import email
from email.header import decode_header
import os
from dotenv import load_dotenv
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
def close_connection(connection) -> None:
    """Close the IMAP connection"""
    if connection: 
        connection.close()
        connection.logout()
        print ("Disconnected from email account")
    else: 
        print ("No active connection to disconnect from")


def get_emails(connection, limit) -> dict:
    """Get emails from the inbox
    
    Params:
        connection: IMAP Object
        limit: number of emails to fetch
    Returns:
        List of emails 
    """
    emails = [] # List to store emails
    try: 
        # Connect to the inbox
        connection.select("INBOX")
        # Search for emails
        status, data = connection.search(None,"All")

        if status != "OK":
            print("Error searching Inbox")
            return emails
        else: 
            print ("Emails found")
        
        # Get the list of email IDs
        email_ids = data[0].split()

        # # Get the most recent emails (last N emails)
        if limit < len(email_ids):
            recent_ids = email_ids[-limit:]
        else: 
            recent_ids = email_ids

        # Fetch each email
        for id in recent_ids: 
            status, data = connection.fetch(id, "(RFC822)")
            # RFC822 is Requests to the entire raw email

            if status != "OK":
                print("Error fetching email with email ID {id}")
                continue
            
            # Parse the email data
            raw_email = data[0][1] # This is raw email data (in bytes) so it needs to be decoded
            msg = email.message_from_bytes(raw_email)

            # Decode the email msg
            subject = decode_email_header(msg["Subject"]) 
            sender = decode_email_header(msg["From"])
            date = msg["Date"]

            # Extract email body
            body = ""

            if msg.is_multipart(): # https://docs.python.org/3/library/email.message.html
                # if email has multiple parts, then just get the text
                for part in msg.walk():
                    content_type = part.get_content_type()
                    if content_type == "text/plain":
                        try: 
                            body = part.get_payload(decode=True).decode()
                            break # Just get the first text or plain part
                        except: 
                            pass           
            else: 
                # If email does not have multiple parts, then just get the email body
                try: 
                    body = msg.get_payload(decode=True).decode()
                except: 
                    body = "Unable to decode email body"
            email_dict = {
                    "Subject": subject,
                    "From": sender,
                    "Date": date,
                    "Body": body
            }

            emails.append(email_dict)
    except Exception as e: 
        print("Error fetching emails: ", e)

    return emails
def decode_email_header(header) -> str:
    """TODO 
    Decode email header from bytes to string
    
    Params: 
        header: email header string to decode
    
    Returns: 
        Decoded email header
    """
    if not header: 
        return ""
    
    decoded_parts = []
    try: 
        for part, encoding in decode_header(header): # decode_header returns a list of pairs (text, encoding) 
            if isinstance(part, bytes): # If part is bytes, then decode it
                if encoding: 
                    decoded_parts.append(part.decode(encoding)) # If encoding is provided, then decode it
                else: 
                    decoded_parts.append(part.decode()) # If no encoding is provided, then just decode it
            else: 
                decoded_parts.append(str(part)) # If part is not bytes, then just convert it to string
            
        return " ".join(decoded_parts)
    except: 
        return header if header else ""
        
        
if __name__ == "__main__":
    # Replace with your Gmail credentials
    USERNAME = os.getenv("GMAIL_USERNAME")
    PASSWORD = os.getenv("GMAIL_PASSWORD")
    
    # Connect to Gmail
    conn = connect_to_gmail(USERNAME, PASSWORD)
    
    if conn:
        # Get recent emails
        emails = get_emails(conn, limit=3)
        
        # Print email information
        for i, email_dict in enumerate(emails, 1):
            print(f"\nEmail {i}:")
            print(f"Subject: {email_dict['Subject']}")
            print(f"From: {email_dict['From']}")
            print(f"Date: {email_dict['Date']}")
            print(f"Body: {email_dict['Body']}")
        
        # Close the connection
        close_connection(conn)