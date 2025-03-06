import imaplib
import email
from email.header import decode_header
from .base_connector import BaseConnector
from typing import List, Dict, Any

class GmailConnector(BaseConnector):
    """
    Connector for gmail service
    """

    def __init__(self, username: str, password: str): 
        self.username = username
        self.password = password
        self.connection = None

    def connect(self) -> bool:
        """"
        Method to connect to Gmail

        Returns: 
            bool: True if connection is successful, false otherwise
        """
        try: 
            # Connect to Gmail's IMAP server
            print(f"Connecting to Gmail as {self.username}...")
            self.connection = imaplib.IMAP4_SSL("imap.gmail.com")
            
            # Login to Gmail
            print("Attempting login...")
            self.connection.login(self.username, self.password) 
            print("Successfully logged in to email account")
            return True
        except imaplib.IMAP4.error as e:
            print(f"IMAP4 Error connecting to Gmail: {e}")
            return False
        except Exception as e: 
            print(f"Error connecting to Gmail: {e}")
            return False

    def close(self) -> None: 
        """
        Safely terminates the established connection to the email service
        """
        try: 
            if self.connection: 
                try:
                    self.connection.close()
                except Exception as e:
                    print(f"Error closing connection: {e}")
                    
                try:
                    self.connection.logout()
                    print("Disconnected from email account")
                except Exception as e:
                    print(f"Error logging out: {e}")
        except Exception as e: 
            print(f"Error disconnecting from Gmail: {e}")
        finally:
            self.connection = None

    def get_emails(self, limit: int) -> List[Dict[str, Any]]:
        """
        Get emails from the inbox

        Params: 
            limit (int): The maximum number of emails to retrieve
        Returns: 
            List[Dict[str, Any]]: A list of email messages dictionary. 
                Each email message dictionary contains keys such as: 
                - 'Subject': Email subject
                - 'From': Email sender
                - 'to': Email recipient
                - 'Date': Email date
                - 'Body': Email body
        """
        emails = [] # List to store emails
        try: 
            # Connect to the inbox
            print("Selecting INBOX...")
            status, mailbox_data = self.connection.select("INBOX")
            if status != "OK":
                print(f"Error selecting INBOX: {status}")
                return emails
                
            # Search for emails
            print("Searching for emails...")
            status, data = self.connection.search(None, "ALL")  # Fixed: "All" -> "ALL"

            if status != "OK":
                print(f"Error searching Inbox: {status}")
                return emails
            else: 
                email_count = len(data[0].split())
                print(f"Found {email_count} emails in inbox")
            
            # Get the list of email IDs
            email_ids = data[0].split()
            if not email_ids:
                print("No emails found in mailbox")
                return emails

            # Get the most recent emails (last N emails)
            if limit < len(email_ids):
                recent_ids = email_ids[-limit:]
            else: 
                recent_ids = email_ids

            print(f"Fetching {len(recent_ids)} emails...")
            
            # Fetch each email
            for id in recent_ids: 
                print(f"Fetching email ID: {id}")
                status, data = self.connection.fetch(id, "(RFC822)")
                # RFC822 is Requests to the entire raw email

                if status != "OK":
                    print(f"Error fetching email with ID {id}: {status}")
                    continue
                
                # Parse the email data
                raw_email = data[0][1]  # This is raw email data (in bytes) so it needs to be decoded
                msg = email.message_from_bytes(raw_email)

                # Decode the email msg
                subject = self.decode_email_header(msg["Subject"]) 
                sender = self.decode_email_header(msg["From"])
                date = msg["Date"]
                print(f"Processing email: '{subject}' from {sender}")

                # Extract email body
                body = ""

                if msg.is_multipart(): 
                    # if email has multiple parts, then just get the text
                    for part in msg.walk():
                        content_type = part.get_content_type()
                        if content_type == "text/plain":
                            try: 
                                body = part.get_payload(decode=True).decode()
                                break # Just get the first text or plain part
                            except Exception as e: 
                                print(f"Error decoding part: {e}")
                                pass           
                else: 
                    # If email does not have multiple parts, then just get the email body
                    try: 
                        body = msg.get_payload(decode=True).decode()
                    except Exception as e: 
                        print(f"Error decoding body: {e}")
                        body = "Unable to decode email body"
                
                email_dict = {
                    "id": id.decode(),
                    "Subject": subject,
                    "From": sender,
                    "Date": date,
                    "Body": body
                }

                emails.append(email_dict)
            
            # This return statement was incorrectly indented in your original code
            # It should be outside the loop but inside the try block
            print(f"Successfully retrieved {len(emails)} emails")
            return emails
            
        except Exception as e: 
            print(f"Error fetching emails: {e}")
            import traceback
            traceback.print_exc()
            return emails

    def decode_email_header(self, header) -> str:
        """
        Decode email header from bytes to string
        
        Params: 
            header: email header string to decode
        
        Returns: 
            Decoded email header
        """
        if not header: 
            return ""
        
        try: 
            decoded_parts = []
            for part, encoding in decode_header(header): 
                if isinstance(part, bytes): 
                    if encoding: 
                        decoded_parts.append(part.decode(encoding)) 
                    else: 
                        try:
                            decoded_parts.append(part.decode()) 
                        except UnicodeDecodeError:
                            # If decoding fails with default encoding, try latin-1
                            decoded_parts.append(part.decode('latin-1'))
                else: 
                    decoded_parts.append(str(part)) 
                
            return " ".join(decoded_parts)
        except Exception as e:
            print(f"Error decoding header '{header}': {e}")
            return header if header else ""