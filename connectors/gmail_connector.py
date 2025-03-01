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
            self.connection = imaplib.IMAP4_SSL("imap.gmail.com")
            # Login to Gmail
            self.connection.login(self.username, self.password) 
            print ("Sucessfully logged in to email account")
            return True
        except Exception as e: 
            print ("Error connecting to Gmail: ", e)
            return False
    def close(self) -> None: 
        """
        Safely terminates the established connection to the email service
        """
        try: 
            if self.connection: 
                self.connection.close()
                self.connection.logout()
                print ("Disconnected from email account")
        except Exception as e: 
            print("Error disconnecting from Gmail: ", e)
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
                - 'subject': Email subject
                - 'from': Email sender
                - 'to': Email recipient
                - 'date': Email date
                - 'body': Email body
        """
        emails = [] # List to store emails
        try: 
            # Connect to the inbox
            self.connection.select("INBOX")
            # Search for emails
            status, data = self.connection.search(None,"All")

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
                status, data = self.connection.fetch(id, "(RFC822)")
                # RFC822 is Requests to the entire raw email

                if status != "OK":
                    print("Error fetching email with email ID {id}")
                    continue
                
                # Parse the email data
                raw_email = data[0][1] # This is raw email data (in bytes) so it needs to be decoded
                msg = email.message_from_bytes(raw_email)

                # Decode the email msg
                subject = self.decode_email_header(msg["Subject"]) 
                sender = self.decode_email_header(msg["From"])
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
                return emails
        except Exception as e: 
            print("Error fetching emails: ", e)
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
            for part, encoding in decode_header(header): # decode_header returns a list of pairs (text, encoding) 
                if isinstance(part, bytes): # If part is bytes, then decode it
                    if encoding: 
                        decoded_parts.append(part.decode(encoding)) # If encoding is provided, then decode it
                    else: 
                        decoded_parts.append(part.decode()) # If no encoding is provided, then just decode it
                else: 
                    decoded_parts.append(str(part)) # If part is not bytes, then just convert it to string
                
            return " ".join(decoded_parts)
        except Exception:
            return header if header else ""