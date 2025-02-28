from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseConnector(ABC):
    """
    Abstract base class defining the interface for email service connectors.
    
    This class serves as a contract that all email connector implementations must follow.
    It defines the required methods for connecting to an email service, retrieving emails,
    and closing the connection.

    For example: its usage in the GmailConnector class is as follows:

    class MyCustomConnector(BaseConnector):
        # Implement all abstract methods
            
    connector = MyCustomConnector()
    if connector.connect():
        emails = connector.get_emails(connection, 10)
        # process emails
        connector.close(connection)
    
    """
    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to the email service

        Establishes a connection to the specific email service and returns
        a boolean indicating success or failure.

        Returns 
            bool: True if connection is successful, false otherwise
        """

    @abstractmethod
    def close(self, connection: object) -> bool:
        """
        Safely terminates the established connection to the email service.
        """
        pass

    @abstractmethod
    def get_emails(self, connection: object, limit: int) -> List[Dict[str, Any]]:
        """
        Get emails from the inbox

        Params: 
            connection (object): The connection object returned by the connect method
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
        pass