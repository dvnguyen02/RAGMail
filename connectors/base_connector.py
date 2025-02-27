from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseConnector(ABC):
    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to the email service

        Returns 
            bool: True if connection is successful, false otherwise
        """

    @abstractmethod
    def close(self, connection: object) -> None:
        """
        Close the connection to the email service
        """
        pass

    @abstractmethod
    def get_emails(self, connection: object, limit: int) -> List[Dict[str, Any]]:
        """
        Get emails from the inbox
        """
        pass