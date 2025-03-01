import re
import html
from typing import Dict, Any, List
from datetime import datetime
import email.utils

class EmailProcessor:
    """
    Process and clean email content.
    """
    
    def __init__(self):
        """Initialize the email processor."""
        pass
    
    def process_email(self, email_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an email dictionary to clean and standardize its content.
        
        Args:
            email_dict: Raw email dictionary from the connector
            
        Returns:
            Processed email dictionary
        """
        # Create a copy to avoid modifying the original
        processed = email_dict.copy()
        
        # Clean and standardize fields
        processed = self._standardize_date(processed)
        processed = self._clean_body(processed)
        processed = self._extract_email_addresses(processed)
        processed = self._extract_links(processed)
        processed = self._remove_links(processed)  # Add this line to remove links
        processed = self._extract_signatures(processed)
        
        # Add metadata
        processed['processed'] = True
        processed['processed_at'] = datetime.now().isoformat()
        
        return processed
    
    def _standardize_date(self, email_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert email date string to a standard ISO format.
        
        Args:
            email_dict: Email dictionary
            
        Returns:
            Email dictionary with standardized date
        """
        if 'date' in email_dict and email_dict['Date']:
            try:
                # Parse the email date string
                parsed_date = email.utils.parsedate_to_datetime(email_dict['Date'])
                # Convert to ISO format
                email_dict['Date_iso'] = parsed_date.isoformat()
                # Add additional date fields for easy querying
                email_dict['Date_year'] = parsed_date.year
                email_dict['Date_month'] = parsed_date.month
                email_dict['Date_day'] = parsed_date.day
            except Exception as e:
                print(f"Error standardizing Date: {str(e)}")
        
        return email_dict
    
    def _clean_body(self, email_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean the email body text.
        
        Args:
            email_dict: Email dictionary
            
        Returns:
            Email dictionary with cleaned body
        """
        if 'body' in email_dict and email_dict['body']:
            body = email_dict['body']
            
            # Decode HTML entities
            body = html.unescape(body)
            
            # Remove excessive whitespace
            body = re.sub(r'\s+', ' ', body)
            
            # Remove common email artifacts
            # 1. Remove forwarded message markers
            body = re.sub(r'----+ ?Forwarded message ?----+', '', body)
            
            # 2. Remove reply markers
            body = re.sub(r'On.*wrote:', '', body)
            
            # 3. Remove email disclaimer boilerplate (simplified example)
            body = re.sub(r'This email and any files.*?confidential', '', body, flags=re.DOTALL)
            
            # Store the cleaned body
            email_dict['body_cleaned'] = body.strip()
            
            # Create a shorter summary (first 200 chars)
            email_dict['body_summary'] = body[:200] + '...' if len(body) > 200 else body
        
        return email_dict
    
    def _extract_email_addresses(self, email_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract email addresses from the email body.
        
        Args:
            email_dict: Email dictionary
            
        Returns:
            Email dictionary with extracted email addresses
        """
        if 'body' in email_dict and email_dict['body']:
            # Simple regex for email addresses
            email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
            
            # Find all matches
            email_addresses = re.findall(email_pattern, email_dict['body'])
            
            # Remove duplicates and store
            email_dict['extracted_emails'] = list(set(email_addresses))
        
        return email_dict
    
    def _extract_links(self, email_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract URLs from the email body.
        
        Args:
            email_dict: Email dictionary
            
        Returns:
            Email dictionary with extracted URLs
        """
        if 'body' in email_dict and email_dict['body']:
            # Simple regex for URLs
            url_pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
            
            # Find all matches
            urls = re.findall(url_pattern, email_dict['body'])
            
            # Remove duplicates and store
            email_dict['extracted_urls'] = list(set(urls))
        
        return email_dict
    
    def _remove_links(self, email_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove all URLs from the email body and create a version without links.
        
        Args:
            email_dict: Email dictionary
            
        Returns:
            Email dictionary with body content that has links removed
        """
        if 'body_cleaned' in email_dict and email_dict['body_cleaned']:
            body = email_dict['body_cleaned']
            
            # Comprehensive regex for URLs (includes http, https, www, and common TLDs)
            url_pattern = r'(https?://|www\.)[^\s<>"]+\.[a-zA-Z]{2,}[^\s<>"]*'
            
            # Replace all URLs with an empty string
            body_no_links = re.sub(url_pattern, '', body)
            
            # Clean up any double spaces from removed links
            body_no_links = re.sub(r'\s{2,}', ' ', body_no_links).strip()
            
            # Store the body without links
            email_dict['body_no_links'] = body_no_links
        
        return email_dict
    
    def _extract_signatures(self, email_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Attempt to extract signatures from the email body.
        
        Args:
            email_dict: Email dictionary
            
        Returns:
            Email dictionary with extracted signature
        """
        if 'body' in email_dict and email_dict['body']:
            body = email_dict['body']
            
            # Common signature markers
            signature_markers = [
                r'--\s*\n',       # Standard signature marker
                r'Best regards,',
                r'Regards,',
                r'Sincerely,',
                r'Thanks,',
                r'Thank you,'
            ]
            
            signature = None
            
            # Try to find a signature based on common markers
            for marker in signature_markers:
                match = re.search(f'({marker}.*)', body, re.DOTALL)
                if match:
                    signature = match.group(1)
                    break
            
            if signature:
                email_dict['signature'] = signature.strip()
                
                # Also provide a version of the body without the signature
                body_without_sig = body.replace(signature, '').strip()
                email_dict['body_without_signature'] = body_without_sig
        
        return email_dict

    def process_batch(self, emails: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a batch of emails.
        
        Args:
            emails: List of email dictionaries
            
        Returns:
            List of processed email dictionaries
        """
        return [self.process_email(email) for email in emails]