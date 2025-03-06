"""
LLM-only Query Service for RAGMail - Fixed Version
Uses LLM for direct retrieval without semantic vector search
"""

from typing import Dict, Any, List, Tuple, Optional
import logging
import json
import os

# OpenAI API integration
import openai
from openai import OpenAI, OpenAIError
from tenacity import retry, wait_random_exponential, stop_after_attempt

from storage.document_store import DocumentStore

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMOnlyQueryService:
    """
    Query service that uses LLM directly for email retrieval without semantic search.
    """
    
    def __init__(
        self, 
        document_store: DocumentStore,
        openai_api_key: str = None,
        model_name: str = "gpt-3.5-turbo"
    ):
        """
        Initialize the LLM-only query service.
        
        Args:
            document_store: Document store for retrieving emails
            openai_api_key: API key for OpenAI
            model_name: Name of the LLM model to use
        """
        self.document_store = document_store
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model_name or os.getenv("OPENAI_MODEL") or "gpt-3.5-turbo"
        self.supports_json_mode = self._check_model_supports_json_mode(self.model_name)
        
        # Initialize OpenAI client
        if self.openai_api_key:
            try:
                self.client = OpenAI(api_key=self.openai_api_key)
                logger.info(f"Initialized OpenAI client with model: {self.model_name}")
                logger.info(f"JSON mode support: {self.supports_json_mode}")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                self.client = None
        else:
            logger.warning("No OpenAI API key provided, LLM features will not be available")
            self.client = None
    def get_email_by_id(self, email_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve an email by its ID.
        
        Args:
            email_id: ID of the email to retrieve
            
        Returns:
            Email dictionary or None if not found
        """
        return self.document_store.get(email_id)
    
    def _check_model_supports_json_mode(self, model_name: str) -> bool:
        """
        Check if the specified model supports JSON mode.
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            True if the model supports JSON mode, False otherwise
        """
        # As of March 2023, these models support JSON mode
        json_supporting_models = [
            "gpt-4-turbo", "gpt-4-0125-preview", "gpt-4-1106-preview",
            "gpt-4", "gpt-4-0613", "gpt-4-32k", "gpt-4-32k-0613", 
            "gpt-3.5-turbo", "gpt-3.5-turbo-1106", "gpt-3.5-turbo-0125"
        ]
        
        # Check if model name contains any of the supported models
        for supported_model in json_supporting_models:
            if supported_model in model_name:
                return True
        
        return False
    
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def _call_llm_api(
    self, 
    messages: List[Dict[str, str]], 
    temperature: float = 0.0, 
    max_tokens: int = 1024,
    json_mode: bool = False
) -> str:
        """
        Call the OpenAI API with retry logic.
        """
        if not self.client:
            raise ValueError("OpenAI client not available. Check your API key.")
        
        try:
            # If using JSON mode, make sure at least one message contains "json"
            if json_mode and self.supports_json_mode:
                json_mentioned = False
                for message in messages:
                    if "json" in message["content"].lower():
                        json_mentioned = True
                        break
                
                # If json not mentioned, add it to the system message
                if not json_mentioned:
                    # Find the system message
                    for message in messages:
                        if message["role"] == "system":
                            # Append JSON instruction to system message
                            message["content"] += "\nPlease format your response as a valid JSON object."
                            break
                    
                    # If no system message, add one
                    if not json_mentioned:
                        messages.insert(0, {
                            "role": "system", 
                            "content": "Please format your response as a valid JSON object."
                        })
            
            kwargs = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            
            # Add response_format for JSON mode only if supported
            if json_mode and self.supports_json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            
            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
            
        except OpenAIError as e:
            logger.error(f"OpenAI API error: {e}")
            
            # If the error is related to JSON mode, retry without it
            if "response_format" in str(e) and json_mode:
                logger.warning("JSON mode not supported, retrying without it")
                return self._call_llm_api(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=False  # Retry without JSON mode
                )
            raise
            
        except Exception as e:
            logger.error(f"Error calling LLM API: {e}")
            raise
    
    def _parse_llm_response(self, response: str, is_json: bool = True) -> Dict[str, Any]:
        """
        Parse response from LLM, handling both JSON and non-JSON formats.
        
        Args:
            response: Response string from the LLM
            is_json: Whether the response is expected to be JSON
            
        Returns:
            Parsed response as a dictionary
        """
        if not is_json:
            # For non-JSON responses, extract information using simple heuristics
            return self._extract_info_from_text(response)
        
        # For JSON responses, try to parse
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse LLM response as JSON. Falling back to text extraction.")
            return self._extract_info_from_text(response)
    
    def _extract_info_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extract structured information from plain text response.
        
        Args:
            text: Plain text response from LLM
            
        Returns:
            Extracted information as a dictionary
        """
        result = {}
        
        # Try to find relevant emails section
        emails_section = None
        if "relevant email" in text.lower():
            # Find the relevant emails section
            lines = text.split('\n')
            for i, line in enumerate(lines):
                if "relevant email" in line.lower():
                    emails_section = lines[i:]
                    break
        
        # Extract email IDs if available
        email_ids = []
        if emails_section:
            for line in emails_section:
                if "id" in line.lower() and ":" in line:
                    parts = line.split(":", 1)
                    if len(parts) > 1:
                        email_id = parts[1].strip().strip('"').strip("'")
                        try:
                            # Check if it's a numeric ID
                            email_id = int(email_id)
                        except ValueError:
                            pass
                        email_ids.append(email_id)
        
        result["relevant_emails"] = email_ids
        
        # Extract other information
        if "answer" in text.lower() or ":" in text:
            lines = text.split('\n')
            for line in lines:
                if ":" in line:
                    parts = line.split(":", 1)
                    if len(parts) > 1:
                        key = parts[0].strip().lower()
                        value = parts[1].strip()
                        if "relevance" in key or "score" in key:
                            try:
                                result["relevance_score"] = float(value)
                            except ValueError:
                                pass
                        elif "reason" in key:
                            result["reason"] = value
        
        # If we couldn't extract anything useful, just use the full text
        if len(result) <= 1:  # Just has relevant_emails
            result["answer"] = text.strip()
        
        return result
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search emails using LLM evaluation.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            
        Returns:
            List of matching email dictionaries
        """
        logger.info(f"Performing LLM search for: '{query}'")
        
        if not self.client:
            logger.warning("OpenAI client not available, returning empty results")
            return []
        
        # Get all emails from document store
        all_emails = self.document_store.get_all()
        
        if not all_emails:
            logger.info("No emails found in document store")
            return []
        
        # If there are too many emails, we need to filter them first
        # to avoid exceeding token limits
        max_emails_for_llm = 30  # Adjust based on your needs
        
        emails_to_evaluate = all_emails
        if len(all_emails) > max_emails_for_llm:
            # Simple pre-filtering based on keyword matching
            emails_to_evaluate = self._pre_filter_emails(all_emails, query, max_emails_for_llm)
        
        # Create simplified email objects for LLM to evaluate
        email_summaries = []
        for email_id, email in emails_to_evaluate.items():
            # Create a concise summary of each email
            summary = {
                "id": email_id,
                "subject": email.get("Subject", "No subject"),
                "from": email.get("From", "Unknown sender"),
                "date": email.get("Date", "Unknown date"),
                "preview": email.get("Body", "")[:200] if "Body" in email else ""
            }
            email_summaries.append(summary)
        
        # Prepare messages for the LLM
        system_message = """
        You are an email search assistant. You will receive a user's search query and a list of email summaries.
        Your task is to identify which emails are relevant to the query and rank them by relevance.
        
        Return your response with the following information:
        1. A list of relevant emails, including:
           - The ID of the email
           - A relevance score from 0-10 (10 being most relevant)
           - A brief reason why this email is relevant
        
        Only include emails that are actually relevant to the query. If no emails are relevant, say so clearly.
        """
        
        user_message = f"""
        Query: "{query}"
        
        Email Summaries:
        ```
        {json.dumps(email_summaries, indent=2)}
        ```
        
        Evaluate which emails are relevant to this query and rank them by relevance.
        """
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        try:
            # Try with JSON mode first, fall back to text if needed
            response = self._call_llm_api(
                messages=messages,
                temperature=0.0,
                max_tokens=1500,
                json_mode=True  # The method will handle falling back if needed
            )
            
            # Parse the response
            result_data = self._parse_llm_response(response, is_json=self.supports_json_mode)
            
            # Extract relevant email data
            relevant_email_data = []
            if "relevant_emails" in result_data:
                relevant_email_data = result_data.get("relevant_emails", [])
            else:
                # This might be a response without the expected structure
                logger.warning("Unexpected response structure, trying to extract relevant emails")
                # Try to find relevant emails in the data
                for key, value in result_data.items():
                    if isinstance(value, list) and len(value) > 0:
                        # This might be our list of relevant emails
                        relevant_email_data = value
                        break
            
            # Convert to list of dicts if it's not already
            if relevant_email_data and not isinstance(relevant_email_data[0], dict):
                # Simple list of IDs, convert to dict with basic info
                relevant_email_data = [{"id": email_id} for email_id in relevant_email_data]
            
            # Sort by relevance_score in descending order if available
            if relevant_email_data and isinstance(relevant_email_data[0], dict) and "relevance_score" in relevant_email_data[0]:
                relevant_email_data.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
            
            # Limit to top_k
            relevant_email_data = relevant_email_data[:top_k]
            
            # Get the full email objects and add the relevance data
            results = []
            for item in relevant_email_data:
                if isinstance(item, dict):
                    email_id = item.get("id")
                else:
                    # Handle case where item might be just an ID
                    email_id = item
                    item = {"id": email_id}
                
                email = self.document_store.get(email_id)
                if email:
                    # Add relevance information to the email
                    if "relevance_score" in item:
                        email["relevance_score"] = item.get("relevance_score")
                    if "reason" in item:
                        email["relevance_reason"] = item.get("reason")
                    results.append(email)
            
            logger.info(f"LLM search returned {len(results)} relevant results")
            return results
                
        except Exception as e:
            logger.error(f"Error in LLM search: {e}")
            return []
    
    def natural_language_query(self, question: str, context_emails: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Answer natural language questions about emails.
        
        Args:
            question: Natural language question about emails
            context_emails: Optional list of emails to use as context
            
        Returns:
            Dictionary with answer and supporting information
        """
        if not self.client:
            return {
                "answer": "Natural language querying is not available. Check your API key configuration.",
                "error": "LLM service not available"
            }
        
        # If no context emails provided, perform a search first
        if not context_emails:
            context_emails = self.search(question, top_k=5)
        
        if not context_emails:
            return {
                "answer": "I couldn't find any relevant emails to answer your question.",
                "supporting_emails": [],
                "confidence": "low"
            }
        
        # Prepare email contexts for the LLM
        email_contexts = []
        for i, email in enumerate(context_emails):
            email_ctx = {
                "id": email.get("id", str(i)),
                "subject": email.get("Subject", "No subject"),
                "from": email.get("From", "Unknown sender"),
                "date": email.get("Date", "Unknown date"),
                "body": email.get("Body", "")[:1000] if "Body" in email else ""  # Limit for token count
            }
            email_contexts.append(email_ctx)
        
        # Prepare messages for the LLM
        system_message = """
        You are an email assistant that answers questions about emails in the user's inbox.
        Answer the user's question based only on the email information provided in the context.
        
        If the information needed to answer is not in the provided emails, state that clearly.
        
        Format your response like this:
        
        Answer: [Your direct answer to the question]
        
        Supporting Emails: [List the IDs of emails that support your answer]
        
        Confidence: [high/medium/low]
        
        Missing Information: [Any information needed but not available in the emails]
        """
        
        user_message = f"""
        Question: {question}
        
        Available Emails:
        ```
        {json.dumps(email_contexts, indent=2)}
        ```
        """
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        try:
            # Don't rely on JSON mode - use structured text instead
            response = self._call_llm_api(
                messages=messages,
                temperature=0.0,
                max_tokens=800,
                json_mode=False  # Use text format instead of JSON
            )
            
            # Extract information from the response
            answer_data = self._extract_answer_from_text(response, context_emails)
            return answer_data
                
        except Exception as e:
            logger.error(f"Error processing natural language query: {e}")
            return {
                "answer": f"Error processing your query: {str(e)}",
                "error": str(e)
            }

    def _extract_answer_from_text(self, text: str, context_emails: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract structured information from the answer text.
        
        Args:
            text: The response from the LLM
            context_emails: The emails used as context
            
        Returns:
            A dictionary with answer information
        """
        result = {
            "answer": "",
            "supporting_emails": [],
            "confidence": "medium",
            "missing_information": ""
        }
        
        # If the response is very short, use it as the answer directly
        if len(text) < 100 and not any(marker in text.lower() for marker in ["answer:", "supporting emails:", "confidence:"]):
            result["answer"] = text.strip()
            return result
        
        # Split the text into sections
        lines = text.split("\n")
        current_section = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for section headers
            if line.lower().startswith("answer:"):
                current_section = "answer"
                result["answer"] = line[7:].strip()
            elif line.lower().startswith("supporting emails:"):
                current_section = "supporting_emails"
                # Extract any IDs that might be on this line
                ids_part = line[17:].strip()
                if ids_part:
                    self._extract_ids_to_result(ids_part, result, context_emails)
            elif line.lower().startswith("confidence:"):
                current_section = "confidence"
                confidence = line[11:].strip().lower()
                if confidence in ["high", "medium", "low"]:
                    result["confidence"] = confidence
            elif line.lower().startswith("missing information:"):
                current_section = "missing_information"
                result["missing_information"] = line[20:].strip()
            elif current_section == "answer":
                result["answer"] += " " + line
            elif current_section == "supporting_emails":
                self._extract_ids_to_result(line, result, context_emails)
            elif current_section == "missing_information":
                result["missing_information"] += " " + line
        
        # If we couldn't extract structured data, use the whole text as the answer
        if not result["answer"]:
            result["answer"] = text.strip()
        
        return result

    def _extract_ids_to_result(self, line: str, result: Dict[str, Any], context_emails: List[Dict[str, Any]]):
        """
        Extract email IDs from a line of text and add them to the result.
        
        Args:
            line: Line of text that might contain email IDs
            result: Result dictionary to update
            context_emails: The emails used as context
        """
        # Look for IDs, which could be numbers, strings in quotes, or words
        import re
        
        # Try to find IDs in various formats
        id_patterns = [
            r'(\d+)',                    # Numeric IDs
            r'"([^"]+)"',                # Double-quoted IDs
            r"'([^']+)'",                # Single-quoted IDs
            r"ID:\s*([^\s,]+)",          # ID: format
            r"#(\d+)"                    # #123 format
        ]
        
        for pattern in id_patterns:
            matches = re.findall(pattern, line)
            if matches:
                for match in matches:
                    # Try to convert to int if it's a number
                    try:
                        id_val = int(match)
                    except ValueError:
                        id_val = match
                    
                    # Add to supporting emails if not already there
                    if id_val not in result["supporting_emails"]:
                        result["supporting_emails"].append(id_val)
        
        # If we still don't have any IDs, try looking for subjects or senders
        if not result["supporting_emails"]:
            for email in context_emails:
                subject = email.get("Subject", "").lower()
                sender = email.get("From", "").lower()
                
                # Check if the line contains a significant part of the subject or sender
                for text in [subject, sender]:
                    # Check if substantial part of text is in line (more than 5 chars)
                    if len(text) > 5 and text in line.lower():
                        email_id = email.get("id")
                        if email_id and email_id not in result["supporting_emails"]:
                            result["supporting_emails"].append(email_id)    