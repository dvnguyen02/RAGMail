from typing import Dict, Any, List, Tuple, Optional, Union
import logging
import json
import os
from datetime import datetime

# OpenAI API integration
import openai
from openai import OpenAI, OpenAIError
from tenacity import retry, wait_random_exponential, stop_after_attempt

from processors.embedding_processor import EmbeddingProcessor
from storage.document_store import DocumentStore
from storage.vector_store import VectorStore
from service.query_service import QueryService
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMQueryService(QueryService):
    """
    Enhanced query service that uses LLM to improve retrieval quality.
    Inherits from the base QueryService and adds LLM-powered capabilities.
    """
    
    def __init__(
        self, 
        document_store: DocumentStore,
        vector_store: VectorStore,
        embedding_processor: EmbeddingProcessor,
        openai_api_key: str = None,
        model_name: str = "gpt-3.5-turbo"
    ):
        """
        Initialize the LLM query service.
        
        Args:
            document_store: Document store for retrieving emails
            vector_store: Vector store for searching embeddings
            embedding_processor: Processor to generate embeddings for queries
            openai_api_key: API key for OpenAI (default: from Config)
            model_name: Name of the LLM model to use
        """
        super().__init__(document_store, vector_store, embedding_processor)
        
        # Initialize OpenAI client
        self.openai_api_key = openai_api_key or Config.OPENAI_API_KEY
        self.model_name = model_name or Config.OPENAI_MODEL or "gpt-3.5-turbo"
        
        # Initialize OpenAI client
        if self.openai_api_key:
            try:
                self.client = OpenAI(api_key=self.openai_api_key)
                logger.info(f"Initialized OpenAI client with model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                self.client = None
        else:
            logger.warning("No OpenAI API key provided, LLM features will not be available")
            self.client = None
    
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
        
        Args:
            messages: List of message dictionaries for the conversation
            temperature: Sampling temperature (0.0 for deterministic, higher for more random)
            max_tokens: Maximum number of tokens to generate
            json_mode: Whether to request JSON output
            
        Returns:
            Generated text from the LLM
        """
        if not self.client:
            raise ValueError("OpenAI client not available. Check your API key.")
        
        try:
            kwargs = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            
            # Add response_format for JSON mode if using a compatible model
            if json_mode and "gpt-4" in self.model_name or "gpt-3.5-turbo" in self.model_name:
                kwargs["response_format"] = {"type": "json_object"}
            
            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
            
        except OpenAIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error calling LLM API: {e}")
            raise
    
    def llm_enhanced_search(
        self, 
        query: str, 
        top_k: int = 5,
        context: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Use LLM to enhance the search query and process results.
        
        Args:
            query: User's search query
            top_k: Number of top results to return
            context: Additional context for the search (user preferences, etc.)
            
        Returns:
            List of matching email dictionaries with LLM enhancements
        """
        logger.info(f"Performing LLM-enhanced search for: '{query}'")
        
        if not self.client:
            logger.warning("OpenAI client not available, falling back to hybrid search")
            return self.hybrid_search(query, top_k)
        
        # Step 1: Use LLM to generate improved search queries
        improved_query = self._generate_improved_query(query, context)
        logger.info(f"Improved query: '{improved_query}'")
        
        # Step 2: Perform hybrid search with the improved query
        search_results = self.hybrid_search(improved_query, top_k=top_k*2)  # Get more for filtering
        
        if not search_results:
            logger.info("No results from hybrid search, returning empty list")
            return []
        
        # Step 3: Use LLM to rerank and filter results
        reranked_results = self._rerank_results(query, search_results, top_k, context)
        
        logger.info(f"LLM-enhanced search returned {len(reranked_results)} results")
        return reranked_results
    
    def _generate_improved_query(
        self, 
        query: str, 
        context: Dict[str, Any] = None
    ) -> str:
        """
        Use LLM to generate an improved search query.
        
        Args:
            query: Original user query
            context: Additional context about the user and their needs
            
        Returns:
            Improved search query
        """
        # Prepare messages for the LLM
        system_message = """
        You are an email search expert. Your task is to improve the user's search query to better find 
        relevant emails. Consider potential synonyms, related concepts, and important contextual terms.
        Return ONLY the improved search query text with no additional commentary.
        """
        
        context_str = ""
        if context:
            context_str = "Context: " + json.dumps(context)
        
        user_message = f"Original search query: \"{query}\"\n{context_str}"
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        try:
            improved_query = self._call_llm_api(
                messages=messages,
                temperature=0.3,
                max_tokens=100
            )
            
            # Remove any quotes and extra whitespace
            improved_query = improved_query.strip().strip('"\'')
            return improved_query
            
        except Exception as e:
            logger.error(f"Error generating improved query: {e}")
            return query  # Fall back to the original query
    
    def _rerank_results(
        self, 
        query: str, 
        results: List[Dict[str, Any]], 
        top_k: int,
        context: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Use LLM to rerank search results based on relevance to the query.
        
        Args:
            query: Original search query
            results: List of email dictionaries from initial search
            top_k: Number of results to return after reranking
            context: Additional context about the user and their preferences
            
        Returns:
            Reranked list of email dictionaries
        """
        if not results:
            return []
        
        # Create a simplified version of the results for the LLM
        email_summaries = []
        for i, email in enumerate(results):
            # Create a concise summary of each email
            summary = {
                "id": i,  # Use index as ID for simplicity
                "subject": email.get("Subject", "No subject"),
                "from": email.get("From", "Unknown sender"),
                "date": email.get("Date", "Unknown date"),
                "preview": email.get("Body", "")[:150] if "Body" in email else ""
            }
            email_summaries.append(summary)
        
        # Prepare messages for the LLM
        system_message = """
        You are an email ranking assistant. You will receive a user's search query and a list of email summaries.
        Your task is to rank the emails by their relevance to the query and return the IDs of the most relevant emails.
        
        Return your response as a JSON object with a single "ranked_ids" array containing the IDs of the most relevant emails,
        ordered from most to least relevant. Include only the IDs that are actually relevant to the query.
        """
        
        context_str = ""
        if context:
            context_str = "Context: " + json.dumps(context) + "\n"
        
        user_message = f"""
        Query: "{query}"
        {context_str}
        Email Summaries:
        ```
        {json.dumps(email_summaries, indent=2)}
        ```
        
        Rank these emails by relevance to the query and return the IDs of relevant emails as a JSON object with a "ranked_ids" array.
        """
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        try:
            response = self._call_llm_api(
                messages=messages,
                temperature=0.0,
                max_tokens=200,
                json_mode=True
            )
            
            # Parse the JSON response
            try:
                ranked_data = json.loads(response)
                ranked_ids = ranked_data.get("ranked_ids", [])
                
                # Validate ranked IDs
                valid_ids = [id for id in ranked_ids if isinstance(id, int) and 0 <= id < len(results)]
                
                # Prepare reranked results
                reranked_results = []
                for id in valid_ids[:top_k]:
                    email = results[id].copy()
                    # Add a relevance_rank field to show the ranking
                    email['relevance_rank'] = valid_ids.index(id) + 1
                    reranked_results.append(email)
                
                # If we don't have enough results, add the remaining ones
                if len(reranked_results) < top_k and len(reranked_results) < len(results):
                    # Get IDs that weren't in the ranked list
                    remaining_indices = [i for i in range(len(results)) if i not in valid_ids]
                    for i in remaining_indices:
                        if len(reranked_results) >= top_k:
                            break
                        email = results[i].copy()
                        email['relevance_rank'] = len(reranked_results) + 1
                        reranked_results.append(email)
                
                return reranked_results
                
            except json.JSONDecodeError:
                logger.error(f"Failed to parse LLM response as JSON: {response}")
                # Fall back to the original results
                return results[:top_k]
                
        except Exception as e:
            logger.error(f"Error reranking results: {e}")
            # Fall back to the original results
            return results[:top_k]
    
    def generate_email_summary(
        self, 
        emails: List[Dict[str, Any]], 
        query: str = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of multiple emails.
        
        Args:
            emails: List of email dictionaries to summarize
            query: Original query that led to these emails, if applicable
            
        Returns:
            Dictionary with summary information
        """
        if not emails:
            return {"summary": "No emails to summarize."}
        
        if not self.client:
            # Basic summary without LLM
            summary = f"Summary of {len(emails)} emails:\n\n"
            for i, email in enumerate(emails, 1):
                subject = email.get("Subject", "No subject")
                sender = email.get("From", "Unknown sender")
                date = email.get("Date", "Unknown date")
                summary += f"{i}. {subject} - from {sender} on {date}\n"
            
            return {"summary": summary}
        
        # Prepare email data for the LLM
        email_data = []
        for email in emails:
            # Prepare email for summarization
            email_info = {
                "subject": email.get("Subject", "No subject"),
                "from": email.get("From", "Unknown sender"),
                "date": email.get("Date", "Unknown date"),
                "body": email.get("Body", "")[:5000]  # Limit to avoid token limits
            }
            email_data.append(email_info)
        
        # Prepare messages for the LLM
        query_context = f"related to the query: \"{query}\"" if query else ""
        
        system_message = """
        You are an email summarization assistant. You'll receive information about multiple emails and need to create a comprehensive summary.
        Your summary should include:
        1. A high-level overview of key themes and topics
        2. Important information extracted from the emails
        3. Any action items, deadlines, or follow-ups mentioned
        4. A brief summary of each individual email
        
        Format your response as a JSON object with the following fields:
        - "overall_summary": A paragraph summarizing the collection of emails
        - "key_themes": Array of main themes or topics across emails
        - "action_items": Array of action items or deadlines mentioned
        - "email_summaries": Array of objects containing brief summaries for each email
        """
        
        user_message = f"""
        Please summarize the following {len(emails)} emails {query_context}:
        
        ```
        {json.dumps(email_data, indent=2)}
        ```
        """
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        try:
            response = self._call_llm_api(
                messages=messages,
                temperature=0.3,
                max_tokens=1000,
                json_mode=True
            )
            
            # Parse the JSON response
            try:
                summary_data = json.loads(response)
                return summary_data
            except json.JSONDecodeError:
                logger.error(f"Failed to parse summary response as JSON: {response}")
                # Return the raw response
                return {"summary": response}
                
        except Exception as e:
            logger.error(f"Error generating email summary: {e}")
            # Basic fallback summary
            summary = f"Summary of {len(emails)} emails:\n\n"
            for i, email in enumerate(emails, 1):
                subject = email.get("Subject", "No subject")
                sender = email.get("From", "Unknown sender")
                date = email.get("Date", "Unknown date")
                summary += f"{i}. {subject} - from {sender} on {date}\n"
            
            return {"summary": summary}
    def draft_email_response(
        self, 
        email: Dict[str, Any], 
        response_type: str = "reply",
        instructions: str = None
    ) -> Dict[str, str]:
        """
        Draft a response to an email using LLM.
        
        Args:
            email: The email dictionary to respond to
            response_type: Type of response ("reply", "forward", or "summary")
            instructions: Specific instructions for drafting the response
            
        Returns:
            Dictionary with response fields (subject, body, etc.)
        """
        if not self.client:
            return {
                "error": "LLM service not available. Check your API key configuration."
            }
        
        # Extract email details
        subject = email.get("Subject", "No subject")
        sender = email.get("From", "Unknown sender")
        body = email.get("Body", "")
        
        # Prepare instructions based on response type
        response_instructions = ""
        if response_type == "reply":
            response_instructions = (
                "Draft a professional reply to this email. Be concise but thorough. "
                "Address the points raised in the original email."
            )
        elif response_type == "forward":
            response_instructions = (
                "Draft a brief introduction for forwarding this email. "
                "Explain why this email is being forwarded and any context needed."
            )
        elif response_type == "summary":
            response_instructions = (
                "Create a brief summary of this email highlighting the key points, "
                "any requests or action items, and important information."
            )
        
        # Add custom instructions if provided
        if instructions:
            response_instructions += f"\n\nAdditional instructions: {instructions}"
        
        # Prepare messages for the LLM
        system_message = """
        You are an email assistant that drafts professional and effective email responses.
        Respond with JSON containing:
        - "subject": The appropriate subject line
        - "body": The draft email body
        - "notes": Any notes or suggestions about the response
        """
        
        user_message = f"""
        Original Email:
        - From: {sender}
        - Subject: {subject}
        - Body:
        
        {body}
        
        {response_instructions}
        """
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        try:
            response = self._call_llm_api(
                messages=messages,
                temperature=0.7,  # Higher temperature for more creative responses
                max_tokens=1000,
                json_mode=True
            )
            
            # Parse the JSON response
            try:
                response_data = json.loads(response)
                
                # Format the subject line based on response type
                if response_type == "reply" and not subject.startswith("Re:"):
                    response_data["subject"] = f"Re: {subject}"
                elif response_type == "forward" and not subject.startswith("Fwd:"):
                    response_data["subject"] = f"Fwd: {subject}"
                
                return response_data
                
            except json.JSONDecodeError:
                logger.error(f"Failed to parse response as JSON: {response}")
                return {
                    "subject": f"{'Re: ' if response_type == 'reply' else 'Fwd: ' if response_type == 'forward' else ''}{subject}",
                    "body": response,
                    "error": "Failed to parse LLM response as JSON"
                }
                
        except Exception as e:
            logger.error(f"Error drafting email response: {e}")
            return {
                "subject": f"{'Re: ' if response_type == 'reply' else 'Fwd: ' if response_type == 'forward' else ''}{subject}",
                "body": f"[Error generating response: {str(e)}]",
                "error": str(e)
            }
    
    def extract_email_insights(self, email: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract key insights and information from an email.
        
        Args:
            email: Email dictionary to analyze
            
        Returns:
            Dictionary with extracted insights
        """
        if not self.client:
            return {"error": "LLM service not available"}
        
        # Extract email content
        subject = email.get("Subject", "No subject")
        sender = email.get("From", "Unknown sender")
        body = email.get("Body", "")
        
        # Prepare messages for the LLM
        system_message = """
        You are an email analysis assistant. Extract key information from the provided email.
        Return your analysis as a JSON object with the following fields:
        - "summary": A brief summary of the email content (1-2 sentences)
        - "key_points": Array of the main points or information in the email
        - "action_items": Array of any tasks, requests or action items mentioned
        - "deadlines": Array of any dates or deadlines mentioned
        - "sentiment": The overall tone of the email (formal, friendly, urgent, etc.)
        - "entities": Object containing people, organizations, or projects mentioned
        """
        
        user_message = f"""
        Email:
        - From: {sender}
        - Subject: {subject}
        - Body:
        
        {body}
        
        Analyze this email and extract the key information as specified.
        """
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        try:
            response = self._call_llm_api(
                messages=messages,
                temperature=0.0,  # Low temperature for more consistent analysis
                max_tokens=800,
                json_mode=True
            )
            
            # Parse the JSON response
            try:
                insights = json.loads(response)
                return insights
            except json.JSONDecodeError:
                logger.error(f"Failed to parse insights response as JSON: {response}")
                return {
                    "summary": "Error parsing insights",
                    "error": "Failed to parse LLM response"
                }
                
        except Exception as e:
            logger.error(f"Error extracting email insights: {e}")
            return {
                "summary": "Error extracting insights",
                "error": str(e)
            }
    
    def categorize_emails(self, emails: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Categorize a list of emails into topics or groups.
        
        Args:
            emails: List of email dictionaries to categorize
            
        Returns:
            Dictionary with categorization information
        """
        if not emails:
            return {"categories": {}}
        
        if not self.client:
            # Simple categorization without LLM
            categories = {"uncategorized": [email.get("id", i) for i, email in enumerate(emails)]}
            return {"categories": categories}
        
        # Prepare simplified emails for the LLM
        email_data = []
        for i, email in enumerate(emails):
            email_info = {
                "id": email.get("id", str(i)),
                "subject": email.get("Subject", "No subject"),
                "from": email.get("From", "Unknown sender"),
                "preview": email.get("Body", "")[:200] if "Body" in email else ""
            }
            email_data.append(email_info)
        
        # Prepare messages for the LLM
        system_message = """
        You are an email categorization assistant. Analyze the provided list of emails and categorize them into logical groups.
        
        Return your categorization as a JSON object with:
        - "categories": An object where each key is a category name and each value is an array of email IDs
        - "category_descriptions": An object with category names as keys and brief descriptions as values
        
        Create between 3-7 meaningful categories. Each email should be assigned to exactly one category.
        """
        
        user_message = f"""
        Please categorize these emails into logical groups:
        
        ```
        {json.dumps(email_data, indent=2)}
        ```
        """
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        try:
            response = self._call_llm_api(
                messages=messages,
                temperature=0.2,
                max_tokens=800,
                json_mode=True
            )
            
            # Parse the JSON response
            try:
                categorization = json.loads(response)
                return categorization
            except json.JSONDecodeError:
                logger.error(f"Failed to parse categorization response as JSON: {response}")
                # Simple fallback categorization
                categories = {"uncategorized": [email.get("id", i) for i, email in enumerate(emails)]}
                return {"categories": categories}
                
        except Exception as e:
            logger.error(f"Error categorizing emails: {e}")
            # Simple fallback categorization
            categories = {"uncategorized": [email.get("id", i) for i, email in enumerate(emails)]}
            return {"categories": categories}
    
    def natural_language_query(
        self,
        query: str,
        context_emails: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Answer natural language questions about emails using LLM.
        
        Args:
            query: Natural language query about emails
            context_emails: List of email dictionaries to provide as context
            
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
            context_emails = self.hybrid_search(query, top_k=5)
        
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
        Structure your response as a JSON object with:
        - "answer": Your direct answer to the question
        - "supporting_emails": Array of email IDs that support your answer
        - "confidence": How confident you are in the answer (high, medium, low)
        - "missing_information": Any information needed but not available in the emails
        """
        
        user_message = f"""
        Question: {query}
        
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
            response = self._call_llm_api(
                messages=messages,
                temperature=0.0,  # Lower temperature for factual answers
                max_tokens=800,
                json_mode=True
            )
            
            # Parse the JSON response
            try:
                answer_data = json.loads(response)
                return answer_data
            except json.JSONDecodeError:
                logger.error(f"Failed to parse answer response as JSON: {response}")
                return {
                    "answer": response,
                    "error": "Failed to parse LLM response as JSON"
                }
                
        except Exception as e:
            logger.error(f"Error processing natural language query: {e}")
            return {
                "answer": f"Error processing your query: {str(e)}",
                "error": str(e)
            }