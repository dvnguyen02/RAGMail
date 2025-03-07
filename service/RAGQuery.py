from typing import Dict, Any, List, Tuple, Optional
import logging
import json
import os

# OpenAI API integration
import openai
from openai import OpenAI, OpenAIError
from tenacity import retry, wait_random_exponential, stop_after_attempt

from storage.document_store import DocumentStore
from storage.vector_store import VectorStore
from processors.embedding_processor import EmbeddingProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGQueryService:
    """
    Retrieval-Augmented Generation Query Service that combines vector search with LLM generation.
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
        Initialize the RAG query service.
        
        Args:
            document_store: Document store for retrieving emails
            vector_store: Vector store for embedding-based search
            embedding_processor: Processor to generate embeddings for queries
            openai_api_key: API key for OpenAI
            model_name: Name of the LLM model to use
        """
        self.document_store = document_store
        self.vector_store = vector_store
        self.embedding_processor = embedding_processor
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
    
    def search(self, query: str, search_type: str = 'semantic', top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search emails using the specified search type.
        
        Args:
            query: Search query string
            search_type: Type of search ('semantic', 'keyword', or 'hybrid')
            top_k: Number of top results to return
            
        Returns:
            List of matching email dictionaries
        """
        logger.info(f"Performing {search_type} search for: '{query}'")
        
        if search_type == 'semantic':
            return self._semantic_search(query, top_k)
        elif search_type == 'keyword':
            return self._keyword_search(query, top_k)
        else:  # Default to hybrid
            return self._hybrid_search(query, top_k)
    
    def _semantic_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform semantic search using vector embeddings.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            
        Returns:
            List of matching email dictionaries
        """
        # Generate embedding for the query
        query_embedding = self.embedding_processor.generate_query_embedding(query)
        if query_embedding is None:
            logger.warning("Failed to generate query embedding, falling back to keyword search")
            return self._keyword_search(query, top_k)
        
        # Find similar vectors
        similar_results = self.vector_store.find_similar(query_embedding, top_k=top_k)
        
        # Retrieve the emails for the similar vectors
        emails = []
        for email_id, similarity in similar_results:
            email = self.document_store.get(email_id)
            if email:
                # Add the similarity score to the email
                email['similarity_score'] = float(similarity)
                emails.append(email)
        
        logger.info(f"Semantic search returned {len(emails)} results")
        return emails
    
    def _keyword_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform keyword-based search.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            
        Returns:
            List of matching email dictionaries
        """
        # Use the document store's search function
        results = self.document_store.search(query)
        
        # Limit to top_k results
        if len(results) > top_k:
            results = results[:top_k]
        
        logger.info(f"Keyword search returned {len(results)} results")
        return results
    
    def _hybrid_search(self, query: str, top_k: int = 5, semantic_weight: float = 0.7) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining semantic and keyword search.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            semantic_weight: Weight for semantic search results (0.0 to 1.0)
            
        Returns:
            List of matching email dictionaries
        """
        # Perform both types of search
        semantic_results = self._semantic_search(query, top_k=top_k*2)  # Get more results for blending
        keyword_results = self._keyword_search(query, top_k=top_k*2)
        
        # Create a dictionary to blend results with scores
        blended_results = {}
        
        # Add semantic results with their weights
        for email in semantic_results:
            email_id = email['id']
            similarity = email.get('similarity_score', 0.5)  # Default if missing
            blended_results[email_id] = {
                'email': email,
                'score': similarity * semantic_weight
            }
        
        # Add keyword results with their weights
        for i, email in enumerate(keyword_results):
            email_id = email['id']
            # For keyword results, we assign decreasing scores based on position
            keyword_score = 1.0 - (i / len(keyword_results))
            
            if email_id in blended_results:
                # If already in results, add the keyword score
                blended_results[email_id]['score'] += keyword_score * (1 - semantic_weight)
            else:
                blended_results[email_id] = {
                    'email': email,
                    'score': keyword_score * (1 - semantic_weight)
                }
        
        # Sort by combined score and take top_k
        sorted_results = sorted(
            blended_results.values(),
            key=lambda x: x['score'],
            reverse=True
        )[:top_k]
        
        # Extract just the emails
        emails = [item['email'] for item in sorted_results]
        
        logger.info(f"Hybrid search returned {len(emails)} results")
        return emails
    
    def natural_language_query(self, question: str, context_emails: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Answer natural language questions about emails using RAG.
        
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
        
        # If no context emails provided, perform a semantic search first
        if not context_emails:
            context_emails = self.search(question, search_type="semantic", top_k=5)
        
        if not context_emails:
            return {
                "answer": "I couldn't find any relevant emails to answer your question.",
                "supporting_emails": [],
                "confidence": "low"
            }
        
        # Prepare email contexts for the LLM
        email_contexts = []
        supporting_email_ids = []
        
        for i, email in enumerate(context_emails):
            email_id = email.get("id", str(i))
            supporting_email_ids.append(email_id)
            
            email_ctx = {
                "id": email_id,
                "subject": email.get("Subject", "No subject"),
                "from": email.get("From", "Unknown sender"),
                "date": email.get("Date", "Unknown date"),
                "body": email.get("Body", "")[:1000] if "Body" in email else ""  # Limit for token count
            }
            email_contexts.append(email_ctx)
        
        # Prepare messages for the LLM - This is the Augmentation part of RAG
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
            # This is the Generation part of RAG
            response = self._call_llm_api(
                messages=messages,
                temperature=0.0,
                max_tokens=800,
                json_mode=False
            )
            
            # Extract information from the response
            answer_data = self._extract_answer_from_text(response, supporting_email_ids)
            return answer_data
                
        except Exception as e:
            logger.error(f"Error processing natural language query: {e}")
            return {
                "answer": f"Error processing your query: {str(e)}",
                "error": str(e)
            }

    def _extract_answer_from_text(self, text: str, supporting_email_ids: List[str]) -> Dict[str, Any]:
        """
        Extract structured information from the answer text.
        
        Args:
            text: The response from the LLM
            supporting_email_ids: IDs of emails used as context
            
        Returns:
            A dictionary with answer information
        """
        result = {
            "answer": "",
            "supporting_emails": supporting_email_ids,  # Default to all context emails
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
                    # Parse supporting emails - only use explicitly mentioned ones
                    mentioned_ids = []
                    for id_part in ids_part.split(","):
                        id_part = id_part.strip()
                        if id_part in supporting_email_ids:
                            mentioned_ids.append(id_part)
                    
                    if mentioned_ids:
                        result["supporting_emails"] = mentioned_ids
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
                # Try to parse mentioned IDs
                for id_part in line.split(","):
                    id_part = id_part.strip()
                    if id_part in supporting_email_ids and id_part not in result["supporting_emails"]:
                        result["supporting_emails"].append(id_part)
            elif current_section == "missing_information":
                result["missing_information"] += " " + line
        
        # If we couldn't extract structured data, use the whole text as the answer
        if not result["answer"]:
            result["answer"] = text.strip()
        
        return result