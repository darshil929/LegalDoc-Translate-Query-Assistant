import json
import logging
import traceback
from typing import List, Dict, Optional
from groq import Groq
from weaviate_manager import WeaviateManager, WeaviateConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """Manages RAG operations using Weaviate vector database and Groq LLM"""

    def __init__(
        self,
        weaviate_config: Optional[WeaviateConfig] = None,
        groq_api_key: str = None,
    ):
        """Initializes RAG pipeline with Weaviate and Groq clients"""
        # Initialize Weaviate manager for vector operations
        self.weaviate_manager = WeaviateManager(config=weaviate_config)

        # Initialize Groq client for LLM operations
        self.groq_client = Groq(api_key=groq_api_key)

        # Default parameters for retrieval
        self.default_top_k = 10
        self.min_certainty = 0.0

    def query_weaviate(self, prompt: str, top_k: int = 10) -> List[Dict]:
        """Performs semantic search using Weaviate's built-in vectorization"""
        try:
            # Use Weaviate's text search with automatic vectorization
            results = self.weaviate_manager.search_documents(
                query_text=prompt, limit=top_k, certainty=self.min_certainty
            )

            # Sort results by score (certainty) in descending order
            sorted_results = sorted(
                results, key=lambda x: x.get("score", 0.0), reverse=True
            )

            return sorted_results

        except Exception as e:
            logger.error(f"Weaviate query failed: {str(e)}")
            return []

    def get_context(self, user_prompt: str, top_k: int = 3) -> str:
        """Retrieves and formats top contextual results for LLM"""
        try:
            # Retrieve relevant documents from Weaviate
            search_results = self.query_weaviate(user_prompt, top_k)

            if not search_results:
                logger.warning("No search results found for query")
                return ""

            # Extract and combine text from results
            context_texts = []
            for result in search_results:
                text = result.get("text", "")
                if text:
                    context_texts.append(text)

            # Join contexts with clear separation
            context = "\n\n".join(context_texts)

            return context

        except Exception as e:
            logger.error(f"Context retrieval failed: {str(e)}")
            return ""

    def generate_response(self, user_query: str, context: str) -> Dict:
        """Generates structured response using Groq LLM with retrieved context"""

        # System prompt for legal analysis
        system_prompt = f"""
        You are an expert legal analyst. Provide precise, evidence-based responses.

        Context: {context}
        
        Response Guidelines:
        - Analyze the query using ONLY the provided context
        - Structure response as JSON with:
          1. "answer": Comprehensive legal explanation
          2. "reasoning": Logical breakdown
          3. "confidence_score": 0-1 rating
          4. "key_sources": Relevant context snippets
        - Be concise but thorough
        - Explicitly state if context is insufficient
        """

        try:
            # Generate response using Groq LLM
            response = self.groq_client.chat.completions.create(
                model="llama3-70b-8192",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query},
                ],
                max_tokens=2048,
                temperature=0.3,
            )

            # Parse and return JSON response
            response_content = response.choices[0].message.content
            parsed_response = json.loads(response_content)

            # Validate response structure
            required_fields = ["answer", "reasoning", "confidence_score", "key_sources"]
            for field in required_fields:
                if field not in parsed_response:
                    parsed_response[field] = (
                        "Not available" if field != "confidence_score" else 0.0
                    )

            return parsed_response

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {str(e)}")
            return {
                "answer": "Failed to parse response",
                "reasoning": "JSON parsing error occurred",
                "confidence_score": 0.0,
                "key_sources": [],
                "error": str(e),
            }
        except Exception as e:
            logger.error(f"Response generation failed: {str(e)}")
            return {
                "answer": "Failed to generate response",
                "reasoning": "An error occurred during response generation",
                "confidence_score": 0.0,
                "key_sources": [],
                "error": str(e),
                "trace": traceback.format_exc(),
            }

    def process_query(self, user_query: str, top_k: int = 3) -> Dict:
        """Complete RAG pipeline execution from query to response"""
        try:
            # Step 1: Retrieve relevant context
            context = self.get_context(user_query, top_k)

            if not context:
                return {
                    "answer": "No relevant information found in the database",
                    "reasoning": "The search did not return any relevant documents",
                    "confidence_score": 0.0,
                    "key_sources": [],
                }

            # Step 2: Generate response with context
            response = self.generate_response(user_query, context)

            # Step 3: Add metadata to response
            response["query"] = user_query
            response["documents_retrieved"] = top_k

            return response

        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}")
            return {
                "answer": "Failed to process query",
                "reasoning": "An unexpected error occurred",
                "confidence_score": 0.0,
                "key_sources": [],
                "error": str(e),
            }

    def health_check(self) -> Dict:
        """Checks health status of RAG pipeline components"""
        health_status = {"weaviate": False, "groq": False, "overall": False}

        try:
            # Check Weaviate connection
            weaviate_health = self.weaviate_manager.health_check()
            health_status["weaviate"] = weaviate_health.get("connected", False)

            # Check Groq client (simple validation)
            health_status["groq"] = self.groq_client is not None

            # Overall health
            health_status["overall"] = (
                health_status["weaviate"] and health_status["groq"]
            )

        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")

        return health_status

    def close(self):
        """Closes connections to external services"""
        try:
            if self.weaviate_manager:
                self.weaviate_manager.close()
                logger.info("RAG pipeline connections closed")
        except Exception as e:
            logger.error(f"Error closing connections: {str(e)}")
