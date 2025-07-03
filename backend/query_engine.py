import google.generativeai as genai
from typing import Dict, List, Any, Optional
import logging

from backend.embeddings import EmbeddingManager

logger = logging.getLogger(__name__)

class MCPQueryEngine:
    def __init__(self, embedding_manager: EmbeddingManager):
        self.model = genai.GenerativeModel('gemini-1.5-pro')
        self.embedding_manager = embedding_manager

    async def process_query(self, query: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a user query
        
        Args:
            query: The user's query
            context: Optional conversational context
            
        Returns:
            Dictionary with response and extracted code samples
        """
        logger.info(f"Processing query: {query}")
        
        # Retrieve relevant documents from knowledge base
        relevant_docs = self._retrieve_relevant_docs(query)
        
        # Construct prompt for Gemini
        prompt = self._construct_prompt(query, relevant_docs, context)
        
        # Generate response
        response = await self._generate_response(prompt)
        
        # Extract code samples
        code_samples = self._extract_code_samples(response)
        
        return {
            "response": response,
            "code_samples": code_samples,
            "references": [doc.get('metadata', {}).get('topic', 'N/A') for doc in relevant_docs]
        }
    
    def _retrieve_relevant_docs(self, query: str, n_results: int = 5) -> List[Dict]:
        """Retrieve relevant documents from ChromaDB"""
        try:
            results = self.embedding_manager.search_similar(query, n_results=n_results)
            logger.debug(f"Retrieved {len(results)} documents for query: {query}")
            return results
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
    
    def _construct_prompt(self, query: str, relevant_docs: List[Dict], context: Optional[str]) -> str:
        """Construct the prompt for the generative model"""
        
        # System prompt
        system_prompt = """You are an expert on the Model Context Protocol (MCP).
Your goal is to provide accurate, helpful, and concise answers to user queries about MCP.
Use the provided context from the knowledge base to inform your response.
If the context doesn't contain the answer, state that you don't have enough information.
Format your responses clearly, using markdown for code blocks where appropriate."""

        # Context from documents
        doc_context = "\n\n".join(
            f"Topic: {doc.get('metadata', {}).get('topic', 'N/A')}\nContent: {doc.get('document', 'N/A')}"
            for doc in relevant_docs
        )

        # Final prompt construction
        prompt = f"{system_prompt}\n\n"
        if relevant_docs:
            prompt += f"**Relevant Knowledge Base Context:**\n{doc_context}\n\n"
        
        if context:
            prompt += f"**Previous Conversation Context:**\n{context}\n\n"
            
        prompt += f"**User Query:** {query}\n\n**Your Answer:**"
        
        return prompt
    
    async def _generate_response(self, prompt: str) -> str:
        """Generate response using Gemini"""
        try:
            # Setting safety settings to be less restrictive
            safety_settings = {
                'HATE_SPEECH': 'BLOCK_NONE',
                'HARASSMENT': 'BLOCK_NONE',
                'SEXUALLY_EXPLICIT': 'BLOCK_NONE',
                'DANGEROUS_CONTENT': 'BLOCK_NONE'
            }
            response = await self.model.generate_content_async(
                prompt,
                safety_settings=safety_settings
            )
            return response.text
        except Exception as e:
            logger.error(f"Error generating response from Gemini: {e}")
            return "Sorry, I encountered an error while generating a response."
    
    def _extract_code_samples(self, response: str) -> List[Dict]:
        """Extract code samples from the response text"""
        code_samples = []
        in_code_block = False
        current_code = ""
        current_lang = "python" # Default language
        
        for line in response.split('\n'):
            if line.strip().startswith("```"):
                if not in_code_block:
                    in_code_block = True
                    current_lang = line.strip()[3:] or "python"
                else:
                    in_code_block = False
                    if current_code:
                        code_samples.append({
                            "language": current_lang,
                            "code": current_code.strip()
                        })
                        current_code = ""
            elif in_code_block:
                current_code += line + "\n"
        
        return code_samples 