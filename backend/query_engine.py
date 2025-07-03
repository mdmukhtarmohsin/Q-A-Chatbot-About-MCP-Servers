import google.generativeai as genai
import json
import os
from typing import Dict, List, Any, Optional
import chromadb
from chromadb.config import Settings
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

class MCPQueryEngine:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-pro')
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.Client(Settings(
            persist_directory="./chroma_db",
            anonymized_telemetry=False
        ))
        
        # Get or create collection
        try:
            self.collection = self.chroma_client.get_collection("mcp_knowledge")
        except:
            self.collection = self.chroma_client.create_collection("mcp_knowledge")
            self._populate_knowledge_base()
    
    def _populate_knowledge_base(self):
        """Populate the ChromaDB with MCP knowledge"""
        logger.info("Populating MCP knowledge base...")
        
        # Load MCP knowledge from JSON file
        knowledge_file = "backend/mcp_kb.json"
        if os.path.exists(knowledge_file):
            with open(knowledge_file, 'r') as f:
                mcp_knowledge = json.load(f)
        else:
            # Fallback to hardcoded knowledge
            mcp_knowledge = self._get_default_knowledge()
        
        # Process and embed knowledge chunks
        documents = []
        metadatas = []
        ids = []
        
        for i, item in enumerate(mcp_knowledge):
            documents.append(item["content"])
            metadatas.append({
                "topic": item["topic"],
                "type": item.get("type", "general"),
                "difficulty": item.get("difficulty", "beginner")
            })
            ids.append(f"doc_{i}")
        
        # Add to collection
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"Added {len(documents)} documents to knowledge base")
    
    def _get_default_knowledge(self) -> List[Dict]:
        """Default MCP knowledge if JSON file doesn't exist"""
        return [
            {
                "topic": "MCP Overview",
                "type": "fundamental",
                "difficulty": "beginner",
                "content": """
Model Context Protocol (MCP) is a standardized protocol for enabling Large Language Models (LLMs) to interact with external tools and data sources. Unlike OpenAI's function calling which is tightly coupled to their API, MCP provides a vendor-neutral approach that can work with any LLM.

Key differences from OpenAI function calling:
1. Protocol Independence: MCP is not tied to any specific LLM provider
2. Bidirectional Communication: Supports both tool calls and data retrieval
3. Resource Management: Can handle persistent connections and state
4. Standardized Schema: Uses JSON Schema for tool definitions

MCP consists of three main components:
- MCP Server: Exposes tools and resources
- MCP Client: Consumes tools (usually the LLM application)
- Transport Layer: Handles communication (stdio, HTTP, WebSocket)
                """
            },
            {
                "topic": "Basic MCP Server",
                "type": "implementation",
                "difficulty": "beginner",
                "content": """
A basic MCP server is implemented by extending the Server class and registering tools using decorators:

```python
from mcp.server import Server
from mcp.types import Tool

server = Server("my-mcp-server")

@server.tool()
async def calculate_sum(a: int, b: int) -> int:
    \"\"\"Calculate the sum of two numbers\"\"\"
    return a + b

@server.tool()
async def get_user_info(user_id: str) -> dict:
    \"\"\"Get user information by ID\"\"\"
    # Your implementation here
    return {"user_id": user_id, "name": "John Doe"}

if __name__ == "__main__":
    import asyncio
    asyncio.run(server.run())
```

The @server.tool() decorator automatically registers the function as an MCP tool, using the function signature for schema generation.
                """
            },
            {
                "topic": "Tool Registration and Schema",
                "type": "implementation",
                "difficulty": "intermediate",
                "content": """
Proper tool registration requires defining clear schemas for parameters and return values:

```python
from mcp.types import Tool, ToolParameter

# Method 1: Explicit schema definition
weather_tool = Tool(
    name="get_weather",
    description="Get current weather for a location",
    parameters={
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "The city name"
            },
            "units": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "default": "celsius"
            }
        },
        "required": ["city"]
    }
)

# Method 2: Using type hints (recommended)
@server.tool()
async def get_weather(city: str, units: str = "celsius") -> dict:
    \"\"\"Get current weather for a location
    
    Args:
        city: The city name
        units: Temperature units (celsius or fahrenheit)
    \"\"\"
    # Implementation here
    pass
```

The schema ensures proper parameter validation and helps the LLM understand how to use your tools.
                """
            },
            {
                "topic": "Error Handling and Debugging",
                "type": "troubleshooting",
                "difficulty": "intermediate",
                "content": """
Common MCP issues and solutions:

1. Tool returns 'null' or empty output:
   - Check parameter serialization
   - Verify schema matches function signature
   - Ensure proper async/await usage
   - Check return type annotations

2. Connection issues:
   - Verify transport configuration (stdio vs HTTP)
   - Check server is properly started
   - Validate client connection settings

3. Schema validation errors:
   - Use JSON Schema validator
   - Check required vs optional parameters
   - Verify enum values match
   - Ensure proper type annotations

4. Debugging techniques:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

@server.tool()
async def debug_tool(param: str) -> str:
    logger.debug(f"Received param: {param}")
    try:
        result = process_param(param)
        logger.debug(f"Returning: {result}")
        return result
    except Exception as e:
        logger.error(f"Error: {e}")
        raise
```
                """
            },
            {
                "topic": "Context and State Management",
                "type": "advanced",
                "difficulty": "advanced",
                "content": """
Managing context between tool calls in MCP:

```python
class StatefulMCPServer:
    def __init__(self):
        self.server = Server("stateful-server")
        self.context = {}
        self.register_tools()
    
    def register_tools(self):
        @self.server.tool()
        async def set_context(key: str, value: str) -> str:
            \"\"\"Set a context value\"\"\"
            self.context[key] = value
            return f"Set {key} = {value}"
        
        @self.server.tool()
        async def get_context(key: str) -> str:
            \"\"\"Get a context value\"\"\"
            return self.context.get(key, "Not found")
        
        @self.server.tool()
        async def process_with_context(data: str) -> str:
            \"\"\"Process data using stored context\"\"\"
            user_pref = self.context.get("user_preference", "default")
            return f"Processed {data} with preference: {user_pref}"

# Usage pattern for multi-step workflows
server = StatefulMCPServer()
```

This allows the LLM to maintain state across multiple tool invocations within a conversation.
                """
            }
        ]
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for Gemini"""
        return """You are an expert on Model Context Protocol (MCP) used in LLM systems for tool invocation. 
You act like a senior developer mentoring a teammate about MCP development.

When answering questions:
- Provide clear, practical explanations
- Include runnable code samples (Python preferred)
- Offer troubleshooting steps when relevant
- Explain best practices and common pitfalls
- Be honest about limitations and admit when unsure

Focus specifically on MCP concepts:
- Server and client implementation
- Tool registration and schemas
- Transport protocols (stdio, HTTP, WebSocket)
- Error handling and debugging
- Context management between tool calls
- Integration patterns with different LLMs

Format your responses with:
1. Clear explanation of the concept
2. Code examples with comments
3. Common issues and solutions (if relevant)
4. Best practices

Avoid hallucinating information about MCP. If you're unsure about something specific, say so and provide general guidance instead."""

    async def process_query(self, query: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Process a user query using RAG + Gemini"""
        try:
            # Retrieve relevant documents
            relevant_docs = self._retrieve_relevant_docs(query)
            
            # Construct prompt with context
            prompt = self._construct_prompt(query, relevant_docs, context)
            
            # Generate response with Gemini
            response = await self._generate_response(prompt)
            
            # Extract code samples and references
            code_samples = self._extract_code_samples(response)
            references = [doc["metadata"]["topic"] for doc in relevant_docs]
            
            return {
                "response": response,
                "code_samples": code_samples,
                "references": references
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "response": f"I apologize, but I encountered an error processing your query: {str(e)}. Please try rephrasing your question about MCP.",
                "code_samples": [],
                "references": []
            }
    
    def _retrieve_relevant_docs(self, query: str, n_results: int = 3) -> List[Dict]:
        """Retrieve relevant documents from ChromaDB"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            relevant_docs = []
            for i in range(len(results["documents"][0])):
                relevant_docs.append({
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i] if "distances" in results else 0
                })
            
            return relevant_docs
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return []
    
    def _construct_prompt(self, query: str, relevant_docs: List[Dict], context: Optional[str]) -> str:
        """Construct the full prompt for Gemini"""
        system_prompt = self._get_system_prompt()
        
        # Add relevant documentation
        doc_context = "\n\n".join([
            f"=== {doc['metadata']['topic']} ===\n{doc['content']}"
            for doc in relevant_docs
        ])
        
        prompt_parts = [
            system_prompt,
            "\n\n=== RELEVANT MCP DOCUMENTATION ===",
            doc_context,
        ]
        
        if context:
            prompt_parts.extend([
                "\n\n=== USER CONTEXT ===",
                context
            ])
        
        prompt_parts.extend([
            "\n\n=== USER QUESTION ===",
            query,
            "\n\nPlease provide a helpful response based on the MCP documentation above."
        ])
        
        return "\n".join(prompt_parts)
    
    async def _generate_response(self, prompt: str) -> str:
        """Generate response using Gemini"""
        try:
            response = await self.model.generate_content_async(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"I apologize, but I'm having trouble generating a response right now. Please try again in a moment."
    
    def _extract_code_samples(self, response: str) -> List[Dict]:
        """Extract code samples from the response"""
        code_samples = []
        lines = response.split('\n')
        in_code_block = False
        current_code = []
        current_language = ""
        
        for line in lines:
            if line.startswith('```'):
                if in_code_block:
                    # End of code block
                    if current_code:
                        code_samples.append({
                            "language": current_language,
                            "code": '\n'.join(current_code)
                        })
                    current_code = []
                    current_language = ""
                    in_code_block = False
                else:
                    # Start of code block
                    current_language = line[3:].strip() or "python"
                    in_code_block = True
            elif in_code_block:
                current_code.append(line)
        
        return code_samples 