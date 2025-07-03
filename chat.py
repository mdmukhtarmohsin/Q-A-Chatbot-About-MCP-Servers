#!/usr/bin/env python3
"""
MCP Expert Chatbot - CLI Interface
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add backend to Python path
sys.path.insert(0, str(Path(__file__).parent))

from backend.embeddings import EmbeddingManager
from backend.query_engine import MCPQueryEngine

# Set up logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

async def main():
    """Main function for the CLI chat interface"""
    print("🤖 MCP Expert Chatbot (CLI Mode)")
    print("Type 'exit' or 'quit' to end the chat.")
    print("-" * 40)

    try:
        # Initialize the embedding manager and query engine
        embedding_manager = EmbeddingManager()
        query_engine = MCPQueryEngine(embedding_manager)

        while True:
            try:
                query = input("You: ")
                if query.lower() in ["exit", "quit"]:
                    break

                if not query.strip():
                    continue

                # Process the query
                result = await query_engine.process_query(query)

                # Print the response
                print(f"\n🤖 Bot: {result['response']}\n")

            except (KeyboardInterrupt, EOFError):
                break

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        print("Sorry, an error occurred. Please try restarting the application.")

    finally:
        print("\n👋 Goodbye!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass 