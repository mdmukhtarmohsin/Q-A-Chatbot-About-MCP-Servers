"""
Embeddings module for MCP Expert Chatbot
Handles document embedding, chunking, and similarity search
"""

import json
import logging
import os
from typing import List, Dict, Any, Optional, Tuple
import hashlib
import pickle
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

class EmbeddingManager:
    def __init__(self, persist_directory: str = "./chroma_db", model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding manager
        
        Args:
            persist_directory: Directory to persist ChromaDB data
            model_name: SentenceTransformer model name
        """
        self.persist_directory = persist_directory
        self.model_name = model_name
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        
        # Collection name
        self.collection_name = "mcp_knowledge"
        
        # Initialize or get collection
        self.collection = self._get_or_create_collection()
        
        # Cache for embeddings
        self.embedding_cache = {}
        self._load_embedding_cache()
    
    def _get_or_create_collection(self):
        """Get existing collection or create new one"""
        try:
            collection = self.chroma_client.get_collection(self.collection_name)
            logger.info(f"Using existing collection: {self.collection_name}")
            return collection
        except Exception:
            logger.info(f"Creating new collection: {self.collection_name}")
            return self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"description": "MCP knowledge base embeddings"}
            )
    
    def _load_embedding_cache(self):
        """Load embedding cache from disk"""
        cache_file = Path(self.persist_directory) / "embedding_cache.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    self.embedding_cache = pickle.load(f)
                logger.info(f"Loaded {len(self.embedding_cache)} cached embeddings")
            except Exception as e:
                logger.error(f"Error loading embedding cache: {e}")
                self.embedding_cache = {}
    
    def _save_embedding_cache(self):
        """Save embedding cache to disk"""
        cache_file = Path(self.persist_directory) / "embedding_cache.pkl"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
            logger.info(f"Saved {len(self.embedding_cache)} embeddings to cache")
        except Exception as e:
            logger.error(f"Error saving embedding cache: {e}")
    
    def _get_text_hash(self, text: str) -> str:
        """Get hash for text to use as cache key"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def embed_text(self, text: str, use_cache: bool = True) -> List[float]:
        """
        Generate embedding for text with caching
        
        Args:
            text: Text to embed
            use_cache: Whether to use cached embeddings
            
        Returns:
            Embedding vector as list of floats
        """
        if use_cache:
            text_hash = self._get_text_hash(text)
            if text_hash in self.embedding_cache:
                return self.embedding_cache[text_hash]
        
        # Generate embedding
        embedding = self.embedding_model.encode(text).tolist()
        
        # Cache the embedding
        if use_cache:
            text_hash = self._get_text_hash(text)
            self.embedding_cache[text_hash] = embedding
        
        return embedding
    
    def embed_texts(self, texts: List[str], use_cache: bool = True) -> List[List[float]]:
        """
        Generate embeddings for multiple texts
        
        Args:
            texts: List of texts to embed
            use_cache: Whether to use cached embeddings
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            if use_cache:
                text_hash = self._get_text_hash(text)
                if text_hash in self.embedding_cache:
                    embeddings.append(self.embedding_cache[text_hash])
                    continue
            
            # Track uncached texts
            uncached_texts.append(text)
            uncached_indices.append(i)
            embeddings.append(None)  # Placeholder
        
        # Generate embeddings for uncached texts
        if uncached_texts:
            logger.info(f"Generating embeddings for {len(uncached_texts)} new texts")
            new_embeddings = self.embedding_model.encode(uncached_texts).tolist()
            
            # Insert new embeddings and cache them
            for idx, embedding in zip(uncached_indices, new_embeddings):
                embeddings[idx] = embedding
                
                if use_cache:
                    text_hash = self._get_text_hash(texts[idx])
                    self.embedding_cache[text_hash] = embedding
        
        return embeddings
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Text to chunk
            chunk_size: Maximum chunk size in characters
            overlap: Overlap between chunks in characters
            
        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundaries
            if end < len(text):
                # Look for sentence endings
                sentence_ends = ['.', '!', '?', '\n\n']
                for i in range(end - 50, start, -1):
                    if text[i] in sentence_ends:
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - overlap
            if start >= len(text):
                break
        
        return chunks
    
    def add_documents(self, documents: List[Dict[str, Any]], chunk_size: int = 500) -> None:
        """
        Add documents to the vector database
        
        Args:
            documents: List of document dictionaries with 'content', 'topic', etc.
            chunk_size: Size for text chunking
        """
        all_chunks = []
        all_metadatas = []
        all_ids = []
        
        for doc_idx, doc in enumerate(documents):
            content = doc.get('content', '')
            base_metadata = {
                'topic': doc.get('topic', 'Unknown'),
                'type': doc.get('type', 'general'),
                'difficulty': doc.get('difficulty', 'beginner'),
                'doc_index': doc_idx
            }
            
            # Chunk the document content
            chunks = self.chunk_text(content, chunk_size)
            
            for chunk_idx, chunk in enumerate(chunks):
                chunk_id = f"doc_{doc_idx}_chunk_{chunk_idx}"
                chunk_metadata = {
                    **base_metadata,
                    'chunk_index': chunk_idx,
                    'total_chunks': len(chunks)
                }
                
                all_chunks.append(chunk)
                all_metadatas.append(chunk_metadata)
                all_ids.append(chunk_id)
        
        logger.info(f"Adding {len(all_chunks)} chunks to vector database")
        
        # Generate embeddings
        embeddings = self.embed_texts(all_chunks)
        
        # Add to ChromaDB
        self.collection.add(
            documents=all_chunks,
            metadatas=all_metadatas,
            ids=all_ids,
            embeddings=embeddings
        )
        
        # Save cache
        self._save_embedding_cache()
        
        logger.info(f"Successfully added {len(all_chunks)} chunks from {len(documents)} documents")
    
    def search_similar(self, query: str, n_results: int = 5, filter_metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents
        
        Args:
            query: Search query
            n_results: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of search results with content, metadata, and similarity scores
        """
        try:
            # Generate query embedding
            query_embedding = self.embed_text(query)
            
            # Search in ChromaDB
            search_kwargs = {
                'query_embeddings': [query_embedding],
                'n_results': n_results
            }
            
            if filter_metadata:
                search_kwargs['where'] = filter_metadata
            
            results = self.collection.query(**search_kwargs)
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    result = {
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i] if results.get('distances') else 0,
                        'similarity': 1 - (results['distances'][0][i] if results.get('distances') else 0)
                    }
                    formatted_results.append(result)
            
            logger.info(f"Found {len(formatted_results)} similar documents for query")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        try:
            count = self.collection.count()
            return {
                'total_documents': count,
                'collection_name': self.collection_name,
                'model_name': self.model_name,
                'persist_directory': self.persist_directory,
                'cached_embeddings': len(self.embedding_cache)
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}
    
    def clear_collection(self) -> None:
        """Clear all documents from the collection"""
        try:
            # Delete the collection
            self.chroma_client.delete_collection(self.collection_name)
            
            # Recreate empty collection
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"description": "MCP knowledge base embeddings"}
            )
            
            # Clear cache
            self.embedding_cache = {}
            self._save_embedding_cache()
            
            logger.info("Collection cleared successfully")
            
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
    
    def export_embeddings(self, filepath: str) -> None:
        """Export embeddings to file"""
        try:
            # Get all documents
            results = self.collection.get()
            
            export_data = {
                'documents': results['documents'],
                'metadatas': results['metadatas'],
                'ids': results['ids'],
                'embeddings': results.get('embeddings', []),
                'collection_stats': self.get_collection_stats()
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Embeddings exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting embeddings: {e}")
    
    def __del__(self):
        """Cleanup on deletion"""
        try:
            self._save_embedding_cache()
        except Exception:
            pass  # Ignore errors during cleanup

def load_mcp_knowledge(filepath: str = "backend/mcp_kb.json") -> List[Dict[str, Any]]:
    """Load MCP knowledge from JSON file"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"Knowledge file {filepath} not found")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing knowledge file: {e}")
        return []

def initialize_knowledge_base(embedding_manager: EmbeddingManager, force_rebuild: bool = False) -> None:
    """Initialize the knowledge base with MCP documents"""
    
    # Check if collection already has documents
    stats = embedding_manager.get_collection_stats()
    if stats.get('total_documents', 0) > 0 and not force_rebuild:
        logger.info(f"Knowledge base already initialized with {stats['total_documents']} documents")
        return
    
    if force_rebuild:
        logger.info("Force rebuilding knowledge base...")
        embedding_manager.clear_collection()
    
    # Load MCP knowledge
    knowledge_data = load_mcp_knowledge()
    
    if not knowledge_data:
        logger.warning("No knowledge data found to initialize")
        return
    
    # Add documents to embedding manager
    embedding_manager.add_documents(knowledge_data)
    
    # Log final stats
    final_stats = embedding_manager.get_collection_stats()
    logger.info(f"Knowledge base initialized: {final_stats}")

if __name__ == "__main__":
    # Test the embedding manager
    logging.basicConfig(level=logging.INFO)
    
    # Initialize embedding manager
    em = EmbeddingManager()
    
    # Initialize knowledge base
    initialize_knowledge_base(em, force_rebuild=True)
    
    # Test search
    results = em.search_similar("How do I create an MCP server?", n_results=3)
    
    print("\n=== Search Results ===")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Topic: {result['metadata']['topic']}")
        print(f"   Similarity: {result['similarity']:.3f}")
        print(f"   Content: {result['content'][:200]}...")
    
    # Print stats
    stats = em.get_collection_stats()
    print(f"\n=== Collection Stats ===")
    for key, value in stats.items():
        print(f"{key}: {value}") 