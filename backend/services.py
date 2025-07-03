from backend.embeddings import EmbeddingManager
from backend.query_engine import MCPQueryEngine
from backend.config import settings

embedding_manager = EmbeddingManager(
    persist_directory=str(settings.database_path),
    model_name=settings.embedding_model
)
query_engine = MCPQueryEngine(embedding_manager) 