"""
Configuration module for MCP Expert Chatbot
Handles environment variables and application settings
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Optional
from pydantic import BaseSettings, Field, validator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    """Application settings from environment variables"""
    
    # Gemini API Configuration
    gemini_api_key: Optional[str] = Field(None, env="GEMINI_API_KEY")
    
    # Server Configuration
    host: str = Field("localhost", env="HOST")
    port: int = Field(8000, env="PORT")
    debug: bool = Field(True, env="DEBUG")
    log_level: str = Field("INFO", env="LOG_LEVEL")
    
    # CORS Settings
    allowed_origins: List[str] = Field(
        ["http://localhost:3000", "http://127.0.0.1:3000"],
        env="ALLOWED_ORIGINS"
    )
    
    # Database/Storage Configuration
    chroma_persist_directory: str = Field("./chroma_db", env="CHROMA_PERSIST_DIRECTORY")
    
    # Rate Limiting
    rate_limit: int = Field(60, env="RATE_LIMIT")  # requests per minute
    
    # Cache Settings
    enable_cache: bool = Field(True, env="ENABLE_CACHE")
    cache_ttl: int = Field(3600, env="CACHE_TTL")  # seconds
    
    # Model Settings
    default_model: str = Field("gemini-pro", env="DEFAULT_MODEL")
    max_tokens: int = Field(2048, env="MAX_TOKENS")
    temperature: float = Field(0.7, env="TEMPERATURE")
    
    # Security
    secret_key: str = Field("your_secret_key_here", env="SECRET_KEY")
    
    # Optional: Analytics/Monitoring
    enable_analytics: bool = Field(False, env="ENABLE_ANALYTICS")
    analytics_endpoint: Optional[str] = Field(None, env="ANALYTICS_ENDPOINT")
    
    # Optional: External Services
    webhook_url: Optional[str] = Field(None, env="WEBHOOK_URL")
    slack_token: Optional[str] = Field(None, env="SLACK_TOKEN")
    
    # Knowledge Base Settings
    embedding_model: str = Field("all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    chunk_size: int = Field(500, env="CHUNK_SIZE")
    chunk_overlap: int = Field(50, env="CHUNK_OVERLAP")
    max_search_results: int = Field(5, env="MAX_SEARCH_RESULTS")
    
    # Response Settings
    max_response_length: int = Field(4000, env="MAX_RESPONSE_LENGTH")
    include_sources: bool = Field(True, env="INCLUDE_SOURCES")
    
    @validator("allowed_origins", pre=True)
    def parse_origins(cls, v):
        """Parse comma-separated origins string"""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @validator("log_level")
    def validate_log_level(cls, v):
        """Validate log level"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}")
        return v.upper()
    
    @validator("temperature")
    def validate_temperature(cls, v):
        """Validate temperature range"""
        if not 0 <= v <= 1:
            raise ValueError("Temperature must be between 0 and 1")
        return v
    
    @validator("rate_limit")
    def validate_rate_limit(cls, v):
        """Validate rate limit"""
        if v <= 0:
            raise ValueError("Rate limit must be positive")
        return v
    
    @property
    def is_production(self) -> bool:
        """Check if running in production"""
        return not self.debug
    
    @property
    def gemini_configured(self) -> bool:
        """Check if Gemini API is configured"""
        return bool(self.gemini_api_key and self.gemini_api_key != "your_gemini_api_key_here")
    
    @property
    def database_path(self) -> Path:
        """Get database path as Path object"""
        return Path(self.chroma_persist_directory)
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
        logging.basicConfig(
            level=getattr(logging, self.log_level),
            format=log_format,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler("mcp_expert.log") if self.is_production else logging.NullHandler()
            ]
        )
        
        # Set specific loggers
        logging.getLogger("uvicorn").setLevel(logging.INFO)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        
        if self.debug:
            logging.getLogger("backend").setLevel(logging.DEBUG)
    
    def validate_configuration(self) -> List[str]:
        """Validate configuration and return list of warnings/errors"""
        issues = []
        
        if not self.gemini_configured:
            issues.append("GEMINI_API_KEY is not configured")
        
        if self.secret_key == "your_secret_key_here":
            issues.append("SECRET_KEY should be changed from default value")
        
        if not self.database_path.parent.exists():
            try:
                self.database_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                issues.append(f"Cannot create database directory: {e}")
        
        if self.is_production and self.debug:
            issues.append("Debug mode should be disabled in production")
        
        return issues
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

# Global settings instance
settings = Settings()

# Additional configuration constants
class Constants:
    """Application constants"""
    
    # API Endpoints
    HEALTH_ENDPOINT = "/health"
    CHAT_ENDPOINT = "/chat"
    EXAMPLES_ENDPOINT = "/examples"
    STATS_ENDPOINT = "/stats"
    
    # Response Types
    RESPONSE_SUCCESS = "success"
    RESPONSE_ERROR = "error"
    RESPONSE_WARNING = "warning"
    
    # MCP Knowledge Categories
    MCP_CATEGORIES = [
        "overview",
        "architecture",
        "tools",
        "resources",
        "prompts",
        "transport",
        "client",
        "server",
        "implementation",
        "debugging",
        "best-practices",
        "examples"
    ]
    
    # Difficulty Levels
    DIFFICULTY_LEVELS = ["beginner", "intermediate", "advanced"]
    
    # Supported Languages
    SUPPORTED_LANGUAGES = ["python", "javascript", "typescript", "json", "bash"]
    
    # Cache Keys
    CACHE_EMBEDDING_PREFIX = "embedding:"
    CACHE_RESPONSE_PREFIX = "response:"
    CACHE_SEARCH_PREFIX = "search:"

def get_config_summary() -> dict:
    """Get a summary of current configuration"""
    return {
        "gemini_configured": settings.gemini_configured,
        "debug_mode": settings.debug,
        "log_level": settings.log_level,
        "host": settings.host,
        "port": settings.port,
        "database_path": str(settings.database_path),
        "cache_enabled": settings.enable_cache,
        "rate_limit": settings.rate_limit,
        "model": settings.default_model,
        "max_tokens": settings.max_tokens,
        "temperature": settings.temperature
    }

def load_configuration():
    """Load and validate configuration at startup"""
    logger.info("Loading MCP Expert Chatbot configuration...")
    
    # Setup logging
    settings.setup_logging()
    
    # Validate configuration
    issues = settings.validate_configuration()
    
    if issues:
        logger.warning("Configuration issues found:")
        for issue in issues:
            logger.warning(f"  - {issue}")
    
    # Log configuration summary
    config_summary = get_config_summary()
    logger.info("Configuration loaded:")
    for key, value in config_summary.items():
        logger.info(f"  {key}: {value}")
    
    return settings

# Environment detection
def is_docker() -> bool:
    """Check if running in Docker"""
    return os.path.exists("/.dockerenv")

def is_kubernetes() -> bool:
    """Check if running in Kubernetes"""
    return "KUBERNETES_SERVICE_HOST" in os.environ

def get_environment_info() -> dict:
    """Get information about the runtime environment"""
    return {
        "python_version": sys.version,
        "platform": os.name,
        "working_directory": os.getcwd(),
        "is_docker": is_docker(),
        "is_kubernetes": is_kubernetes(),
        "environment_variables": {
            key: value for key, value in os.environ.items()
            if not key.upper().endswith(("KEY", "TOKEN", "SECRET", "PASSWORD"))
        }
    }

if __name__ == "__main__":
    # Test configuration loading
    config = load_configuration()
    
    print("\n=== MCP Expert Chatbot Configuration ===")
    print(f"Debug Mode: {config.debug}")
    print(f"Gemini Configured: {config.gemini_configured}")
    print(f"Host: {config.host}:{config.port}")
    print(f"Database: {config.database_path}")
    print(f"Log Level: {config.log_level}")
    
    # Check for issues
    issues = config.validate_configuration()
    if issues:
        print(f"\n⚠️  Issues found: {len(issues)}")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✅ Configuration is valid")
    
    # Environment info
    env_info = get_environment_info()
    print(f"\n📊 Runtime Environment:")
    print(f"  Platform: {env_info['platform']}")
    print(f"  Docker: {env_info['is_docker']}")
    print(f"  Kubernetes: {env_info['is_kubernetes']}")
    print(f"  Working Dir: {env_info['working_directory']}") 