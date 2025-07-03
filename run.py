#!/usr/bin/env python3
"""
MCP Expert Chatbot - Main Runner
Entry point for the MCP Expert Chatbot application
"""

import sys
import os
import asyncio
import logging
import signal
import webbrowser
from pathlib import Path
from typing import Optional

# Add backend to Python path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

try:
    import uvicorn
    from fastapi import FastAPI
    
    from backend.config import load_configuration, settings, get_environment_info
    from backend.main import app
    from backend.embeddings import EmbeddingManager, initialize_knowledge_base
    
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("Please install requirements: pip install -r requirements.txt")
    sys.exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MCPExpertServer:
    """Main server class for MCP Expert Chatbot"""
    
    def __init__(self):
        self.config = None
        self.server = None
        self.embedding_manager = None
        self.is_running = False
        
    async def startup_checks(self) -> bool:
        """Perform startup checks and initialization"""
        logger.info("🚀 Starting MCP Expert Chatbot...")
        
        try:
            # Load configuration
            self.config = load_configuration()
            logger.info("✅ Configuration loaded")
            
            # Check configuration issues
            issues = self.config.validate_configuration()
            if issues:
                logger.warning("⚠️  Configuration issues found:")
                for issue in issues:
                    logger.warning(f"   {issue}")
            
            # Initialize embedding manager
            logger.info("🧠 Initializing embedding manager...")
            self.embedding_manager = EmbeddingManager(
                persist_directory=str(self.config.database_path),
                model_name=self.config.embedding_model
            )
            
            # Initialize knowledge base
            logger.info("📚 Initializing knowledge base...")
            initialize_knowledge_base(self.embedding_manager)
            
            # Get stats
            stats = self.embedding_manager.get_collection_stats()
            logger.info(f"✅ Knowledge base ready: {stats.get('total_documents', 0)} documents")
            
            # Check Gemini configuration
            if not self.config.gemini_configured:
                logger.error("❌ Gemini API key not configured!")
                logger.error("   Please set GEMINI_API_KEY environment variable")
                return False
            
            logger.info("✅ Gemini API configured")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Startup failed: {e}")
            return False
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"📥 Received signal {signum}, shutting down...")
            self.shutdown()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def start_server(self):
        """Start the FastAPI server"""
        try:
            # Configure uvicorn
            config = uvicorn.Config(
                app=app,
                host=self.config.host,
                port=self.config.port,
                log_level=self.config.log_level.lower(),
                reload=self.config.debug,
                access_log=self.config.debug
            )
            
            # Create server
            self.server = uvicorn.Server(config)
            
            logger.info(f"🌐 Starting server on http://{self.config.host}:{self.config.port}")
            
            # Start server
            self.is_running = True
            await self.server.serve()
            
        except Exception as e:
            logger.error(f"❌ Server startup failed: {e}")
            self.is_running = False
    
    def shutdown(self):
        """Graceful shutdown"""
        logger.info("🛑 Shutting down MCP Expert Chatbot...")
        
        self.is_running = False
        
        if self.server:
            self.server.should_exit = True
        
        # Save embedding cache if available
        if self.embedding_manager:
            try:
                self.embedding_manager._save_embedding_cache()
                logger.info("💾 Embedding cache saved")
            except Exception as e:
                logger.error(f"Failed to save embedding cache: {e}")
        
        logger.info("✅ Shutdown complete")
    
    async def run(self):
        """Main run method"""
        # Setup signal handlers
        self.setup_signal_handlers()
        
        # Perform startup checks
        if not await self.startup_checks():
            logger.error("❌ Startup checks failed, exiting...")
            return False
        
        # Print startup information
        self.print_startup_info()
        
        # Start server
        await self.start_server()
        
        return True
    
    def print_startup_info(self):
        """Print startup information"""
        print("\n" + "="*60)
        print("🤖 MCP Expert Chatbot")
        print("="*60)
        print(f"🌐 Server: http://{self.config.host}:{self.config.port}")
        print(f"📚 Knowledge Base: {self.embedding_manager.get_collection_stats().get('total_documents', 0)} documents")
        print(f"🧠 Model: {self.config.default_model}")
        print(f"🔧 Debug Mode: {self.config.debug}")
        print(f"📊 Log Level: {self.config.log_level}")
        print("="*60)
        print("\n💡 Try these example queries:")
        print("   • What is MCP and how does it work?")
        print("   • Give me code for a basic MCP server")
        print("   • How do I handle errors in MCP tools?")
        print("   • What are MCP best practices?")
        print("\n⌨️  Press Ctrl+C to stop the server")
        print("-"*60 + "\n")

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        print(f"Current version: {sys.version}")
        return False
    return True

def check_dependencies():
    """Check if all required dependencies are available"""
    required_packages = [
        "fastapi",
        "uvicorn",
        "google-generativeai",
        "chromadb",
        "sentence-transformers",
        "pydantic"
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing.append(package)
    
    if missing:
        print("❌ Missing required packages:")
        for package in missing:
            print(f"   {package}")
        print("\nInstall with: pip install -r requirements.txt")
        return False
    
    return True

def open_browser(url: str, delay: int = 2):
    """Open browser after server starts"""
    async def delayed_open():
        await asyncio.sleep(delay)
        try:
            webbrowser.open(url)
            logger.info(f"🌐 Opened browser: {url}")
        except Exception as e:
            logger.warning(f"Could not open browser: {e}")
    
    return delayed_open()

async def main():
    """Main entry point"""
    print("🤖 MCP Expert Chatbot - Starting up...\n")
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Create and run server
    server = MCPExpertServer()
    
    try:
        # Option to open browser
        if "--no-browser" not in sys.argv:
            asyncio.create_task(open_browser(f"http://{settings.host}:{settings.port}"))
        
        # Run server
        success = await server.run()
        return 0 if success else 1
        
    except KeyboardInterrupt:
        logger.info("👋 Received keyboard interrupt")
        server.shutdown()
        return 0
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return 1

def cli():
    """Command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MCP Expert Chatbot")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser")
    parser.add_argument("--rebuild-kb", action="store_true", help="Rebuild knowledge base")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--config-check", action="store_true", help="Check configuration and exit")
    
    args = parser.parse_args()
    
    # Override settings from CLI args
    if args.host:
        os.environ["HOST"] = args.host
    if args.port:
        os.environ["PORT"] = str(args.port)
    if args.debug:
        os.environ["DEBUG"] = "true"
    
    # Configuration check only
    if args.config_check:
        config = load_configuration()
        issues = config.validate_configuration()
        
        print("📋 Configuration Check:")
        print(f"   Gemini Configured: {config.gemini_configured}")
        print(f"   Database Path: {config.database_path}")
        print(f"   Debug Mode: {config.debug}")
        
        if issues:
            print(f"\n⚠️  Issues: {len(issues)}")
            for issue in issues:
                print(f"   - {issue}")
            return 1
        else:
            print("\n✅ Configuration is valid")
            return 0
    
    # Rebuild knowledge base
    if args.rebuild_kb:
        print("🔄 Rebuilding knowledge base...")
        config = load_configuration()
        em = EmbeddingManager(str(config.database_path))
        initialize_knowledge_base(em, force_rebuild=True)
        print("✅ Knowledge base rebuilt")
        return 0
    
    # Run main server
    return asyncio.run(main())

if __name__ == "__main__":
    sys.exit(cli()) 