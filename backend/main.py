from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import google.generativeai as genai
import os
from typing import List, Optional
import asyncio
import json
from dotenv import load_dotenv
from query_engine import MCPQueryEngine
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI(title="MCP Expert Chatbot", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize query engine
query_engine = MCPQueryEngine()

class ChatMessage(BaseModel):
    message: str
    context: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    code_samples: List[dict] = []
    references: List[str] = []

# Store active WebSocket connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

manager = ConnectionManager()

@app.get("/")
async def root():
    return {"message": "MCP Expert Chatbot API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "gemini_configured": bool(os.getenv("GEMINI_API_KEY"))}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_request: ChatMessage):
    """
    Main chat endpoint that processes user queries about MCP
    """
    try:
        logger.info(f"Received query: {chat_request.message}")
        
        # Process the query through our RAG system
        response_data = await query_engine.process_query(
            chat_request.message, 
            context=chat_request.context
        )
        
        return ChatResponse(
            response=response_data["response"],
            code_samples=response_data.get("code_samples", []),
            references=response_data.get("references", [])
        )
        
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time chat
    """
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            query_data = json.loads(data)
            
            # Process the query
            response_data = await query_engine.process_query(
                query_data["message"],
                context=query_data.get("context")
            )
            
            # Send response back
            await manager.send_personal_message(
                json.dumps(response_data), 
                websocket
            )
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        await manager.send_personal_message(
            json.dumps({"error": str(e)}), 
            websocket
        )

@app.get("/examples")
async def get_examples():
    """
    Get example MCP code snippets
    """
    examples = {
        "basic_server": {
            "title": "Basic MCP Server",
            "description": "A simple MCP server with two tools",
            "code": """
import asyncio
from typing import Any, Dict, List
from mcp.server import Server
from mcp.types import Tool

server = Server("example-mcp-server")

@server.tool()
async def calculate_sum(a: int, b: int) -> int:
    \"\"\"Calculate the sum of two numbers\"\"\"
    return a + b

@server.tool()
async def get_weather(city: str) -> Dict[str, Any]:
    \"\"\"Get weather information for a city\"\"\"
    # Mock weather data
    return {
        "city": city,
        "temperature": "22°C",
        "condition": "Sunny"
    }

if __name__ == "__main__":
    asyncio.run(server.run())
            """
        },
        "tool_registration": {
            "title": "Tool Registration",
            "description": "How to register tools with proper schemas",
            "code": """
from mcp.types import Tool, ToolParameter

# Define tool with schema
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

# Register the tool
server.add_tool(weather_tool, handler_function)
            """
        }
    }
    return examples

# Mount static files for the web interface
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True) 