"""
Example MCP Server Implementation
Demonstrates various MCP patterns and best practices
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

# Mock MCP imports - replace with actual MCP library imports
class Server:
    def __init__(self, name: str):
        self.name = name
        self.tools = {}
        
    def tool(self, name: Optional[str] = None):
        def decorator(func):
            tool_name = name or func.__name__
            self.tools[tool_name] = func
            return func
        return decorator
    
    async def run(self):
        print(f"Starting MCP server: {self.name}")
        # Actual MCP server would handle protocol communication here
        await asyncio.Event().wait()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize server
server = Server("example-mcp-server")

# Server state for demonstration
server_state = {
    "user_preferences": {},
    "session_data": {},
    "file_cache": {}
}

@server.tool()
async def calculate_sum(a: int, b: int) -> Dict[str, Any]:
    """Calculate the sum of two numbers
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        Dictionary with the calculation result
    """
    logger.info(f"Calculating sum: {a} + {b}")
    result = a + b
    
    return {
        "operation": "sum",
        "operands": [a, b],
        "result": result,
        "timestamp": datetime.now().isoformat()
    }

@server.tool()
async def get_weather_info(city: str, units: str = "celsius") -> Dict[str, Any]:
    """Get mock weather information for a city
    
    Args:
        city: Name of the city
        units: Temperature units (celsius, fahrenheit, kelvin)
        
    Returns:
        Weather information dictionary
    """
    logger.info(f"Getting weather for {city} in {units}")
    
    # Mock weather data - in real implementation, this would call a weather API
    mock_temps = {
        "celsius": {"temp": 22, "unit": "°C"},
        "fahrenheit": {"temp": 72, "unit": "°F"},
        "kelvin": {"temp": 295, "unit": "K"}
    }
    
    temp_data = mock_temps.get(units, mock_temps["celsius"])
    
    return {
        "city": city,
        "temperature": temp_data["temp"],
        "unit": temp_data["unit"],
        "condition": "Sunny",
        "humidity": 65,
        "wind_speed": "10 km/h",
        "last_updated": datetime.now().isoformat()
    }

@server.tool()
async def set_user_preference(key: str, value: str) -> str:
    """Set a user preference for the session
    
    Args:
        key: Preference key
        value: Preference value
        
    Returns:
        Confirmation message
    """
    logger.info(f"Setting user preference: {key} = {value}")
    server_state["user_preferences"][key] = value
    return f"Set preference '{key}' to '{value}'"

@server.tool()
async def get_user_preference(key: str) -> str:
    """Get a user preference from the session
    
    Args:
        key: Preference key to retrieve
        
    Returns:
        Preference value or 'Not set' if not found
    """
    value = server_state["user_preferences"].get(key, "Not set")
    logger.info(f"Retrieved user preference: {key} = {value}")
    return value

@server.tool()
async def list_files(directory: str = ".", pattern: str = "*") -> List[Dict[str, Any]]:
    """List files in a directory with optional pattern matching
    
    Args:
        directory: Directory path to list (default: current directory)
        pattern: File pattern to match (default: all files)
        
    Returns:
        List of file information dictionaries
    """
    logger.info(f"Listing files in {directory} with pattern {pattern}")
    
    try:
        path = Path(directory)
        if not path.exists():
            return {"error": f"Directory '{directory}' does not exist"}
        
        files = []
        for file_path in path.glob(pattern):
            if file_path.is_file():
                stat = file_path.stat()
                files.append({
                    "name": file_path.name,
                    "path": str(file_path),
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "type": "file"
                })
            elif file_path.is_dir():
                files.append({
                    "name": file_path.name,
                    "path": str(file_path),
                    "type": "directory"
                })
        
        return files
        
    except Exception as e:
        logger.error(f"Error listing files: {e}")
        return {"error": str(e)}

@server.tool()
async def read_file_content(file_path: str, max_lines: int = 100) -> Dict[str, Any]:
    """Read content from a text file
    
    Args:
        file_path: Path to the file to read
        max_lines: Maximum number of lines to read
        
    Returns:
        File content and metadata
    """
    logger.info(f"Reading file: {file_path} (max {max_lines} lines)")
    
    try:
        path = Path(file_path)
        if not path.exists():
            return {"error": f"File '{file_path}' does not exist"}
        
        if not path.is_file():
            return {"error": f"'{file_path}' is not a file"}
        
        # Check cache first
        cache_key = f"{file_path}:{path.stat().st_mtime}"
        if cache_key in server_state["file_cache"]:
            logger.info("Returning cached file content")
            return server_state["file_cache"][cache_key]
        
        with open(path, 'r', encoding='utf-8') as f:
            lines = []
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                lines.append(line.rstrip())
        
        result = {
            "file_path": file_path,
            "lines_read": len(lines),
            "content": lines,
            "truncated": len(lines) == max_lines,
            "file_size": path.stat().st_size,
            "last_modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat()
        }
        
        # Cache the result
        server_state["file_cache"][cache_key] = result
        
        return result
        
    except UnicodeDecodeError:
        return {"error": f"File '{file_path}' is not a text file or has unsupported encoding"}
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        return {"error": str(e)}

@server.tool()
async def search_in_files(directory: str, search_term: str, file_pattern: str = "*.py") -> List[Dict[str, Any]]:
    """Search for a term in files within a directory
    
    Args:
        directory: Directory to search in
        search_term: Term to search for
        file_pattern: Pattern for files to search (default: *.py)
        
    Returns:
        List of search results with file paths and line numbers
    """
    logger.info(f"Searching for '{search_term}' in {directory}/{file_pattern}")
    
    try:
        path = Path(directory)
        if not path.exists():
            return {"error": f"Directory '{directory}' does not exist"}
        
        results = []
        for file_path in path.glob(file_pattern):
            if file_path.is_file():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line_num, line in enumerate(f, 1):
                            if search_term.lower() in line.lower():
                                results.append({
                                    "file": str(file_path),
                                    "line_number": line_num,
                                    "line_content": line.strip(),
                                    "match_position": line.lower().find(search_term.lower())
                                })
                except (UnicodeDecodeError, PermissionError):
                    # Skip files that can't be read
                    continue
        
        return {
            "search_term": search_term,
            "directory": directory,
            "pattern": file_pattern,
            "results": results,
            "total_matches": len(results)
        }
        
    except Exception as e:
        logger.error(f"Error searching files: {e}")
        return {"error": str(e)}

@server.tool()
async def get_system_info() -> Dict[str, Any]:
    """Get system information
    
    Returns:
        Dictionary with system details
    """
    import platform
    import psutil
    
    logger.info("Getting system information")
    
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "python_version": platform.python_version()
            },
            "resources": {
                "cpu_percent": cpu_percent,
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "percent": memory.percent
                },
                "disk": {
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free,
                    "percent": (disk.used / disk.total) * 100
                }
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except ImportError:
        # Fallback if psutil is not available
        return {
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "python_version": platform.python_version()
            },
            "note": "Install psutil for detailed resource information",
            "timestamp": datetime.now().isoformat()
        }

@server.tool()
async def clear_cache() -> str:
    """Clear the server's internal cache
    
    Returns:
        Confirmation message
    """
    logger.info("Clearing server cache")
    cache_size = len(server_state["file_cache"])
    server_state["file_cache"].clear()
    return f"Cleared cache containing {cache_size} entries"

@server.tool()
async def get_server_stats() -> Dict[str, Any]:
    """Get server statistics and state information
    
    Returns:
        Dictionary with server statistics
    """
    logger.info("Getting server statistics")
    
    return {
        "server_name": server.name,
        "tools_registered": len(server.tools),
        "tool_names": list(server.tools.keys()),
        "state": {
            "user_preferences": len(server_state["user_preferences"]),
            "cached_files": len(server_state["file_cache"]),
            "session_data": len(server_state["session_data"])
        },
        "uptime": "Server runtime tracking not implemented",
        "timestamp": datetime.now().isoformat()
    }

async def main():
    """Main entry point for the MCP server"""
    logger.info(f"Starting {server.name}")
    logger.info(f"Registered tools: {list(server.tools.keys())}")
    
    try:
        await server.run()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 