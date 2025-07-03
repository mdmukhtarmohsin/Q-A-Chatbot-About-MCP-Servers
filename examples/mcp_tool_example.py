"""
MCP Tool Registration Examples
Demonstrates different patterns for registering tools with proper schemas
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Literal, Union
from enum import Enum

# Mock MCP types - replace with actual MCP library imports
class Tool:
    def __init__(self, name: str, description: str, parameters: Dict[str, Any]):
        self.name = name
        self.description = description
        self.parameters = parameters

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

logger = logging.getLogger(__name__)
server = Server("tool-examples")

# Example 1: Simple tool with basic types
@server.tool()
async def simple_calculator(operation: str, a: float, b: float) -> Dict[str, Any]:
    """Perform basic arithmetic operations
    
    Args:
        operation: The operation to perform (add, subtract, multiply, divide)
        a: First number
        b: Second number
        
    Returns:
        Result of the calculation
    """
    operations = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y if y != 0 else None
    }
    
    if operation not in operations:
        return {"error": f"Unknown operation: {operation}"}
    
    if operation == "divide" and b == 0:
        return {"error": "Division by zero is not allowed"}
    
    result = operations[operation](a, b)
    
    return {
        "operation": operation,
        "operands": [a, b],
        "result": result,
        "timestamp": datetime.now().isoformat()
    }

# Example 2: Tool with enums and default values
@server.tool()
async def format_text(
    text: str,
    format_type: Literal["uppercase", "lowercase", "title", "sentence"] = "sentence",
    remove_spaces: bool = False,
    max_length: Optional[int] = None
) -> Dict[str, Any]:
    """Format text according to specified parameters
    
    Args:
        text: The text to format
        format_type: How to format the text case
        remove_spaces: Whether to remove all spaces
        max_length: Maximum length to truncate to (optional)
        
    Returns:
        Formatted text and metadata
    """
    
    # Apply case formatting
    if format_type == "uppercase":
        formatted = text.upper()
    elif format_type == "lowercase":
        formatted = text.lower()
    elif format_type == "title":
        formatted = text.title()
    elif format_type == "sentence":
        formatted = text.capitalize()
    else:
        formatted = text
    
    # Remove spaces if requested
    if remove_spaces:
        formatted = formatted.replace(" ", "")
    
    # Apply length limit
    original_length = len(formatted)
    if max_length and len(formatted) > max_length:
        formatted = formatted[:max_length]
        truncated = True
    else:
        truncated = False
    
    return {
        "original_text": text,
        "formatted_text": formatted,
        "format_applied": format_type,
        "spaces_removed": remove_spaces,
        "original_length": len(text),
        "final_length": len(formatted),
        "truncated": truncated,
        "max_length": max_length
    }

# Example 3: Tool with complex nested objects
@server.tool()
async def create_user_profile(
    name: str,
    age: int,
    email: str,
    preferences: Dict[str, Union[str, int, bool]],
    tags: List[str] = None
) -> Dict[str, Any]:
    """Create a user profile with validation
    
    Args:
        name: Full name of the user
        age: Age in years (must be 13 or older)
        email: Email address
        preferences: Dictionary of user preferences
        tags: Optional list of user tags
        
    Returns:
        Created user profile or validation errors
    """
    
    errors = []
    
    # Validate age
    if age < 13:
        errors.append("Age must be 13 or older")
    if age > 150:
        errors.append("Age must be 150 or younger")
    
    # Validate email format (basic check)
    if "@" not in email or "." not in email:
        errors.append("Invalid email format")
    
    # Validate name
    if len(name.strip()) < 2:
        errors.append("Name must be at least 2 characters")
    
    if errors:
        return {"errors": errors, "profile_created": False}
    
    profile = {
        "id": hash(f"{name}{email}") % 100000,  # Simple ID generation
        "name": name.strip(),
        "age": age,
        "email": email.lower(),
        "preferences": preferences or {},
        "tags": tags or [],
        "created_at": datetime.now().isoformat(),
        "active": True
    }
    
    return {
        "profile": profile,
        "profile_created": True,
        "message": f"Profile created for {name}"
    }

# Example 4: Tool with file operations
@server.tool()
async def file_operations(
    operation: Literal["read", "write", "append", "delete", "info"],
    file_path: str,
    content: Optional[str] = None,
    encoding: str = "utf-8"
) -> Dict[str, Any]:
    """Perform file operations with proper error handling
    
    Args:
        operation: The file operation to perform
        file_path: Path to the file
        content: Content for write/append operations
        encoding: File encoding (default: utf-8)
        
    Returns:
        Operation result and file information
    """
    
    from pathlib import Path
    import os
    
    try:
        path = Path(file_path)
        
        if operation == "read":
            if not path.exists():
                return {"error": f"File {file_path} does not exist"}
            
            with open(path, 'r', encoding=encoding) as f:
                file_content = f.read()
            
            return {
                "operation": "read",
                "file_path": file_path,
                "content": file_content,
                "size": len(file_content),
                "encoding": encoding
            }
        
        elif operation == "write":
            if content is None:
                return {"error": "Content is required for write operation"}
            
            # Create directory if it doesn't exist
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w', encoding=encoding) as f:
                f.write(content)
            
            return {
                "operation": "write",
                "file_path": file_path,
                "bytes_written": len(content.encode(encoding)),
                "encoding": encoding
            }
        
        elif operation == "append":
            if content is None:
                return {"error": "Content is required for append operation"}
            
            with open(path, 'a', encoding=encoding) as f:
                f.write(content)
            
            return {
                "operation": "append",
                "file_path": file_path,
                "bytes_appended": len(content.encode(encoding)),
                "encoding": encoding
            }
        
        elif operation == "delete":
            if not path.exists():
                return {"error": f"File {file_path} does not exist"}
            
            path.unlink()
            
            return {
                "operation": "delete",
                "file_path": file_path,
                "deleted": True
            }
        
        elif operation == "info":
            if not path.exists():
                return {"error": f"File {file_path} does not exist"}
            
            stat = path.stat()
            
            return {
                "operation": "info",
                "file_path": file_path,
                "size": stat.st_size,
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "is_directory": path.is_dir(),
                "is_file": path.is_file(),
                "permissions": oct(stat.st_mode)[-3:]
            }
        
        else:
            return {"error": f"Unknown operation: {operation}"}
    
    except PermissionError:
        return {"error": f"Permission denied: {file_path}"}
    except UnicodeDecodeError:
        return {"error": f"Cannot decode file with {encoding} encoding"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

# Example 5: Tool with data processing and analysis
@server.tool()
async def analyze_data(
    data: List[Union[int, float]],
    operations: List[Literal["mean", "median", "mode", "std", "min", "max", "sum"]] = None
) -> Dict[str, Any]:
    """Analyze numerical data with various statistical operations
    
    Args:
        data: List of numerical values to analyze
        operations: List of statistical operations to perform
        
    Returns:
        Statistical analysis results
    """
    
    if not data:
        return {"error": "Data list cannot be empty"}
    
    if not all(isinstance(x, (int, float)) for x in data):
        return {"error": "All data values must be numbers"}
    
    if operations is None:
        operations = ["mean", "median", "min", "max", "sum"]
    
    results = {}
    
    try:
        if "mean" in operations:
            results["mean"] = sum(data) / len(data)
        
        if "median" in operations:
            sorted_data = sorted(data)
            n = len(sorted_data)
            if n % 2 == 0:
                results["median"] = (sorted_data[n//2 - 1] + sorted_data[n//2]) / 2
            else:
                results["median"] = sorted_data[n//2]
        
        if "mode" in operations:
            from collections import Counter
            counts = Counter(data)
            max_count = max(counts.values())
            modes = [k for k, v in counts.items() if v == max_count]
            results["mode"] = modes[0] if len(modes) == 1 else modes
        
        if "std" in operations:
            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)
            results["standard_deviation"] = variance ** 0.5
        
        if "min" in operations:
            results["min"] = min(data)
        
        if "max" in operations:
            results["max"] = max(data)
        
        if "sum" in operations:
            results["sum"] = sum(data)
        
        return {
            "data_length": len(data),
            "operations_performed": operations,
            "results": results,
            "data_range": [min(data), max(data)] if data else None
        }
    
    except Exception as e:
        return {"error": f"Analysis error: {str(e)}"}

# Example 6: Tool with external API simulation
@server.tool()
async def mock_api_request(
    method: Literal["GET", "POST", "PUT", "DELETE"],
    endpoint: str,
    headers: Optional[Dict[str, str]] = None,
    payload: Optional[Dict[str, Any]] = None,
    timeout: int = 30
) -> Dict[str, Any]:
    """Simulate an API request (mock implementation for demonstration)
    
    Args:
        method: HTTP method
        endpoint: API endpoint URL
        headers: Optional request headers
        payload: Optional request payload for POST/PUT
        timeout: Request timeout in seconds
        
    Returns:
        Mock API response
    """
    
    # Simulate processing time
    await asyncio.sleep(0.1)
    
    # Mock response based on endpoint
    if "users" in endpoint:
        mock_data = {
            "users": [
                {"id": 1, "name": "Alice", "email": "alice@example.com"},
                {"id": 2, "name": "Bob", "email": "bob@example.com"}
            ]
        }
    elif "products" in endpoint:
        mock_data = {
            "products": [
                {"id": 1, "name": "Widget", "price": 19.99},
                {"id": 2, "name": "Gadget", "price": 29.99}
            ]
        }
    else:
        mock_data = {"message": "Mock API response", "endpoint": endpoint}
    
    # Simulate different status codes based on method
    if method == "POST" and payload:
        status_code = 201
        mock_data = {"created": True, "data": payload}
    elif method == "DELETE":
        status_code = 204
        mock_data = {"deleted": True}
    elif method == "PUT" and payload:
        status_code = 200
        mock_data = {"updated": True, "data": payload}
    else:
        status_code = 200
    
    return {
        "request": {
            "method": method,
            "endpoint": endpoint,
            "headers": headers or {},
            "payload": payload,
            "timeout": timeout
        },
        "response": {
            "status_code": status_code,
            "data": mock_data,
            "timestamp": datetime.now().isoformat()
        },
        "mock": True
    }

def main():
    """Display information about registered tools"""
    print(f"MCP Server: {server.name}")
    print(f"Registered tools: {len(server.tools)}")
    print("\nAvailable tools:")
    
    for tool_name, tool_func in server.tools.items():
        doc = tool_func.__doc__ or "No description"
        first_line = doc.split('\n')[0].strip()
        print(f"  - {tool_name}: {first_line}")
    
    print("\nThis is a demonstration server showing MCP tool patterns.")
    print("In a real implementation, these tools would be registered with an actual MCP server.")

if __name__ == "__main__":
    main() 