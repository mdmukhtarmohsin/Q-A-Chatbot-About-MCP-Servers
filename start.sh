#!/bin/bash
# MCP Expert Chatbot - Startup Script

echo "🚀 Starting MCP Expert Chatbot setup..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}❌${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.8 or later."
    exit 1
fi

print_status "Python 3 found"

# Check Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
print_info "Python version: $PYTHON_VERSION"

# Verify minimum Python version (3.8+)
if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    print_status "Python version is compatible"
else
    print_error "Python 3.8 or later is required. Current version: $PYTHON_VERSION"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    print_info "Creating virtual environment..."
    python3 -m venv venv
    if [ $? -eq 0 ]; then
        print_status "Virtual environment created"
    else
        print_error "Failed to create virtual environment"
        exit 1
    fi
else
    print_status "Virtual environment already exists"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source venv/bin/activate

if [ $? -eq 0 ]; then
    print_status "Virtual environment activated"
else
    print_error "Failed to activate virtual environment"
    exit 1
fi

# Upgrade pip and install essential tools
print_info "Upgrading pip and installing essential tools..."
python -m pip install --upgrade pip setuptools wheel

# For Python 3.12+, install setuptools-scm if needed
if python3 -c "import sys; exit(0 if sys.version_info >= (3, 12) else 1)"; then
    print_info "Python 3.12+ detected, installing additional compatibility packages..."
    pip install setuptools-scm
fi

# Install dependencies with retry mechanism
print_info "Installing dependencies..."
MAX_RETRIES=3
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if pip install -r requirements.txt; then
        print_status "Dependencies installed successfully"
        break
    else
        RETRY_COUNT=$((RETRY_COUNT + 1))
        if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
            print_warning "Installation failed, retrying ($RETRY_COUNT/$MAX_RETRIES)..."
            sleep 2
        else
            print_error "Failed to install dependencies after $MAX_RETRIES attempts"
            print_info "Trying alternative installation method..."
            
            # Try installing packages individually for better error reporting
            echo "Installing packages individually..."
            pip install fastapi uvicorn[standard] pydantic python-dotenv
            pip install sentence-transformers numpy
            pip install chromadb
            pip install google-generativeai
            pip install aiofiles python-multipart websockets jinja2 markdown requests
            
            if [ $? -eq 0 ]; then
                print_status "Dependencies installed via alternative method"
                break
            else
                print_error "Failed to install dependencies. Please check the error messages above."
                exit 1
            fi
        fi
    fi
done

# Check if .env file exists
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        print_info "Creating .env file from .env.example..."
        cp .env.example .env
        print_warning "Please edit .env file and add your GEMINI_API_KEY"
        print_info "Get your API key from: https://aistudio.google.com/app/apikey"
    else
        print_warning ".env file not found. Creating basic .env file..."
        echo "GEMINI_API_KEY=your_api_key_here" > .env
        print_warning "Please edit .env file and add your GEMINI_API_KEY"
        print_info "Get your API key from: https://aistudio.google.com/app/apikey"
    fi
else
    print_status ".env file found"
fi

# Check if GEMINI_API_KEY is set
if grep -q "your_api_key_here" .env 2>/dev/null || ! grep -q "GEMINI_API_KEY=" .env 2>/dev/null; then
    print_warning "GEMINI_API_KEY not configured in .env file"
    print_info "The chatbot will not work without a valid Gemini API key"
    print_info "Get your API key from: https://aistudio.google.com/app/apikey"
    echo
    read -p "Do you want to continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Please configure your API key and run this script again"
        exit 1
    fi
fi

# Start the server
print_status "All dependencies installed successfully!"
print_info "Starting MCP Expert Chatbot server..."
echo
print_info "The chatbot will be available at: http://localhost:8000"
print_info "Press Ctrl+C to stop the server"
echo

# Run the server
python3 run.py 