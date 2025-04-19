#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting RAG Chatbot Setup...${NC}"

# Check Python version
echo -e "${YELLOW}Checking Python version...${NC}"
python_version=$(python3 --version 2>&1 | awk '{print $2}')
if [[ $python_version < "3.8" ]]; then
    echo -e "${RED}Error: Python 3.8 or higher is required${NC}"
    exit 1
fi

# Create and activate virtual environment
echo -e "${YELLOW}Setting up virtual environment...${NC}"
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install -r requirements.txt

# Check if Ollama is installed
echo -e "${YELLOW}Checking Ollama installation...${NC}"
if ! command -v ollama &> /dev/null; then
    echo -e "${YELLOW}Ollama not found. Installing...${NC}"
    
    # Install Ollama based on OS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        brew install ollama
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        curl -fsSL https://ollama.com/install.sh | sh
    else
        echo -e "${RED}Unsupported operating system${NC}"
        exit 1
    fi
fi

# Start Ollama service
echo -e "${YELLOW}Starting Ollama service...${NC}"
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to start
sleep 5

# Pull required model
echo -e "${YELLOW}Pulling Phi-3 model...${NC}"
ollama pull phi3

# Start Streamlit app
echo -e "${GREEN}Starting Streamlit app...${NC}"
streamlit run src/chatbot.py

# Cleanup on exit
trap "kill $OLLAMA_PID" EXIT 