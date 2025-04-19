# RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that uses Ollama's phi4 model for intelligent question answering over documents.

## Features

- PDF document processing and text extraction
- Intelligent text chunking and embedding generation
- Vector-based semantic search
- RAG-based question answering
- LangGraph workflow for state management
- Configurable model parameters
- Progress tracking and error handling
- Modern Streamlit UI

## System Architecture

### Data Flow
The system follows a RAG architecture with the following components:
1. Document Processing
   - PDF text extraction
   - Text chunking
   - Embedding generation
   - Vector store creation

2. Query Processing
   - User question input
   - Context retrieval
   - Response generation
   - Answer presentation

For a detailed view of the data flow, see the [RAG Flow Diagram](docs/rag_flow_diagram.md).

### LangGraph Workflow
The system uses LangGraph for state management and workflow control:
1. State Management
   - AgentState for tracking question, context, and answer
   - TypedDict for type-safe state handling
   - Clear state transitions

2. Processing Nodes
   - Retrieve node for context gathering
   - Generate node for answer creation
   - Error handling and logging

For a detailed view of the workflow, see the [LangGraph Workflow Diagram](docs/langgraph_workflow.md).

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rag-chatbot.git
cd rag-chatbot
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install Ollama and pull the phi4 model:
```bash
# Install Ollama (follow instructions for your OS)
# Pull the phi4 model
ollama pull phi4
```

5. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your settings
```

## Configuration

The system uses the following environment variables:

### Ollama Configuration
- `OLLAMA_BASE_URL`: Base URL for Ollama server (default: http://127.0.0.1:11434)
- `OLLAMA_MODEL`: Model name (default: phi4)
- `OLLAMA_TEMPERATURE`: Response creativity (default: 0.7)
- `OLLAMA_TOP_P`: Response diversity (default: 0.9)
- `OLLAMA_NUM_CTX`: Context length (default: 2048)
- `OLLAMA_NUM_THREAD`: Thread count (default: 4)
- `OLLAMA_STOP`: Stop sequences for response control
- `OLLAMA_REPEAT_PENALTY`: Penalty for repetitive responses
- `OLLAMA_TOP_K`: Top-k sampling parameter

## Usage

1. Start the Ollama server:
```bash
ollama serve
```

2. Run the Streamlit application:
```bash
streamlit run rag_chatbot/chatbot.py
```

3. Upload a PDF document and start asking questions!

## Project Structure

```
rag-chatbot/
├── docs/
│   ├── rag_flow_diagram.md
│   └── langgraph_workflow.md
├── rag_chatbot/
│   ├── chatbot.py
│   ├── models/
│   │   └── ollama_client.py
│   ├── utils/
│   │   ├── text_processor.py
│   │   ├── vector_store.py
│   │   ├── pdf_processor.py
│   │   └── logging_config.py
│   └── rag_chain.py
├── .env
├── .env.example
├── .cursorrules
├── requirements.txt
└── README.md
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.