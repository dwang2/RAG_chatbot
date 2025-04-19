# RAG Flow Diagram

```mermaid
graph TD
    subgraph Document Processing
        A[PDF Upload] --> B[Text Extraction]
        B --> C[Text Chunking]
        C --> D[Embedding Generation]
        D --> E[Vector Store]
    end

    subgraph Query Processing
        F[User Question] --> G[Query Embedding]
        G --> H[Vector Search]
        H --> I[Context Retrieval]
        I --> J[LLM Processing]
        J --> K[Response Generation]
    end

    subgraph Components
        L[PDF Processor]
        M[Text Processor]
        N[Vector Store]
        O[Ollama LLM]
    end

    %% Connections
    E --> N
    F --> O
    H --> N
    I --> O

    %% Styling
    classDef input fill:#f9f,stroke:#333,stroke-width:2px;
    classDef process fill:#bfb,stroke:#333,stroke-width:2px;
    classDef store fill:#bbf,stroke:#333,stroke-width:2px;
    classDef output fill:#fbb,stroke:#333,stroke-width:2px;

    class A,F input;
    class B,C,D,G,H,I,J,K process;
    class E,N store;
    class L,M,O output;
```

## Document Processing Flow

1. **PDF Upload and Processing**
   - PDF file is uploaded
   - Text is extracted from pages
   - Content is validated

2. **Text Processing**
   - Text is split into chunks
   - Chunks are normalized
   - Overlap is managed

3. **Embedding Generation**
   - Chunks are converted to embeddings
   - Embeddings are normalized
   - Batch processing is used

4. **Vector Store**
   - Embeddings are indexed
   - Metadata is stored
   - Persistence is managed

## Query Processing Flow

1. **Question Processing**
   - User question is received
   - Query embedding is generated
   - Search parameters are set

2. **Context Retrieval**
   - Vector search is performed
   - Relevant chunks are retrieved
   - Context is formatted

3. **Response Generation**
   - Context and question are combined
   - LLM generates response
   - Answer is formatted

## Component Interactions

```mermaid
graph LR
    A[PDF Processor] --> B[Text Processor]
    B --> C[Vector Store]
    C --> D[Ollama LLM]
    D --> E[Response]
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#bfb,stroke:#333,stroke-width:2px
    style C fill:#bbf,stroke:#333,stroke-width:2px
    style D fill:#fbb,stroke:#333,stroke-width:2px
    style E fill:#dfd,stroke:#333,stroke-width:2px
``` 