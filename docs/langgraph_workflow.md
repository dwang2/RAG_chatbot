# LangGraph Workflow Diagram

```mermaid
graph TD
    subgraph Input
        A[User Question] --> B[AgentState]
    end

    subgraph State Management
        B --> C[Initialize State]
        C --> D[State: question, context, answer]
    end

    subgraph Processing Nodes
        D --> E[Retrieve Node]
        E --> F[Vector Store Search]
        F --> G[Update Context]
        G --> H[Generate Node]
        H --> I[LLM Processing]
        I --> J[Update Answer]
    end

    subgraph Output
        J --> K[Final State]
        K --> L[User Response]
    end

    %% Styling
    classDef input fill:#f9f,stroke:#333,stroke-width:2px;
    classDef state fill:#bbf,stroke:#333,stroke-width:2px;
    classDef process fill:#bfb,stroke:#333,stroke-width:2px;
    classDef output fill:#fbb,stroke:#333,stroke-width:2px;

    class A input;
    class B,C,D state;
    class E,F,G,H,I,J process;
    class K,L output;
```

## Workflow Description

1. **Input Processing**
   - User question is received
   - Initial AgentState is created

2. **State Management**
   - State is initialized with empty context and answer
   - Question is stored in state
   - State transitions are tracked

3. **Processing Nodes**
   - Retrieve Node: Searches vector store
   - Context Update: Adds relevant documents
   - Generate Node: Processes with LLM
   - Answer Update: Stores final response

4. **Output Generation**
   - Final state is processed
   - Response is formatted
   - Answer is returned to user

## State Transitions

```mermaid
stateDiagram-v2
    [*] --> Initialized
    Initialized --> Retrieving
    Retrieving --> Generating
    Generating --> Completed
    Completed --> [*]
```

## Error Handling

```mermaid
graph TD
    A[Process Start] --> B{Error?}
    B -->|No| C[Continue]
    B -->|Yes| D[Log Error]
    D --> E[Update State]
    E --> F[Handle Recovery]
    F --> C
``` 