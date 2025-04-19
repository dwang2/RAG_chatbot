from typing import List, Dict, Any, TypedDict, Annotated
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import Graph, StateGraph
from rag_chatbot.utils.vector_store import VectorStore
from rag_chatbot.models.ollama_client import get_ollama_client
from rag_chatbot.utils.logging_config import get_logger

# Initialize logger
logger = get_logger(__name__)

class AgentState(TypedDict):
    """State for the RAG agent."""
    question: str
    context: List[str]
    answer: str

def format_docs(docs: List[str]) -> str:
    """Format the retrieved documents into a single string."""
    return "\n\n".join(docs)

def retrieve(state: AgentState, vector_store: VectorStore) -> AgentState:
    """Retrieve relevant documents from the vector store."""
    try:
        logger.info(f"Retrieving documents for question: {state['question']}")
        results, _ = vector_store.search(state['question'])
        state['context'] = results
        logger.info(f"Retrieved {len(results)} documents")
        return state
    except Exception as e:
        logger.error(f"Error in retrieval: {str(e)}")
        raise

def generate(state: AgentState) -> AgentState:
    """Generate answer using the LLM."""
    try:
        llm = get_ollama_client()
        
        # Create the prompt template
        template = """Answer the question based on the following context:

Context:
{context}

Question: {question}

Answer the question based on the context above. If the context doesn't contain relevant information, say "I don't have enough information to answer that question." """
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Create the chain
        chain = prompt | llm | StrOutputParser()
        
        # Generate answer
        logger.info("Generating answer")
        state['answer'] = chain.invoke({
            "context": format_docs(state['context']),
            "question": state['question']
        })
        logger.info("Answer generated successfully")
        return state
    except Exception as e:
        logger.error(f"Error in generation: {str(e)}")
        raise

def create_rag_graph(vector_store: VectorStore) -> Graph:
    """Create a RAG graph using LangGraph."""
    try:
        # Create the workflow graph with AgentState
        workflow = StateGraph(AgentState)
        
        # Add nodes with proper state handling
        workflow.add_node("retrieve", lambda state: retrieve(state, vector_store))
        workflow.add_node("generate", generate)
        
        # Add edges
        workflow.add_edge("retrieve", "generate")
        
        # Set entry point
        workflow.set_entry_point("retrieve")
        
        # Compile the graph
        graph = workflow.compile()
        
        logger.info("RAG graph created successfully")
        return graph
    except Exception as e:
        logger.error(f"Error creating RAG graph: {str(e)}")
        raise

def get_answer(graph: Graph, question: str) -> str:
    """Get an answer from the RAG graph for a given question."""
    try:
        logger.info(f"Processing question: {question}")
        
        # Initialize AgentState
        initial_state: AgentState = {
            "question": question,
            "context": [],
            "answer": ""
        }
        
        # Run the graph with initial state
        final_state = graph.invoke(initial_state)
        
        # Ensure we have a valid state with an answer
        if not isinstance(final_state, dict) or 'answer' not in final_state:
            raise ValueError("Invalid state returned from graph")
            
        logger.info("Answer generated successfully")
        return final_state['answer']
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        raise

def process_pdf_file(pdf_file):
    # Step 1: Extract text
    texts = extract_text_from_pdf(pdf_file)
    
    # Step 2: Split into chunks
    text_chunks = split_text(texts)
    
    # Step 3: Create embeddings
    vector_store = VectorStore()
    embeddings = vector_store.create_embeddings(text_chunks)
    
    # Step 4: Save vector store
    vector_store.save_vector_store("data/vector_store")
    st.session_state.vector_store = vector_store 