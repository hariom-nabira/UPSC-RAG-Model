import uvicorn
import json # For serializing data for SSE
import uuid # For generating session IDs
# import traceback # No longer needed directly, logger.exception handles it
from enum import Enum
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks # Added Request and BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse # For SSE
from pydantic import BaseModel
from typing import List, Dict, Any, AsyncGenerator, Optional, Union # Added AsyncGenerator, Optional, Union
from langchain.memory import ConversationBufferMemory # Ensure this is imported if not already
from langchain_core.language_models import BaseLanguageModel
from langchain.chains import LLMChain # LLMChain will be replaced by LCEL
from langchain.prompts import PromptTemplate
# TODO: Potentially add imports for ParentDocumentRetriever, CohereRerank, or other specific components if used directly here.
# from langchain.retrievers import ContextualCompressionRetriever, ParentDocumentRetriever
# from langchain.retrievers.document_compressors import CohereRerank, LLMChainExtractor


# Import RAG components (adjust paths/imports as necessary based on your project structure)
# These will be the functions and classes you've already defined for your RAG system
from config import OPENAI_API_KEY, LLM_CHAT_MODEL, EMBEDDING_MODEL, EMBEDDING_DIMENSIONS, LLM_UTILITY_MODEL
from embeddings import get_embedding_model, get_utility_llm
# retrieval.py has been updated for parent/child and re-ranking
from retrieval import (
    get_vectorstore_and_parent_store, # Updated function name
    get_retriever, 
    get_memory, 
    get_conversational_qa_chain, 
    get_current_chat_llm,
    global_parent_doc_store # Import the global parent doc store if needed directly, or rely on it being set
)
# from security import check_openai_api_key # Removed import
from utils import get_logger # Import the logger

# Initialize logger
logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="UPSC RAG API",
    description="API for interacting with the UPSC RAG model, incorporating advanced retrieval strategies.",
    version="0.2.0" # Version update
)

# CORS (Cross-Origin Resource Sharing) middleware
# Allows your React frontend (running on a different port) to communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify your frontend URL, e.g., "http://localhost:3000"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"], # Allow all headers, or specify like ["Content-Type", "X-CSRF-Token"]
)

# --- Globals for RAG components & Sessions ---
embedding_model_instance = None
utility_llm_instance = None
vectorstore_instance = None # This will be our ChromaDB with child docs
parent_doc_store_instance = None # This will be the InMemoryStore for parent docs
retriever_instance = None 
active_sessions: Dict[str, ConversationBufferMemory] = {} 
intent_detection_chain: LLMChain = None 
small_talk_chain: LLMChain = None 

# --- Pydantic Models for Request/Response ---
class ChatQuery(BaseModel):
    message: str
    session_id: Optional[str] = None # Client can send an existing session_id
    chat_history: List[Dict[str, str]] = [] # Frontend history, used if new session or to reconcile
    # TODO: Add task_type if client needs to specify it for dynamic retrieval,
    # otherwise infer server-side based on intent.
    # task_type: Optional[str] = None

class ChatResponse(BaseModel): # This is not used for SSE, but good for potential non-streaming endpoints
    response: str
    session_id: str
    status: str
    error: Optional[str] = None
    # TODO: Potentially add sources here if not streaming or for a summary

# --- Intent Definitions ---
class UserIntent(str, Enum):
    RAG_QUERY = "rag_query"
    GREETING = "greeting"
    CLEAR_SESSION = "clear_session"
    SMALL_TALK = "small_talk"
    COMPLEX_RAG_QUERY = "complex_rag_query" # Example for dynamic retrieval
    SUMMARY_REQUEST = "summary_request" # Example for dynamic retrieval
    UNKNOWN = "unknown"

# Updated prompt to include new intents if necessary.
INTENT_DETECTION_PROMPT_TEMPLATE = """\
You are an expert intent classification assistant.
Given the user's message, classify it *strictly* into one of the following intents:
{intent_list_str}

Return only the intent name, e.g., 'rag_query' or 'greeting'. Do not add any other text or explanation.

Examples:
User: Hello there
Intent: greeting

User: What were the main causes of the 1857 revolt?
Intent: rag_query

User: Can you start a new chat?
Intent: clear_session

User: How are you doing today?
Intent: small_talk

User: Can you give me a detailed explanation of the Non-Cooperation Movement and its impact on different social groups?
Intent: complex_rag_query

User: Summarize the key aspects of India's foreign policy.
Intent: summary_request

User: {user_message}
Intent:"""

# --- API Endpoints ---
@app.on_event("startup")
async def startup_event():
    global embedding_model_instance, utility_llm_instance, vectorstore_instance, parent_doc_store_instance, retriever_instance
    global intent_detection_chain, small_talk_chain
    # Make sure global_parent_doc_store from retrieval.py is also accessible if get_retriever relies on it implicitly
    # However, it's cleaner to pass it explicitly, which we do below.
    
    logger.info("Checking API key...")
    if not OPENAI_API_KEY:
        logger.critical("OpenAI API Key is missing. Set OPENAI_API_KEY environment variable.")
        raise RuntimeError("OpenAI API Key is missing. Set OPENAI_API_KEY environment variable.")
    logger.info("API key check passed.")

    logger.info("Initializing RAG components and utility LLMs on startup...")
    embedding_model_instance = get_embedding_model()
    utility_llm_instance = get_utility_llm()
    
    logger.info("Initializing vectorstore and parent document store...")
    # get_vectorstore_and_parent_store now returns both
    # Note: _utility_llm argument was used by old get_vectorstore if RAPTOR was active for summaries.
    # The new process_documents in data_processing.py (called by get_vectorstore_and_parent_store)
    # currently bypasses RAPTOR, so _utility_llm is not strictly needed by it for that path.
    # However, keeping it in case RAPTOR is re-enabled in data_processing.py and needs the llm.
    vectorstore_instance, parent_doc_store_instance = get_vectorstore_and_parent_store(
        _utility_llm=utility_llm_instance, # Pass utility LLM in case data_processing re-enables RAPTOR
        _embedding_model=embedding_model_instance
    )
    
    if vectorstore_instance is None or parent_doc_store_instance is None:
        logger.critical("Failed to initialize vectorstore or parent document store.")
        raise RuntimeError("Failed to initialize vectorstore or parent document store.")
    logger.info("Vectorstore (ChromaDB for child docs) and Parent Document Store (InMemory) initialized.")
    
    # Get the advanced retriever (ParentDocumentRetriever + Compression)
    # It now requires the parent_doc_store_instance
    retriever_instance = get_retriever(vectorstore_instance, utility_llm_instance, parent_doc_store_instance)
    logger.info("Advanced RAG retriever initialized (ParentDocumentRetriever with Contextual Compression).")

    # Initialize Intent Detection Chain
    # Consider using LCEL for chains: PromptTemplate(...) | llm | StrOutputParser()
    intent_list = [intent.value for intent in UserIntent]
    intent_prompt = PromptTemplate(
        input_variables=["user_message", "intent_list_str"],
        template=INTENT_DETECTION_PROMPT_TEMPLATE
    )
    # Ensure all intent values are passed to the template
    # The template expects {intent_list_str}
    intent_detection_chain = LLMChain(
        llm=utility_llm_instance,
        prompt=intent_prompt,
        verbose=False
    ).with_config({"run_name": "IntentDetection"})
    logger.info("Intent detection chain initialized.")

    # Initialize Small Talk Chain
    small_talk_prompt_template = "You are a friendly and helpful AI assistant designed for UPSC exam aspirants. Keep your responses concise and encouraging. User: {user_message}\nAssistant:"
    small_talk_prompt = PromptTemplate(input_variables=["user_message"], template=small_talk_prompt_template)
    small_talk_chain = LLMChain(llm=utility_llm_instance, prompt=small_talk_prompt, verbose=False).with_config({"run_name": "SmallTalk"})
    logger.info("Small talk chain initialized.")

async def stream_rag_response(query: ChatQuery) -> AsyncGenerator[str, None]:
    global active_sessions, intent_detection_chain, small_talk_chain, retriever_instance
    session_id = query.session_id
    new_session_created = False

    # --- Session and Memory Management ---
    if session_id and session_id in active_sessions:
        session_memory = active_sessions[session_id]
        logger.info(f"Using existing session: {session_id}, memory has {len(session_memory.chat_memory.messages)} messages.")
    else:
        session_id = str(uuid.uuid4())
        session_memory = get_memory()
        active_sessions[session_id] = session_memory
        new_session_created = True
        logger.info(f"Created new session: {session_id}")
        if query.chat_history:
            logger.info(f"Populating memory for new session {session_id} from client history ({len(query.chat_history)} entries).")
            temp_user_msg_for_new_session_mem_load = None
            for entry in query.chat_history:
                if entry["role"] == "user":
                    temp_user_msg_for_new_session_mem_load = entry["content"]
                elif entry["role"] == "ai" and temp_user_msg_for_new_session_mem_load is not None:
                    session_memory.chat_memory.add_user_message(temp_user_msg_for_new_session_mem_load)
                    session_memory.chat_memory.add_ai_message(entry["content"])
                    temp_user_msg_for_new_session_mem_load = None
    
    if new_session_created or not query.session_id:
        session_id_event = {"type": "session_id", "id": session_id}
        yield f"data: {json.dumps(session_id_event)}\n\n"

    try:
        # --- 1. Intent Detection ---
        intent_list_str = ", ".join([f"'{intent.value}'" for intent in UserIntent])
        intent_result = await intent_detection_chain.arun(user_message=query.message, intent_list_str=intent_list_str)
        # Normalize and validate intent
        try:
            detected_intent = UserIntent(intent_result.strip().lower().replace("'", ""))
        except ValueError:
            logger.warning(f"Unknown intent '{intent_result}' detected for message: {query.message}. Defaulting to RAG_QUERY.")
            detected_intent = UserIntent.RAG_QUERY # Fallback or UNKNOWN
        
        intent_event = {"type": "intent", "content": detected_intent.value}
        yield f"data: {json.dumps(intent_event)}\n\n"
        logger.info(f"Detected intent for session {session_id}: {detected_intent.value}")

        # --- 2. Task-Specific Processing based on Intent ---
        if detected_intent == UserIntent.GREETING:
            response_content = "Hello! How can I help you with your UPSC preparation today?"
            # Or use a simple LLM call for varied greetings.
            token_event = {"type": "token", "content": response_content}
            yield f"data: {json.dumps(token_event)}\n\n"
        elif detected_intent == UserIntent.CLEAR_SESSION:
            if session_id in active_sessions:
                del active_sessions[session_id]
                logger.info(f"Cleared session: {session_id}")
            session_memory = get_memory() # Reset memory for current interaction
            active_sessions[session_id] = session_memory # Store new memory
            response_content = "Session cleared. Let's start a fresh conversation!"
            token_event = {"type": "token", "content": response_content}
            yield f"data: {json.dumps(token_event)}\n\n"
        elif detected_intent == UserIntent.SMALL_TALK:
            # Use the small_talk_chain
            response_content = await small_talk_chain.arun(user_message=query.message)
            token_event = {"type": "token", "content": response_content}
            yield f"data: {json.dumps(token_event)}\n\n"

        elif detected_intent in [UserIntent.RAG_QUERY, UserIntent.COMPLEX_RAG_QUERY, UserIntent.SUMMARY_REQUEST]:
            if not retriever_instance or not vectorstore_instance:
                error_msg = "RAG system not fully initialized."
                logger.error(f"Attempt to use RAG while not initialized. Session: {session_id}")
                error_event = {"type": "error", "content": error_msg}
                yield f"data: {json.dumps(error_event)}\n\n"
                return

            current_chat_llm = get_current_chat_llm()
            
            # --- FEATURE: Dynamic Retrieval Parameters ---
            # Based on detected_intent, you might adjust k, metadata filters, etc.
            # e.g., for COMPLEX_RAG_QUERY, retrieve more diverse documents or enable parent chunks.
            # For SUMMARY_REQUEST, retrieve broader context.
            # This logic would modify how `retriever_instance` is used or configured for this call.
            # Example:
            # retrieval_kwargs = {}
            # if detected_intent == UserIntent.COMPLEX_RAG_QUERY:
            #   retrieval_kwargs['search_kwargs'] = {'k': 10} # retrieve more docs
            #   # Potentially tell retriever to fetch parent documents if using ParentDocumentRetriever
            # elif detected_intent == UserIntent.SUMMARY_REQUEST:
            #   retrieval_kwargs['search_kwargs'] = {'k': 5, 'fetch_k': 20} # fetch more for context, then filter/compress
            # This assumes your get_retriever in retrieval.py returns a retriever that can accept such dynamic kwargs
            # or that you have different pre-configured retrievers.

            # --- FEATURE: Structured Retrieval & Decoupled Chunks (ParentDocumentRetriever) ---
            # If using ParentDocumentRetriever, it handles fetching parent chunks for context.
            # The `retriever_instance` should be configured for this in `retrieval.py`.
            # Structured retrieval (metadata filtering) would also be part of the retriever's configuration
            # or dynamically passed search_kwargs.
            
            # --- FEATURE: Optimize Context Embeddings (Re-ranking) ---
            # If a re-ranker is used, it would typically be part of the `retriever_instance` pipeline.
            # For example, `ContextualCompressionRetriever` with `CohereRerank` or similar.
            # This would happen within the retriever's `get_relevant_documents` (or async equivalent) method.
            # The `qa_chain` would then receive the re-ranked, possibly larger/parent documents.

            qa_chain = get_conversational_qa_chain(
                llm=current_chat_llm,
                retriever=retriever_instance, # This retriever is now expected to be more advanced
                memory=session_memory
                # TODO: Ensure return_source_documents=True and return_generated_question=True if needed by chain type
            )

            full_answer = ""
            source_documents = []
            
            async for chunk in qa_chain.astream({"question": query.message, "chat_history": []}):
                if isinstance(chunk, dict):
                    if "answer" in chunk and chunk["answer"] is not None:
                        token = chunk["answer"]
                        full_answer += token
                        token_event = {"type": "token", "content": token}
                        yield f"data: {json.dumps(token_event)}\n\n"
                    # Source documents might come at the end with ConversationalRetrievalChain
                    # or specific chain types might stream them differently.
                    if "source_documents" in chunk and chunk["source_documents"]:
                        source_documents = chunk["source_documents"] 
                elif isinstance(chunk, str): # some chains might stream raw strings
                    full_answer += chunk
                    token_event = {"type": "token", "content": chunk}
                    yield f"data: {json.dumps(token_event)}\n\n"

            # Fallback for source documents if not streamed:
            if not source_documents and hasattr(qa_chain, 'ainvoke'): # Check if chain has a final call method
                logger.info(f"Source documents not found during streaming for session {session_id}. Attempting a final call.")
                try:
                    # Pass history from memory as it's updated by astream
                    final_history_for_sources = session_memory.chat_memory.messages
                    non_streamed_result = await qa_chain.ainvoke({"question": query.message, "chat_history": final_history_for_sources})
                    source_documents = non_streamed_result.get("source_documents", [])
                    if source_documents:
                        logger.info(f"Successfully retrieved {len(source_documents)} source documents via ainvoke fallback for session {session_id}.")
                    else:
                        logger.warning(f"No source documents found even after ainvoke fallback for session {session_id}.")
                except Exception as e_invoke:
                    logger.error(f"Error during ainvoke fallback for sources in session {session_id}: {e_invoke}")

            if source_documents:
                formatted_sources = []
                for doc in source_documents:
                    # Decoupled chunks: If `doc` is a parent document, its content is larger.
                    # Metadata from structured retrieval will be in `doc.metadata`.
                    formatted_sources.append({
                        "page_content": doc.page_content, # This would be the content of the (potentially larger) synthesis chunk
                        "metadata": doc.metadata # Contains info from structured retrieval
                    })
                sources_event = {"type": "sources", "content": formatted_sources}
                yield f"data: {json.dumps(sources_event)}\n\n"
        else: # UserIntent.UNKNOWN or other unhandled cases
            response_content = "I'm not sure how to handle that. Could you try rephrasing?"
            token_event = {"type": "token", "content": response_content}
            yield f"data: {json.dumps(token_event)}\n\n"

        end_event = {"type": "end"}
        yield f"data: {json.dumps(end_event)}\n\n"

    except Exception as e:
        logger.exception(f"Error during RAG streaming for session {session_id}:")
        user_error_message = "An error occurred while processing your request."
        if isinstance(e, HTTPException) and hasattr(e, 'detail'):
            user_error_message = e.detail
        error_event = {"type": "error", "content": user_error_message }
        yield f"data: {json.dumps(error_event)}\n\n"
    finally:
        logger.info(f"Finished streaming response for session {session_id}.")

@app.post("/api/v1/chat")
async def chat_with_rag_streaming(query: ChatQuery, request: Request): # Removed response_model for SSE
    logger.info(f"Received streaming query (session: {query.session_id or 'New'}): '{query.message[:50]}...'")
    
    # Basic check for essential RAG components
    if not retriever_instance or not utility_llm_instance : # Add other critical components if needed
        logger.critical("Core RAG components (retriever or utility LLM) not initialized. API cannot serve RAG queries.")
        # This is a server configuration error.
        # We could raise HTTPException here, but stream_rag_response also handles it.
        # However, for intent detection and small talk, some components might still work.
        # Let's allow it to proceed to stream_rag_response which has more granular checks.
        pass

    return StreamingResponse(stream_rag_response(query), media_type="text/event-stream")

# To run this app (save as main_api.py):
# uvicorn main_api:app --reload --port 8000
# You will need to install uvicorn: pip install uvicorn fastapi python-multipart

if __name__ == "__main__":
    logger.info("Starting Uvicorn server for development (with session memory and advanced RAG features)...")
    uvicorn.run("main_api:app", host="0.0.0.0", port=8000, reload=True) 