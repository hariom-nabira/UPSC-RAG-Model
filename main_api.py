import uvicorn
import json # For serializing data for SSE
import uuid # For generating session IDs
# import traceback # No longer needed directly, logger.exception handles it
from enum import Enum
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks # Added Request and BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse # For SSE
from pydantic import BaseModel
from typing import List, Dict, Any, AsyncGenerator, Optional, Union # Added AsyncGenerator, Optional, Union
from langchain.memory import ConversationBufferMemory # Ensure this is imported if not already
from langchain_core.language_models import BaseLanguageModel
from langchain.chains import LLMChain # LLMChain will be replaced by LCEL
from langchain.prompts import PromptTemplate
# Unused imports for ParentDocumentRetriever, CohereRerank were here.

# Import RAG components (adjust paths/imports as necessary based on your project structure)
# These will be the functions and classes you've already defined for your RAG system
from config import OPENAI_API_KEY, LLM_CHAT_MODEL, EMBEDDING_MODEL, EMBEDDING_DIMENSIONS, LLM_UTILITY_MODEL, DATA_DIR
from embeddings import get_embedding_model, get_utility_llm
# Updated import: get_vectorstore instead of get_vectorstore_and_parent_store
# Removed global_parent_doc_store as it's no longer used
from retrieval import (
    get_vectorstore, 
    get_retriever, 
    get_memory, 
    get_conversational_qa_chain, 
    get_current_chat_llm
)
# from security import check_openai_api_key # Removed import
from utils import get_logger # Import the logger

# Add these new imports for file serving
import os.path
from urllib.parse import unquote

# Initialize logger
logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="UPSC RAG API",
    description="API for interacting with the UPSC RAG model, incorporating advanced retrieval strategies.",
    version="0.2.1" # Version update reflecting parent store removal
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
# parent_doc_store_instance = None # Removed, no longer used
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
    # Sources are handled via SSE; if a non-streaming endpoint uses this model, sources can be added.

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
    global embedding_model_instance, utility_llm_instance, vectorstore_instance, retriever_instance
    global intent_detection_chain, small_talk_chain
    # parent_doc_store_instance removed from globals
    
    logger.info("Checking API key...")
    if not OPENAI_API_KEY:
        logger.critical("OpenAI API Key is missing. Set OPENAI_API_KEY environment variable.")
        raise RuntimeError("OpenAI API Key is missing. Set OPENAI_API_KEY environment variable.")
    logger.info("API key check passed.")

    logger.info("Initializing RAG components and utility LLMs on startup...")
    embedding_model_instance = get_embedding_model()
    utility_llm_instance = get_utility_llm()
    
    logger.info("Initializing vectorstore...") # Log message updated
    # Call the renamed get_vectorstore, which now only returns the vectorstore
    vectorstore_instance = get_vectorstore(
        _utility_llm=utility_llm_instance, 
        _embedding_model=embedding_model_instance
    )
    
    if vectorstore_instance is None:
        logger.critical("Failed to initialize vectorstore.") # Log message updated
        raise RuntimeError("Failed to initialize vectorstore.") # Error message updated
    logger.info("Vectorstore (ChromaDB for child docs) initialized.") # Log message updated
    
    # Get the retriever; it no longer needs parent_doc_store_instance
    retriever_instance = get_retriever(vectorstore_instance, utility_llm_instance)
    logger.info("RAG retriever initialized.") # Log message updated

    intent_list = [intent.value for intent in UserIntent]
    intent_prompt = PromptTemplate(
        input_variables=["user_message", "intent_list_str"],
        template=INTENT_DETECTION_PROMPT_TEMPLATE
    )
    intent_detection_chain = LLMChain(
        llm=utility_llm_instance,
        prompt=intent_prompt,
        verbose=False
    ).with_config({"run_name": "IntentDetection"})
    logger.info("Intent detection chain initialized.")

    small_talk_prompt_template = "You are a friendly and helpful AI assistant designed for UPSC exam aspirants. Keep your responses concise and encouraging. User: {user_message}\\nAssistant:"
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
        event_str = f"data: {json.dumps(session_id_event)}\n\n" # Store it in a variable
        logger.info(f"SERVER YIELDING RAW: >>>{event_str}<<<") # Log it clearly
        yield event_str

    try:
        # --- 1. Intent Detection ---
        intent_list_str = ", ".join([f"'{intent.value}'" for intent in UserIntent])
        intent_result = await intent_detection_chain.arun(user_message=query.message, intent_list_str=intent_list_str)
        try:
            detected_intent = UserIntent(intent_result.strip().lower().replace("'", ""))
        except ValueError:
            logger.warning(f"Unknown intent '{intent_result}' detected for message: {query.message}. Defaulting to RAG_QUERY.")
            detected_intent = UserIntent.RAG_QUERY 
        
        intent_event = {"type": "intent", "content": detected_intent.value}
        event_str = f"data: {json.dumps(intent_event)}\n\n"
        logger.info(f"SERVER YIELDING RAW INTENT: >>>{event_str}<<<")
        yield event_str
        logger.info(f"Detected intent for session {session_id}: {detected_intent.value}")

        # --- 2. Task-Specific Processing based on Intent ---
        if detected_intent == UserIntent.GREETING:
            response_content = "Hello! How can I help you with your UPSC preparation today?"
            token_event = {"type": "token", "content": response_content}
            event_str = f"data: {json.dumps(token_event)}\n\n"
            logger.info(f"SERVER YIELDING RAW GREETING: >>>{event_str}<<<")
            yield event_str
        elif detected_intent == UserIntent.CLEAR_SESSION:
            if session_id in active_sessions:
                del active_sessions[session_id]
                logger.info(f"Cleared session: {session_id}")
            session_memory = get_memory() 
            active_sessions[session_id] = session_memory 
            response_content = "Session cleared. Let's start a fresh conversation!"
            token_event = {"type": "token", "content": response_content}
            event_str = f"data: {json.dumps(token_event)}\n\n"
            logger.info(f"SERVER YIELDING RAW CLEAR_SESSION: >>>{event_str}<<<")
            yield event_str
        elif detected_intent == UserIntent.SMALL_TALK:
            response_content = await small_talk_chain.arun(user_message=query.message)
            token_event = {"type": "token", "content": response_content}
            event_str = f"data: {json.dumps(token_event)}\n\n"
            logger.info(f"SERVER YIELDING RAW SMALL_TALK: >>>{event_str}<<<")
            yield event_str

        elif detected_intent in [UserIntent.RAG_QUERY, UserIntent.COMPLEX_RAG_QUERY, UserIntent.SUMMARY_REQUEST, UserIntent.UNKNOWN]:
            if detected_intent == UserIntent.UNKNOWN:
                logger.info(f"Intent was UNKNOWN, attempting to treat as RAG_QUERY for message: {query.message}")
            
            if not retriever_instance or not vectorstore_instance:
                error_msg = "RAG system not fully initialized."
                logger.error(f"Attempt to use RAG while not initialized. Session: {session_id}")
                error_event = {"type": "error", "content": error_msg}
                event_str = f"data: {json.dumps(error_event)}\n\n"
                logger.info(f"SERVER YIELDING RAW ERROR: >>>{event_str}<<<")
                yield event_str
                return

            current_chat_llm = get_current_chat_llm()
            
            qa_chain = get_conversational_qa_chain(
                llm=current_chat_llm,
                retriever=retriever_instance, 
                memory=session_memory
            )

            debug_docs = await retriever_instance.aget_relevant_documents(query.message)
            logger.info(f"Retrieved {len(debug_docs)} documents for query: {query.message}")
            
            for idx, doc in enumerate(debug_docs):
                source_path = doc.metadata.get('source', doc.metadata.get('original_source', 'Unknown'))
                page_for_fragment = doc.metadata.get('page') 
                page_info_display = doc.metadata.get('page_label', doc.metadata.get('page_number_str', ''))
                if page_for_fragment is None and page_info_display: 
                    page_for_fragment = page_info_display 
                
                file_name = doc.metadata.get('file_name', os.path.basename(source_path) if source_path else 'Unknown')
                
                logger.info(f"Doc {idx+1}: Source={source_path}, File={file_name}, Page={page_info_display}")
                logger.info(f"Content preview: {doc.page_content[:150]}")
                
                meta_keys = list(doc.metadata.keys())
                logger.info(f"Available metadata: {meta_keys}")
                for key in ['page', 'page_number', 'page_number_str', 'page_label']:
                    if key in doc.metadata:
                        logger.info(f"Metadata - {key}: {doc.metadata[key]}")

            full_answer = ""
            source_documents = []
            
            async for chunk in qa_chain.astream({"question": query.message, "chat_history": []}):
                if isinstance(chunk, dict):
                    if "answer" in chunk and chunk["answer"] is not None:
                        token = chunk["answer"]
                        full_answer += token
                        token_event = {"type": "token", "content": token}
                        event_str = f"data: {json.dumps(token_event)}\n\n"
                        # logger.info(f"SERVER YIELDING RAW RAG TOKEN: >>>{event_str[:100]}...<<<") # Log less verbosely for tokens
                        yield event_str
                    if "source_documents" in chunk and chunk["source_documents"]:
                        source_documents = chunk["source_documents"] 
                        # logger.info(f"Received {len(source_documents)} source documents during streaming for session {session_id}.")
                elif isinstance(chunk, str): 
                    full_answer += chunk
                    token_event = {"type": "token", "content": chunk}
                    event_str = f"data: {json.dumps(token_event)}\n\n"
                    # logger.info(f"SERVER YIELDING RAW RAG TOKEN (str): >>>{event_str[:100]}...<<<") # Log less verbosely for tokens
                    yield event_str

            # Fallback to retrieve source documents if not found during streaming
            if not source_documents:
                logger.info(f"Source documents not populated during streaming for session {session_id}. Attempting ainvoke fallback.")
                try:
                    # Use the same input as astream for consistency, memory is part of the chain.
                    final_result = await qa_chain.ainvoke({"question": query.message, "chat_history": []})
                    source_documents = final_result.get("source_documents", [])
                    if source_documents:
                        logger.info(f"Successfully retrieved {len(source_documents)} source documents via ainvoke fallback for session {session_id}.")
                    else:
                        logger.warning(f"No source documents found even after ainvoke fallback for session {session_id}.")
                except Exception as e_invoke:
                    logger.error(f"Error during ainvoke fallback for sources in session {session_id}: {e_invoke}", exc_info=True)

            if source_documents:
                formatted_sources = []
                for doc in source_documents:
                    formatted_sources.append({
                        "page_content": doc.page_content, 
                        "metadata": doc.metadata 
                    })
                sources_event = {"type": "sources", "content": formatted_sources}
                event_str = f"data: {json.dumps(sources_event)}\n\n"
                logger.info(f"SERVER YIELDING RAW SOURCES: >>>{event_str}<<<")
                yield event_str
        else: 
            response_content = "I'm not sure how to handle that. Could you try rephrasing?"
            token_event = {"type": "token", "content": response_content}
            event_str = f"data: {json.dumps(token_event)}\n\n"
            logger.info(f"SERVER YIELDING RAW UNHANDLED: >>>{event_str}<<<")
            yield event_str

        end_event = {"type": "end"}
        event_str = f"data: {json.dumps(end_event)}\n\n"
        logger.info(f"SERVER YIELDING RAW END: >>>{event_str}<<<")
        yield event_str

    except Exception as e:
        logger.exception(f"Error during RAG streaming for session {session_id}:")
        user_error_message = "An error occurred while processing your request."
        if isinstance(e, HTTPException) and hasattr(e, 'detail'):
            user_error_message = e.detail
        error_event = {"type": "error", "content": user_error_message }
        event_str = f"data: {json.dumps(error_event)}\n\n"
        logger.info(f"SERVER YIELDING RAW ERROR: >>>{event_str}<<<")
        yield event_str
    finally:
        logger.info(f"Finished streaming response for session {session_id}.")

@app.post("/api/v1/chat")
async def chat_with_rag_streaming(query: ChatQuery, request: Request): 
    logger.info(f"Received streaming query (session: {query.session_id or 'New'}): '{query.message[:50]}...'" )
    
    if not retriever_instance or not utility_llm_instance: 
        logger.critical("Core RAG components (retriever or utility LLM) not initialized. API cannot serve RAG queries.")
        # This case should ideally be prevented by failing loudly during startup_event if initialization fails.
        # However, as a safeguard at the endpoint level:
        raise HTTPException(status_code=503, detail="Service Unavailable: RAG components not ready.")

    return StreamingResponse(stream_rag_response(query), media_type="text/event-stream")

@app.get("/api/v1/documents/{file_path:path}")
async def get_document(file_path: str):
    """
    Serve document files from the data directory.
    This endpoint allows the frontend to open source documents when clicked.
    """
    # Decode the URL-encoded path
    decoded_path = unquote(file_path)
    logger.info(f"Document request received for path: {decoded_path}")
    
    # Handle paths that might be relative to DATA_DIR or already include it
    if not decoded_path.startswith(DATA_DIR):
        full_path = os.path.join(DATA_DIR, decoded_path)
    else:
        full_path = decoded_path
    
    # Normalize the path to resolve any .. or . components
    full_path = os.path.normpath(os.path.abspath(full_path))
    data_dir_full = os.path.normpath(os.path.abspath(DATA_DIR))
    
    logger.info(f"Attempting to serve document: {full_path}")
    logger.info(f"DATA_DIR is: {data_dir_full}")
    
    # Security check: Ensure the path refers to a file within DATA_DIR
    if not full_path.startswith(data_dir_full) and not os.path.isfile(full_path):
        logger.warning(f"Attempted to access file outside data directory: {full_path}")
        raise HTTPException(status_code=403, detail="Access denied: File path is outside the data directory")
    
    if not os.path.exists(full_path):
        logger.warning(f"Requested document not found: {full_path}")
        
        # Try one more approach - check if it's directly in the workspace root
        alternative_path = os.path.abspath(decoded_path)
        if os.path.exists(alternative_path) and os.path.isfile(alternative_path):
            logger.info(f"Found document at alternative path: {alternative_path}")
            return FileResponse(alternative_path)
            
        raise HTTPException(status_code=404, detail=f"Document not found: {decoded_path}")
    
    if not os.path.isfile(full_path):
        logger.warning(f"Requested path is not a file: {full_path}")
        raise HTTPException(status_code=400, detail="The requested path is not a file")
    
    logger.info(f"Serving document: {full_path}")
    return FileResponse(full_path)

@app.get("/api/v1/debug/retrieval")
async def debug_retrieval(query: str):
    """
    Debug endpoint to test document retrieval for a specific query.
    This helps diagnose retrieval issues without going through the full chat process.
    """
    if not retriever_instance:
        raise HTTPException(status_code=500, detail="Retriever not initialized")
        
    logger.info(f"Debug retrieval request for query: {query}")
    
    try:
        # Get relevant documents for the query
        docs = await retriever_instance.aget_relevant_documents(query)
        
        # Format results for better readability
        results = []
        for idx, doc in enumerate(docs):
            source_path = doc.metadata.get('source', doc.metadata.get('original_source', 'Unknown'))
            results.append({
                "index": idx,
                "content": doc.page_content,
                "source": source_path,
                "metadata": {k: v for k, v in doc.metadata.items() if k not in ['source', 'original_source']}
            })
            
        return {
            "query": query,
            "num_docs_retrieved": len(docs),
            "documents": results
        }
    except Exception as e:
        logger.exception(f"Error in debug retrieval for query '{query}':")
        raise HTTPException(status_code=500, detail=f"Error in retrieval: {str(e)}")

@app.get("/api/v1/debug/fullcontent")
async def debug_fullcontent(query: str):
    """
    Verbose debug endpoint that shows the complete content of retrieved documents.
    This helps diagnose content issues when documents are retrieved but not presented properly.
    """
    if not retriever_instance:
        raise HTTPException(status_code=500, detail="Retriever not initialized")
        
    logger.info(f"Full content debug retrieval for query: {query}")
    
    try:
        # Get relevant documents for the query
        docs = await retriever_instance.aget_relevant_documents(query)
        
        # Print full content to logs
        for idx, doc in enumerate(docs):
            source_path = doc.metadata.get('source', doc.metadata.get('original_source', 'Unknown'))
            # Use 'page' for fragment, then 'page_number_str' for display
            page_value_for_fragment = doc.metadata.get('page')
            page_display = doc.metadata.get('page_label', doc.metadata.get('page_number_str', 'Unknown page'))
            
            logger.info(f"FULL CONTENT - Doc {idx+1}/{len(docs)}")
            logger.info(f"Source: {source_path} | Page Display: {page_display} (Fragment Value: {page_value_for_fragment})")
            logger.info(f"Content: {doc.page_content}")
            logger.info(f"Metadata: {doc.metadata}")
            logger.info("-" * 50)
        
        # Format results for better readability in the API response
        results = []
        for idx, doc in enumerate(docs):
            source_path = doc.metadata.get('source', doc.metadata.get('original_source', 'Unknown'))
            page_to_use = doc.metadata.get('page') # Front-end expects 'page' for fragment
            file_name = doc.metadata.get('file_name', os.path.basename(source_path) if source_path else 'Unknown')
            results.append({
                "index": idx,
                "file_name": file_name,
                "page": page_to_use, # Ensure this is the one Message.js will use
                "content": doc.page_content,
                "metadata": doc.metadata # Pass all metadata
            })
            
        return {
            "query": query,
            "num_docs_retrieved": len(docs),
            "documents": results
        }
    except Exception as e:
        logger.exception(f"Error in full content debug retrieval for query '{query}':")
        raise HTTPException(status_code=500, detail=f"Error in retrieval: {str(e)}")

# To run this app (save as main_api.py):
# uvicorn main_api:app --reload --port 8000
# You will need to install uvicorn: pip install uvicorn fastapi python-multipart

if __name__ == "__main__":
    logger.info("Starting Uvicorn server for development...") # Updated log message
    uvicorn.run("main_api:app", host="0.0.0.0", port=8000, reload=True) 