import os
from typing import Tuple # Added Tuple
# import streamlit as st # Removed Streamlit import
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.retrievers import ContextualCompressionRetriever, ParentDocumentRetriever # Added ParentDocumentRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.prompts import PromptTemplate
from langchain.storage import InMemoryStore # Added InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter # Added for ParentDocumentRetriever
from utils import get_logger
from config import (
    DATA_DIR, CHROMA_PERSIST_DIR, TOP_K_RESULTS, 
    LLM_CHAT_MODEL, OPENAI_API_KEY,
    CHUNK_SIZE, CHUNK_OVERLAP # Added CHUNK_SIZE, CHUNK_OVERLAP
)
from data_processing import process_documents

logger = get_logger(__name__)

# Global store for parent documents, initialized by get_vectorstore_and_parent_store
# This is a simplification for this example. In a more complex app, consider a dedicated store manager.
global_parent_doc_store: InMemoryStore = None 

def get_vectorstore_and_parent_store(
    _utility_llm: ChatOpenAI, 
    _embedding_model: OpenAIEmbeddings
) -> Tuple[Chroma, InMemoryStore]:
    global global_parent_doc_store
    try:
        if os.path.exists(CHROMA_PERSIST_DIR) and os.listdir(CHROMA_PERSIST_DIR):
            logger.info(f"Loading existing vector store from {CHROMA_PERSIST_DIR}")
            # If vectorstore exists, we assume parent_doc_store also needs to be rebuilt or loaded
            # For simplicity, we will rebuild it on each startup if Chroma exists by reprocessing.
            # A more robust solution would persist/load the parent_doc_store or have a flag to re-process.
            logger.warning("Existing Chroma store found. Re-processing documents to rebuild parent_doc_store for consistency.")
            # Fall through to re-processing to ensure parent_doc_store is populated.

        logger.info(f"Creating new vector store and parent doc store. Loading documents from {DATA_DIR}")
        if not os.path.exists(DATA_DIR): 
            os.makedirs(DATA_DIR)
            logger.warning(f"Data directory '{DATA_DIR}' was missing and has been created. Please add documents.")
            raise ValueError(f"Data directory '{DATA_DIR}' was empty at startup. Please add documents.")
        
        if not os.listdir(DATA_DIR):
            logger.warning(f"Data directory '{DATA_DIR}' is empty. Cannot create vector store.")
            raise ValueError(f"Data directory '{DATA_DIR}' is empty. Please add documents to process.")

        loader = DirectoryLoader(
            path=DATA_DIR, glob="**/*.*", loader_cls=UnstructuredFileLoader,
            show_progress=True, use_multithreading=True, silent_errors=True
        )
        raw_docs = loader.load()

        if not raw_docs: 
            logger.error(f"No documents loaded from {DATA_DIR}. Ensure files are present and readable.")
            raise ValueError(f"No documents could be loaded from {DATA_DIR}.")
        
        logger.info(f"Loaded {len(raw_docs)} raw documents. Processing for parent/child structure...")
        
        # process_documents now returns child_docs for Chroma and the parent_store
        # We are not passing llm_for_raptor and emb_model_for_raptor, so RAPTOR is bypassed as per data_processing.py
        child_documents_for_vectorstore, parent_store = process_documents(raw_docs)
        
        if not child_documents_for_vectorstore: 
            logger.error("No processable child documents were generated.")
            raise ValueError("No processable child documents were generated from the documents.")

        global_parent_doc_store = parent_store # Store the parent_store globally
        logger.info(f"Created {len(child_documents_for_vectorstore)} child documents for vector store.")
        logger.info(f"Parent document store initialized with {len(list(global_parent_doc_store.yield_keys()))} entries.")
        
        # Create Chroma vector store from child documents
        vectordb = Chroma.from_documents(
            documents=child_documents_for_vectorstore, 
            embedding=_embedding_model, 
            persist_directory=CHROMA_PERSIST_DIR
        )
        logger.info("Chroma vector store created and persisted with child documents.")

        return vectordb, global_parent_doc_store # Return both

    except Exception as e:
        logger.error(f"Error in vectorstore and parent store initialization: {type(e).__name__} - {e}", exc_info=True)
        # st.error(f"Error initializing vector store: {e}"); # Removed Streamlit error
        # Re-raise the exception so the FastAPI startup event can catch it
        raise RuntimeError(f"Failed to initialize vectorstore/parent store: {e}")

def get_retriever(vectordb: Chroma, utility_llm: ChatOpenAI, parent_store: InMemoryStore):
    logger.info(f"Initializing ParentDocumentRetriever with {TOP_K_RESULTS} results.")
    
    # The child splitter should be the same as used in data_processing.py
    # This is used by ParentDocumentRetriever internally if it needs to re-split parent docs (though less common if children already stored)
    # However, it's good practice to provide it.
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\\n\\n", "\\n", ". ", "! ", "? ", ",", " ", ""] 
    )

    parent_retriever = ParentDocumentRetriever(
        vectorstore=vectordb, # Chroma store with child document embeddings
        docstore=parent_store,    # InMemoryStore with parent document full content
        child_splitter=child_splitter, # Splitter instance
        id_key="parent_id", # Explicitly set the ID key for parent lookup
        search_kwargs={"k": TOP_K_RESULTS} # How many child docs to fetch initially
    )

    # Re-ranking with LLMChainExtractor (as you had before)
    # This will run on the *parent documents* retrieved by ParentDocumentRetriever
    logger.info("Wrapping ParentDocumentRetriever with ContextualCompressionRetriever (LLMChainExtractor).")
    compressor = LLMChainExtractor.from_llm(utility_llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=parent_retriever
    )
    
    return compression_retriever # Return the compression retriever

def get_memory():
    return ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key="answer"
    )

QA_CHAIN_PROMPT_TEMPLATE = """You are a helpful AI assistant, specifically designed to support users preparing for the UPSC (Union Public Service Commission) examinations.
Your primary instruction is to derive answers **solely and exclusively** from the 'Context' provided below.

**Core Directives:**
1.  **Strict Context Adherence:** Your entire answer MUST be based ONLY on the information explicitly found within the 'Context' section. Do NOT use any external knowledge or pre-trained information.
2.  **Direct Answer from Context:** If the 'Context' directly answers the 'Human's question', your response must be a direct extraction or a close paraphrase of that information. Clearly indicate if the information comes from the context.
3.  **Information Not in Context:** If the 'Context' does NOT contain the information to answer the 'Human's question', you MUST state: "Based on the provided documents, I cannot answer this question." Do NOT attempt to infer, guess, or use general knowledge.
4.  **No External Synthesis:** Do NOT synthesize information with your broader UPSC-related knowledge. If the context provides fragments, only present those fragments if they directly relate to the question; do not combine them with external knowledge to form a more complete answer.
5.  **Clarity and Structure:** If the 'Context' provides information that can be structured (e.g., bullet points, lists for facts, arguments, steps), use that structure. Only include details like names, dates, articles, schemes, and statistics if they are explicitly present in the 'Context'.

**Input Sections:**

Context:
{context}

Chat History:
{chat_history}

Human: {question}

**Your Response (AI Assistant - UPSC Helper):**"""

QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["chat_history", "context", "question"], template=QA_CHAIN_PROMPT_TEMPLATE
)

def get_conversational_qa_chain(llm: ChatOpenAI, retriever, memory):
    return ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=retriever, # This will now be the compression_retriever (ParentDocumentRetriever + LLMChainExtractor)
        memory=memory,
        combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT},
        return_source_documents=True
    )

def get_current_chat_llm():
    return ChatOpenAI(
        model_name=LLM_CHAT_MODEL, 
        temperature=0.5, 
        max_tokens=2000, 
        streaming=True, # Streaming is enabled here, FastAPI endpoint can leverage this later
        openai_api_key=OPENAI_API_KEY
    ) 