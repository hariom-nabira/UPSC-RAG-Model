import os
import json # Added for manifest
import logging # For more detailed logging
from typing import Tuple, List, Any # Added Tuple, List, Any
# import streamlit as st # Removed Streamlit import
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader, PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
# ParentDocumentRetriever is no longer used with the direct retriever
# from langchain.retrievers import ContextualCompressionRetriever, ParentDocumentRetriever 
from langchain.retrievers.document_compressors import LLMChainExtractor # Potentially for future compression
from langchain.prompts import PromptTemplate
# InMemoryStore is no longer needed as parent store is removed
# from langchain.storage import InMemoryStore 
from langchain.text_splitter import RecursiveCharacterTextSplitter # Still used in data_processing
from utils import get_logger
from config import (
    DATA_DIR, CHROMA_PERSIST_DIR, TOP_K_RESULTS, 
    LLM_CHAT_MODEL, OPENAI_API_KEY,
    CHUNK_SIZE, CHUNK_OVERLAP 
)
from data_processing import process_documents
from embeddings import get_embedding_model 
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document

logger = get_logger(__name__)

# --- Manifest Helper Functions ---
MANIFEST_FILE_NAME = "processing_manifest.json"
# FORCE_REPROCESS_ONCE has been removed. Reprocessing is now solely based on manifest comparison.

def get_data_dir_state(data_dir_path: str) -> dict:
    """Gets the state of the data directory (file paths and modification timestamps)."""
    state = {}
    if not os.path.exists(data_dir_path):
        logger.warning(f"Data directory '{data_dir_path}' does not exist.")
        return state
    for root, _, files in os.walk(data_dir_path):
        for file in files:
            if file == ".DS_Store": # Explicitly ignore .DS_Store files
                continue
            file_path = os.path.join(root, file)
            try:
                state[file_path] = os.path.getmtime(file_path)
            except Exception as e:
                logger.warning(f"Could not get mtime for {file_path}: {e}")
    return state

def load_manifest(manifest_path: str) -> dict:
    """Loads the processing manifest file."""
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Manifest file {manifest_path} is corrupted. Will re-process.")
            return None # Explicitly return None for clarity
        except Exception as e:
            logger.warning(f"Could not load manifest {manifest_path}: {e}. Will re-process.")
            return None # Explicitly return None for clarity
    logger.info(f"Manifest file {manifest_path} not found. Will process data.")
    return None

def save_manifest(manifest_path: str, state: dict):
    """Saves the current data directory state to the manifest file."""
    try:
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
        with open(manifest_path, "w") as f:
            json.dump(state, f, indent=4)
        logger.info(f"Processing manifest saved to {manifest_path}")
    except Exception as e:
        logger.error(f"Could not save manifest {manifest_path}: {e}")

def log_data_state_differences(current_state: dict, last_state: dict):
    """Logs the differences between two data directory states."""
    current_files = set(current_state.keys())
    last_files = set(last_state.keys())

    added_files = current_files - last_files
    removed_files = last_files - current_files
    common_files = current_files.intersection(last_files)

    changed_files = []
    for f in common_files:
        if current_state[f] != last_state[f]:
            changed_files.append(f)

    if not any([added_files, removed_files, changed_files]):
        logger.info("Manifest Diff - No significant changes detected in data files.")
        if len(current_files) != len(last_files): # Should ideally not happen if sets are same
             logger.warning("Manifest Diff - File sets seem identical by content but lengths differ. This is unexpected.")
        elif current_state.keys() != last_state.keys(): # Check order if sets are same but dicts not equal
            logger.warning("Manifest Diff - File sets and mtimes seem identical, but dictionary key order might differ or other subtle issue.")
        return # Return early if no differences

    if added_files:
        logger.warning(f"Manifest Diff - Added files: {sorted(list(added_files))}")
    if removed_files:
        logger.warning(f"Manifest Diff - Removed files: {sorted(list(removed_files))}")
    if changed_files:
        logger.warning(f"Manifest Diff - Changed mtime for files: {sorted(changed_files)}")
        for f in sorted(changed_files):
            logger.debug(f"  - File: {f}. Current mtime: {current_state[f]}, Last mtime: {last_state.get(f)}")


def get_vectorstore(
    _utility_llm: ChatOpenAI, # Kept for signature consistency, though not directly used if RAPTOR is out
    _embedding_model: OpenAIEmbeddings
) -> Chroma:
    manifest_path = os.path.join(CHROMA_PERSIST_DIR, MANIFEST_FILE_NAME)
    current_data_state = get_data_dir_state(DATA_DIR)
    last_data_state = load_manifest(manifest_path)

    logger.debug(f"Current DATA_DIR state keys: {sorted(list(current_data_state.keys()))}")
    if last_data_state is not None:
        logger.debug(f"Last DATA_DIR state keys from manifest: {sorted(list(last_data_state.keys()))}")
    else:
        logger.debug("No last_data_state loaded from manifest.")

    if not os.path.exists(DATA_DIR):
        try:
            os.makedirs(DATA_DIR)
            logger.warning(f"Data directory '{DATA_DIR}' was missing and has been created. Please add documents.")
        except OSError as e:
            logger.error(f"Failed to create missing data directory '{DATA_DIR}': {e}")
            raise ValueError(f"Data directory '{DATA_DIR}' is missing and could not be created.")

    if not current_data_state: # Check after ensuring DATA_DIR exists
        logger.error(f"Data directory '{DATA_DIR}' is empty. Cannot create or load vector store.")
        # It might be valid to have an empty vector store if data is added later,
        # but for now, we expect initial data.
        # If an empty store is permissible on first run, adjust this logic.
        # For now, we create an empty Chroma store if CHROMA_PERSIST_DIR doesn't exist.
        if not os.path.exists(CHROMA_PERSIST_DIR):
            logger.warning(f"Data directory '{DATA_DIR}' is empty, and no existing Chroma store at '{CHROMA_PERSIST_DIR}'. Creating an empty store.")
            os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
            vectordb = Chroma(
                persist_directory=CHROMA_PERSIST_DIR,
                embedding_function=_embedding_model
            )
            vectordb.persist() # Persist the empty store
            save_manifest(manifest_path, current_data_state) # Save manifest for empty state
            logger.info(f"Empty Chroma vector store created and persisted at {CHROMA_PERSIST_DIR}.")
            return vectordb
        else: # Chroma store exists, but data dir is empty. Load existing store.
             logger.warning(f"Data directory '{DATA_DIR}' is empty, but an existing Chroma store found. Loading it.")
             # This will be handled by the "process_new_data = False" path below.


    process_new_data = True
    # Decision to re-process logic:
    # 1. No manifest? -> Process.
    # 2. Manifest exists, but Chroma dir is empty/missing? -> Process.
    # 3. Manifest exists, Chroma dir exists, but data state differs? -> Process.
    # 4. Manifest exists, Chroma dir exists, data state same? -> Load existing.

    if last_data_state is None:
        logger.info("No processing manifest found. Will process data and create new Chroma store.")
    elif not (os.path.exists(CHROMA_PERSIST_DIR) and os.listdir(CHROMA_PERSIST_DIR)):
        logger.info(f"Manifest found, but Chroma store at {CHROMA_PERSIST_DIR} is missing or empty. Will re-process.")
    elif current_data_state == last_data_state:
        logger.info(f"Data in '{DATA_DIR}' unchanged according to manifest. Loading existing Chroma store.")
        process_new_data = False
    else:
        logger.warning(f"Data in '{DATA_DIR}' has changed or manifest mismatch. Re-processing all documents for Chroma store.")
        log_data_state_differences(current_data_state, last_data_state)
        # If we re-process, it implies deleting the old store to ensure freshness.
        # This is handled within the "if process_new_data:" block.

    try:
        if not process_new_data:
            vectordb = Chroma(
                persist_directory=CHROMA_PERSIST_DIR,
                embedding_function=_embedding_model
            )
            logger.info(f"Successfully loaded existing Chroma vector store from {CHROMA_PERSIST_DIR}.")
        else:
            if not current_data_state: # Double check after all decisions if we are about to process.
                 logger.error(f"Data directory '{DATA_DIR}' is empty. Halting processing to prevent empty vector store creation from this path.")
                 raise ValueError(f"Cannot process new data: Data directory '{DATA_DIR}' is empty.")

            logger.info(f"Loading documents from {DATA_DIR} for new vector store processing...")
            
            raw_docs = []
            if not os.path.exists(DATA_DIR):
                logger.error(f"Data directory {DATA_DIR} does not exist.")
                raise ValueError(f"Data directory {DATA_DIR} does not exist.")

            for root, _, files in os.walk(DATA_DIR):
                for file in files:
                    if file == ".DS_Store": # Explicitly ignore .DS_Store files
                        continue
                    file_path = os.path.join(root, file)
                    try:
                        if file.lower().endswith(".pdf"):
                            logger.info(f"Loading PDF: {file_path} using PyMuPDFLoader")
                            loader = PyMuPDFLoader(file_path)
                        else:
                            logger.info(f"Loading file: {file_path} using UnstructuredFileLoader")
                            loader = UnstructuredFileLoader(file_path, loader_kwargs={"exclude": [".DS_Store"]})
                        
                        loaded_documents = loader.load()
                        if loaded_documents:
                            raw_docs.extend(loaded_documents)
                        else:
                            logger.warning(f"No documents loaded from file: {file_path}")
                    except Exception as e:
                        logger.error(f"Error loading file {file_path}: {e}", exc_info=True)
            
            if not raw_docs:
                logger.error(f"No documents loaded from {DATA_DIR} despite data files being present. Check loader configuration and file types.")
                raise ValueError(f"No documents could be loaded from {DATA_DIR}. Ensure files are readable and supported.")
            logger.info(f"Loaded {len(raw_docs)} raw documents for processing.")

            logger.info("Processing documents to get child documents for vector store...")
            child_documents_for_vectorstore = process_documents(raw_docs, CHUNK_SIZE, CHUNK_OVERLAP)
            
            if not child_documents_for_vectorstore:
                logger.error("No processable child documents were generated. Cannot create vector store. Check document content and processing logic.")
                raise ValueError("No processable child documents were generated from the documents.")

            logger.info(f"Creating new Chroma vector store with {len(child_documents_for_vectorstore)} child documents...")
            import shutil
            if os.path.exists(CHROMA_PERSIST_DIR):
                logger.warning(f"Deleting existing Chroma store at {CHROMA_PERSIST_DIR} for re-processing.")
                shutil.rmtree(CHROMA_PERSIST_DIR)
            os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
            
            vectordb = Chroma( # Initialize fresh Chroma instance
                embedding_function=_embedding_model,
                persist_directory=CHROMA_PERSIST_DIR
            )

            logger.info("Adding documents to Chroma in batches...")
            batch_size = 100
            num_batches = (len(child_documents_for_vectorstore) + batch_size - 1) // batch_size
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(child_documents_for_vectorstore))
                batch_docs = child_documents_for_vectorstore[start_idx:end_idx]
                if batch_docs:
                    logger.info(f"Adding batch {i+1}/{num_batches} to Chroma ({len(batch_docs)} documents)...")
                    vectordb.add_documents(documents=batch_docs)
            
            vectordb.persist()
            logger.info("All batches added and Chroma vector store persisted.")
            save_manifest(manifest_path, current_data_state)

        return vectordb

    except Exception as e:
        logger.error(f"Error in vector store initialization: {type(e).__name__} - {e}", exc_info=True)
        # Specific error for empty data directory if we reach here and it's the cause.
        if not current_data_state and "Cannot process new data" not in str(e):
             logger.error("Vector store initialization failed, and data directory is empty. This might be the root cause.")
        raise RuntimeError(f"Failed to initialize vector store: {e}")


class FilteredRetriever(BaseRetriever):
    vectordb: Chroma
    k: int
    score_threshold: float

    def __init__(self, vectordb: Chroma, k: int, score_threshold: float = 0.7, **caller_kwargs: Any):
        init_data = {
            "vectordb": vectordb,
            "k": k,
            "score_threshold": score_threshold,
            **caller_kwargs
        }
        super().__init__(**init_data)
        
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        results_with_scores = self.vectordb.similarity_search_with_score(
            query,
            k=self.k * 2 
        )
        
        if not results_with_scores:
            logger.warning(f"No documents found by similarity search for query: '{query}'")
            return []

        filtered_documents = [
            doc for doc, score in results_with_scores
            if score <= self.score_threshold 
        ]
        
        final_documents = filtered_documents[:self.k]
        
        if not final_documents and results_with_scores: # Only log fallback if there were initial results
            logger.warning(
                f"No documents met similarity threshold {self.score_threshold} for query: '{query}'. "
                f"Falling back to top {self.k} of {len(results_with_scores)} initially retrieved unfiltered documents."
            )
            return [doc for doc, _score in results_with_scores[:self.k]]
            
        return final_documents

def get_retriever(vectordb: Chroma, utility_llm: ChatOpenAI): # utility_llm is not used by FilteredRetriever but kept for API consistency if other retrievers are added.
    """
    Returns a FilteredRetriever that uses a similarity score threshold.
    """
    logger.info(f"Initializing FilteredRetriever with k={TOP_K_RESULTS} and score_threshold=0.7.")
    return FilteredRetriever(vectordb, TOP_K_RESULTS, score_threshold=0.7)

def get_memory():
    return ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key="answer"
    )

QA_CHAIN_PROMPT_TEMPLATE = """You are a helpful AI assistant, specifically designed to support users preparing for the UPSC (Union Public Service Commission) examinations.
Your primary instruction is to derive answers **solely and exclusively** from the 'Context' provided below.

**Core Directives:**
1. **PROVIDE COMPREHENSIVE INFORMATION**: Extract and present ALL relevant details from the context that answer the question. Be thorough and complete, including ALL facts, definitions, explanations, and examples from the context.
2. **QUOTE EXACT TEXT**: When definitions, key concepts, or explanations are present in the context, use the EXACT wording from the context. Do not summarize or paraphrase key definitions.
3. **COMPLETE ANSWERS**: If the context contains a multi-part answer or extensive information about the topic, include ALL parts in your response. Never truncate important information.
4. **STRUCTURE FOR READABILITY**: Use clear formatting including bullet points, numbered lists, and paragraphs to organize information from the context in a readable way.
5. **MAINTAIN SPECIFIC DETAILS**: Include all specific details like statistics, dates, names, and technical terminology exactly as they appear in the context.
6. **COMPREHENSIVE DEFINITIONS**: For "What is X?" questions, provide the complete definition and ALL additional context and details about X that appear in the retrieved documents.
7. **MISSING INFORMATION**: Only state information is missing if you've thoroughly checked all provided context and confirmed the answer isn't there.
8. **SPECIFIC CITATIONS**: When presenting information, mention the specific source document (e.g., "According to the February 2024 document from 'source_file_name.pdf'").

**Input Sections:**

Context:
{context}

Chat History:
{chat_history}

H: {question}

**Your Response (AI Assistant - UPSC Helper):**"""

QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["chat_history", "context", "question"], template=QA_CHAIN_PROMPT_TEMPLATE
)

def get_conversational_qa_chain(llm: ChatOpenAI, retriever, memory):
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT},
        return_source_documents=True,
        return_generated_question=True # Kept for now, but can be reviewed if generated_question is not used
    )

def get_current_chat_llm():
    return ChatOpenAI(
        model_name=LLM_CHAT_MODEL,
        temperature=0.5, # Consider making this configurable
        max_tokens=2000, # Consider making this configurable
        streaming=True,
        openai_api_key=OPENAI_API_KEY
    ) 