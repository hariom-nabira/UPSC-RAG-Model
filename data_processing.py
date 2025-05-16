import os
from typing import List, Dict, Tuple
import numpy as np
from sklearn.cluster import KMeans
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import filter_complex_metadata
from utils import get_logger
from config import CHUNK_SIZE, CHUNK_OVERLAP
import uuid
from langchain.storage import InMemoryStore

logger = get_logger(__name__)

def create_summary(docs: List[Document], llm: ChatOpenAI) -> str:
    summary_prompt_template = PromptTemplate(
        template="Please provide a concise summary of the following text, focusing on the key points and main ideas:\n\n{text}\n\nSummary:",
        input_variables=["text"]
    )
    combined_text = "\n\n".join([doc.page_content for doc in docs])
    max_summary_input_length = 30000 # Consider moving to config if frequently changed
    if len(combined_text) > max_summary_input_length:
        combined_text = combined_text[:max_summary_input_length]
        logger.warning(f"Combined text for summary truncated to {max_summary_input_length} chars.")
    response = llm.invoke(summary_prompt_template.format(text=combined_text))
    return response.content

def create_raptor_tree(docs: List[Document], llm: ChatOpenAI, local_embedding_model: OpenAIEmbeddings, max_clusters: int = 3) -> List[Document]:
    if not docs or len(docs) <= max_clusters:
        return docs
    try:
        embeddings, valid_docs = [], []
        for doc in docs:
            try:
                emb = local_embedding_model.embed_query(doc.page_content)
                embeddings.append(emb)
                valid_docs.append(doc)
            except Exception as e:
                logger.error(f"Error creating embedding for doc (source: {doc.metadata.get('source', 'Unknown')}): {e}")
                continue
        if not embeddings or len(embeddings) < 2:
            return valid_docs
        
        embeddings_array = np.array(embeddings)
        n_clusters = min(max_clusters, len(valid_docs))
        if n_clusters < 2: return valid_docs
            
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        try: 
            clusters = kmeans.fit_predict(embeddings_array)
        except Exception as e:
            logger.error(f"Error during KMeans: {e}. Returning valid_docs."); 
            return valid_docs

        clustered_docs_map: Dict[int, List[Document]] = {}
        for i, cluster_id_val in enumerate(clusters):
            cluster_id = int(cluster_id_val)
            clustered_docs_map.setdefault(cluster_id, []).append(valid_docs[i])
        
        summarized_docs = []
        for cluster_id, cluster_docs_list in clustered_docs_map.items():
            if len(cluster_docs_list) > 1:
                try:
                    summary_content = create_summary(cluster_docs_list, llm)
                    first_orig_meta = cluster_docs_list[0].metadata
                    display_name_prefix = first_orig_meta.get('file_name', 'multiple documents')
                    
                    summary_display_name = f"Summary ({len(cluster_docs_list)} sources, e.g., {display_name_prefix})"

                    summary_doc = Document(
                        page_content=summary_content,
                        metadata={
                            'source_identifier': f'summary_cluster_{cluster_id}',
                            'display_name': summary_display_name,
                            'page_label': '', 
                            'chunk_text': summary_content, # For hover tooltip
                            'is_summary': True,
                            'file_name': f'summary_cluster_{cluster_id}', # For internal consistency
                            'original_doc_sources': [d.metadata.get('source_identifier', d.metadata.get('source', 'Unknown')) for d in cluster_docs_list]
                        }
                    )
                    summarized_docs.append(summary_doc)
                except Exception as e:
                    logger.error(f"Error creating summary for cluster {cluster_id}: {e}")
                    summarized_docs.extend(cluster_docs_list) # Add original docs if summary fails
            else:
                summarized_docs.extend(cluster_docs_list)
        return summarized_docs
    except Exception as e:
        logger.error(f"Error in RAPTOR tree creation: {e}"); 
        return docs # Return original docs on error

def process_documents(
    raw_docs: List[Document],
    # Parameters for RAPTOR if we re-enable it:
    # llm_for_raptor: ChatOpenAI, 
    # emb_model_for_raptor: OpenAIEmbeddings
) -> Tuple[List[Document], InMemoryStore]:
    """
    Processes raw documents into child documents for vector store and parent documents for a docstore.
    Implements parent/child chunking.
    Temporarily bypasses RAPTOR for simplicity in setting up ParentDocumentRetriever.
    """
    logger.info(f"Starting parent/child document processing for {len(raw_docs)} raw documents.")
    parent_doc_store = InMemoryStore()
    all_child_documents = []

    # Define the child splitter here or pass it as an argument
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,  # CHUNK_SIZE from config
        chunk_overlap=CHUNK_OVERLAP, # CHUNK_OVERLAP from config
        length_function=len,
        separators=["\n\n", "\n", ". ", "!", "?", ",", " ", ""] # Added space to separators
    )

    for i, raw_doc in enumerate(raw_docs):
        parent_id = str(uuid.uuid4())
        
        # Prepare metadata for the parent document
        parent_metadata = {
            "doc_id": parent_id, # Unique ID for this parent document
            "source": raw_doc.metadata.get("source", f"unknown_source_{i}"),
            "file_name": os.path.basename(raw_doc.metadata.get("source", f"unknown_source_{i}")),
            # TODO: Add more parent-level metadata if available (e.g., document title, subject)
            # This might require more sophisticated parsing in the loading stage if not in raw_doc.metadata
        }
        # Ensure page_content is string
        parent_page_content = raw_doc.page_content if isinstance(raw_doc.page_content, str) else str(raw_doc.page_content)
        
        # Create a Document object first, then filter its metadata
        parent_doc = Document(page_content=parent_page_content, metadata=parent_metadata.copy())
        filtered_parent_doc = filter_complex_metadata([parent_doc])[0]
        parent_doc_store.mset([(parent_id, filtered_parent_doc)])
        logger.debug(f"Stored parent document {parent_id} from {parent_metadata['file_name']}")

        # Split the raw document into child chunks
        # Ensure raw_doc content is suitable for splitting (it should be a Document object)
        child_chunks = child_splitter.split_documents([raw_doc]) 
        
        temp_child_docs_for_parent = []
        for child_idx, chunk_doc in enumerate(child_chunks):
            child_meta = chunk_doc.metadata.copy() if hasattr(chunk_doc, 'metadata') and chunk_doc.metadata is not None else {}
            child_meta["parent_id"] = parent_id
            child_meta["original_source"] = parent_metadata["source"] # Keep original source
            child_meta["file_name"] = parent_metadata["file_name"] # Inherit filename

            # Extract page number if available from loader (e.g., UnstructuredFileLoader)
            page_number = child_meta.get("page", child_meta.get("page_number"))
            if page_number is not None:
                try:
                    child_meta['page_label'] = f"p. {int(page_number)}"
                except ValueError:
                    child_meta['page_label'] = str(page_number) # if not int-able
            
            # Add a snippet of chunk_text for easier identification/debugging
            chunk_content = chunk_doc.page_content if isinstance(chunk_doc.page_content, str) else str(chunk_doc.page_content)
            child_meta['chunk_preview'] = chunk_content[:100] + "..." if len(chunk_content) > 100 else chunk_content
            
            # Metadata for structured retrieval (can be expanded)
            # e.g., child_meta['subject'] = extract_subject(chunk_doc.page_content)
            # e.g., child_meta['keywords'] = extract_keywords_llm(chunk_doc.page_content, llm_utility)

            temp_doc = Document(page_content=chunk_content, metadata=child_meta.copy())
            final_child_meta = filter_complex_metadata([temp_doc])[0].metadata
            processed_child_doc = Document(page_content=chunk_content, metadata=final_child_meta)
            temp_child_docs_for_parent.append(processed_child_doc)
        
        all_child_documents.extend(temp_child_docs_for_parent)
        logger.info(f"Processed parent doc {parent_id} ({parent_metadata['file_name']}), created {len(temp_child_docs_for_parent)} child chunks.")

    # --- RAPTOR Integration Point (Optional - currently bypassed) ---
    # If you want to re-integrate RAPTOR:
    # 1. It could run on `all_child_documents`.
    # 2. The output of RAPTOR (summary_docs) would need to ensure `parent_id` is correctly propagated
    #    or handled if summaries span multiple parents.
    # 3. The `all_child_documents` to be returned would then be these RAPTOR-processed docs.
    # Example:
    # if llm_for_raptor and emb_model_for_raptor and all_child_documents:
    #     logger.info(f"Applying RAPTOR processing to {len(all_child_documents)} child documents...")
    #     # You'd need to ensure RAPTOR summaries correctly link back to parent_ids if they are the final child docs
    #     # This might require modification of create_raptor_tree to accept/use parent_id.
    #     all_child_documents = create_raptor_tree(all_child_documents, llm_for_raptor, emb_model_for_raptor, max_clusters=5)
    #     logger.info(f"RAPTOR processing resulted in {len(all_child_documents)} documents for vector store.")
    # else:
    #     logger.info("RAPTOR processing skipped (dependencies not provided or no child documents).")

    if not all_child_documents:
        logger.warning("No child documents were generated after processing. Returning empty list.")
        return [], parent_doc_store # Return empty list and the (possibly empty) parent store

    logger.info(f"Total child documents for vector store: {len(all_child_documents)}")
    logger.info(f"Parent document store contains {len(list(parent_doc_store.yield_keys()))} entries.")
    
    return all_child_documents, parent_doc_store 