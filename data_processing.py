import os
from typing import List, Dict, Tuple, Any
import numpy as np
from sklearn.cluster import KMeans
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores.utils import filter_complex_metadata # No longer using this
from utils import get_logger
# CHUNK_SIZE, CHUNK_OVERLAP will now be passed as arguments to process_documents
# from config import CHUNK_SIZE, CHUNK_OVERLAP 
import uuid

logger = get_logger(__name__)

def create_summary(docs: List[Document], llm: ChatOpenAI) -> str:
    summary_prompt_template = PromptTemplate(
        template="Please provide a concise summary of the following text, focusing on the key points and main ideas:\\n\\n{text}\\n\\nSummary:",
        input_variables=["text"]
    )
    combined_text = "\\n\\n".join([doc.page_content for doc in docs])
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

def process_documents(raw_docs: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    logger.info(f"Starting document processing for {len(raw_docs)} raw documents with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}.")
    all_child_documents = []
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap, 
        length_function=len,
        separators=["\\n\\n", "\\n", ". ", "!", "?", ",", " ", ""]
    )

    for i, raw_doc in enumerate(raw_docs):
        raw_doc_source = raw_doc.metadata.get("source", f"unknown_source_{i}")
        raw_doc_filename = os.path.basename(raw_doc_source)
        # parent_doc_id for grouping chunks from the same original doc, not for a separate store
        parent_doc_id_for_grouping = str(uuid.uuid4()) 

        child_chunks_from_splitter = child_splitter.split_documents([raw_doc])
        logger.info(f"Processing '{raw_doc_filename}': Split into {len(child_chunks_from_splitter)} child chunks.")
        
        for child_idx, chunk_doc_from_splitter in enumerate(child_chunks_from_splitter):
            # Start with a fresh, controlled metadata dictionary
            final_child_meta: Dict[str, Any] = {
                "source": raw_doc_source,
                "file_name": raw_doc_filename,
                "parent_doc_id_group": parent_doc_id_for_grouping 
            }

            # Log raw metadata from the splitter for this specific chunk
            raw_splitter_meta = chunk_doc_from_splitter.metadata or {}
            logger.debug(f"  Chunk {child_idx} (File: {raw_doc_filename}): Raw metadata from splitter: {raw_splitter_meta}")

            # Determine original document loader type based on available metadata keys if possible
            # PyMuPDFLoader adds 'file_path', 'page', 'total_pages', 'format', 'title', 'author', etc.
            # Unstructured typically adds 'source', 'filename', sometimes 'page_number' or 'category'.
            is_pymupdf_doc = 'file_path' in raw_splitter_meta and 'total_pages' in raw_splitter_meta

            page_number_for_fragment = None # This will be used for the 'page' key in final metadata
            page_label_display = "" # This will be used for the 'page_label' key (user-facing)

            if is_pymupdf_doc:
                # PyMuPDFLoader provides 0-indexed 'page'.
                if 'page' in raw_splitter_meta and raw_splitter_meta['page'] is not None:
                    try:
                        page_number_zero_indexed = int(raw_splitter_meta['page'])
                        page_number_for_fragment = page_number_zero_indexed # Keep 0-indexed for potential fragment use
                        page_label_display = f"p. {page_number_zero_indexed + 1}" # Display as 1-indexed
                        logger.debug(f"    PyMuPDF: Extracted page '{page_number_zero_indexed}' (0-indexed). Label: '{page_label_display}'")
                    except (ValueError, TypeError):
                        logger.warning(f"    PyMuPDF: Could not parse page from 'page' key (value: '{raw_splitter_meta['page']}').")
                else:
                    logger.debug("    PyMuPDF: 'page' key not found in metadata.")
            else: # Handle UnstructuredFileLoader or other loaders
                possible_page_keys = ['page_number', 'page', 'page_num', 'pagenum', 'pg_num', 'pg']
                for key in possible_page_keys:
                    if key in raw_splitter_meta and raw_splitter_meta[key] is not None:
                        try:
                            # Assume these might be 1-indexed or other formats, try to make them page numbers for fragments
                            # This part might need refinement based on what Unstructured actually provides for non-PDFs
                            parsed_page = int(raw_splitter_meta[key]) # Attempt to get an int
                            page_number_for_fragment = parsed_page # Use as is, or adjust if known to be 0/1 indexed
                            page_label_display = f"p. {parsed_page}" # Assume it's 1-indexed for display if not from PyMuPDF
                            logger.debug(f"    Unstructured: Extracted page '{parsed_page}' from key '{key}'. Label: '{page_label_display}'")
                            break 
                        except (ValueError, TypeError):
                            page_label_display = f"Page: {raw_splitter_meta[key]} (raw)"
                            logger.warning(f"    Unstructured: Could not parse page from key '{key}' (value: '{raw_splitter_meta[key]}'). Using raw label: '{page_label_display}'")
                            # page_number_for_fragment might remain None here if parsing fails
                            break
            
            if page_number_for_fragment is None: # Fallback if no page info was extracted by any method
                # Estimate page based on chunk index. Note: CHUNK_SIZE_FOR_PAGE_ESTIMATE might not be ideal for all loaders.
                # This estimation is less likely to be hit if PyMuPDF works for PDFs.
                estimated_page_one_indexed = (child_idx // CHUNK_SIZE_FOR_PAGE_ESTIMATE if CHUNK_SIZE_FOR_PAGE_ESTIMATE > 0 else child_idx // 3) + 1
                page_number_for_fragment = estimated_page_one_indexed # Or 0-indexed if preferred: estimated_page_one_indexed - 1
                page_label_display = f"~p. {estimated_page_one_indexed} (est.)"
                logger.debug(f"    Fallback: No page info extracted, estimated as page {page_number_for_fragment}. Label: '{page_label_display}'")

            # Add to final_child_meta - these are the fields that will go into Chroma
            if page_number_for_fragment is not None:
                # The frontend's SourceLink component was seen to use doc.metadata.get('page')
                # PyMuPDF provides 0-indexed 'page'. Let's store this.
                final_child_meta['page'] = page_number_for_fragment 
            if page_label_display: # Should always have some value by now
                final_child_meta['page_label'] = page_label_display
            
            # Add chunk preview to final_child_meta
            chunk_content = chunk_doc_from_splitter.page_content
            final_child_meta['chunk_preview'] = chunk_content[:100] + "..." if len(chunk_content) > 100 else chunk_content
            
            logger.debug(f"  Chunk {child_idx} (File: {raw_doc_filename}): Final constructed metadata for Chroma: {final_child_meta}")

            # Create the Document with the carefully constructed metadata
            doc_for_chroma = Document(page_content=chunk_content, metadata=final_child_meta)
            all_child_documents.append(doc_for_chroma)
        
        logger.info(f"Finished processing source doc '{raw_doc_filename}'.")

    if not all_child_documents:
        logger.warning("No child documents were generated. Returning empty list.")
        return []

    logger.info(f"Total child documents for vector store: {len(all_child_documents)}.")
    return all_child_documents

# Dummy constant for page estimation, can be refined or made configurable
CHUNK_SIZE_FOR_PAGE_ESTIMATE = 3 