from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import logging
import os

logger = logging.getLogger(__name__)


class VectorStoreManager:
    
    def __init__(self):
        logger.info("Initializing HuggingFace embeddings...")
        
import torch
        # Limit CPU threads to prevent server freeze
        torch.set_num_threads(1)
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        logger.info("Embeddings initialized successfully")
    
    def create_vector_store(self, documents: List[Document]) -> FAISS:
        if not documents:
            raise ValueError("Cannot create vector store with empty documents")
        
        try:
            logger.info(f"Creating vector store with {len(documents)} documents (Batch mode)")
            
            # Process in small batches to keep server responsive
            batch_size = 500  # Process 500 docs at a time
            vector_store = None
            
            total_batches = (len(documents) + batch_size - 1) // batch_size
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{total_batches}...")
                
                if vector_store is None:
                    vector_store = FAISS.from_documents(batch, self.embeddings)
                else:
                    batch_vector_store = FAISS.from_documents(batch, self.embeddings)
                    vector_store.merge_from(batch_vector_store)
            
            logger.info("Vector store created successfully")
            return vector_store
            
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise ValueError(f"Failed to create vector store: {str(e)}")
    
    def similarity_search(
        self, 
        vector_store: FAISS, 
        query: str, 
        k: int = 4
    ) -> List[Document]:
        try:
            logger.info(f"Performing similarity search for: {query[:50]}...")
            
            results = vector_store.similarity_search(
                query=query,
                k=k
            )
            
            logger.info(f"Found {len(results)} relevant documents")
            return results
            
        except Exception as e:
            logger.error(f"Error during similarity search: {str(e)}")
            return []
    
    def similarity_search_with_score(
        self, 
        vector_store: FAISS, 
        query: str, 
        k: int = 4
    ) -> List[tuple[Document, float]]:
        try:
            logger.info(f"Performing similarity search with scores for: {query[:50]}...")
            
            results = vector_store.similarity_search_with_score(
                query=query,
                k=k
            )
            
            logger.info(f"Found {len(results)} relevant documents with scores")
            return results
            
        except Exception as e:
            logger.error(f"Error during similarity search: {str(e)}")
            return []
    
    def save_vector_store(self, vector_store: FAISS, path: str) -> None:
        try:
            os.makedirs(path, exist_ok=True)
            vector_store.save_local(path)
            logger.info(f"Vector store saved to {path}")
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            raise
    
    def load_vector_store(self, path: str) -> FAISS:
        try:
            vector_store = FAISS.load_local(
                path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info(f"Vector store loaded from {path}")
            return vector_store
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            raise
