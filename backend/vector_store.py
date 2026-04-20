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
        
        # Limit CPU threads to prevent server freeze on single-core instances
        try:
            import torch
            torch.set_num_threads(1)
        except ImportError:
            pass
            
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.index_base_dir = "vector_stores"
        os.makedirs(self.index_base_dir, exist_ok=True)
        
        logger.info("Vector Store Manager initialized with local FAISS (multi-tenant)")
    
    def create_vector_store(self, documents: List[Document]) -> FAISS:
        if not documents:
            raise ValueError("Cannot create vector store with empty documents")
        
        assistant_id = documents[0].metadata.get("assistant_id", "default")
        index_path = os.path.join(self.index_base_dir, assistant_id)
        
        try:
            logger.info(f"Adding {len(documents)} documents to FAISS Vector Store {index_path}...")
            
            vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
                
            vector_store.save_local(index_path)
            
            logger.info("Documents added to Vector Store successfully")
            return vector_store
            
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise ValueError(f"Failed to create vector store: {str(e)}")
    
    def similarity_search(
        self, 
        vector_store: FAISS,
        query: str, 
        k: int = 4,
        filter: Optional[dict] = None
    ) -> List[Document]:
        try:
            logger.info(f"Performing similarity search for: {query[:50]}... Filter bypassed due to directory isolation")
            
            results = vector_store.similarity_search(
                query=query,
                k=k,
                fetch_k=10000
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
                k=k,
                fetch_k=10000
            )
            
            logger.info(f"Found {len(results)} relevant documents with scores")
            return results
            
        except Exception as e:
            logger.error(f"Error during similarity search: {str(e)}")
            return []

    def get_vector_store(self, assistant_id: str):
        """Helper to get an existing vector store object"""
        index_path = os.path.join(self.index_base_dir, assistant_id)
        if os.path.exists(index_path):
            return FAISS.load_local(index_path, self.embeddings, allow_dangerous_deserialization=True)
        else:
            doc = Document(page_content="initialization", metadata={"assistant_id": assistant_id})
            return FAISS.from_documents([doc], self.embeddings)
