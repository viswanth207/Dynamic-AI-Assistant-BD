from typing import List, Optional
from langchain_core.documents import Document
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_community.embeddings import HuggingFaceEmbeddings
from pymongo import MongoClient
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
        
        # Initialize MongoDB Client for Vector Search
        self.mongo_uri = os.getenv("MONGODB_URL")
        self.db_name = "dynamic_assistant_db"
        self.collection_name = "embeddings"
        self.client = MongoClient(self.mongo_uri)
        self.collection = self.client[self.db_name][self.collection_name]
        
        logger.info("Vector Store Manager initialized with MongoDB Atlas")
    
    def create_vector_store(self, documents: List[Document]) -> MongoDBAtlasVectorSearch:
        if not documents:
            raise ValueError("Cannot create vector store with empty documents")
        
        try:
            logger.info(f"Adding {len(documents)} documents to MongoDB Atlas Vector Store...")
            
            # MongoDB Atlas Vector Search handles batching internally, 
            # but we can do it explicitly if needed. 
            # For now, we trust the library but we might need to batch if 10k explodes.
            
            vector_store = MongoDBAtlasVectorSearch.from_documents(
                documents=documents,
                embedding=self.embeddings,
                collection=self.collection,
                index_name="vector_index" 
            )
            
            logger.info("Documents added to Vector Store successfully")
            return vector_store
            
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise ValueError(f"Failed to create vector store: {str(e)}")
    
    def similarity_search(
        self, 
        vector_store: MongoDBAtlasVectorSearch,  # Type hint updated
        query: str, 
        k: int = 4
    ) -> List[Document]:
        try:
            # Note: vector_store object might be reconstructed, 
            # so we ensure we look at the right place.
            # Actually, we don't need to pass 'vector_store' object around as much 
            # because the state is in DB. But to keep API consistent for now:
            
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
        vector_store: MongoDBAtlasVectorSearch, 
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

    # No longer needed: save_vector_store, load_vector_store 
    # (Because DB is the storage)

    def get_vector_store(self):
        """Helper to get an existing vector store object connected to DB"""
        return MongoDBAtlasVectorSearch(
            collection=self.collection,
            embedding=self.embeddings,
            index_name="vector_index"
        )
