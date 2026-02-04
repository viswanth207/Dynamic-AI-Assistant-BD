import os
import time
from pymongo import MongoClient
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from dotenv import load_dotenv

load_dotenv()

print("Initializing Embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2", 
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

print("Connecting to Mongo...")
mongo_url = os.getenv("MONGODB_URL")
client = MongoClient(mongo_url)
collection = client["dynamic_assistant_db"]["embeddings"]

vector_store = MongoDBAtlasVectorSearch(
    collection=collection,
    embedding=embeddings,
    index_name="vector_index"
)

# 1. Search WITHOUT Filter
print("\n--- TEST 1: Search WITHOUT Filter ---")
try:
    results = vector_store.similarity_search("car model", k=2)
    print(f"Found {len(results)} documents.")
    if results:
        print(f"Sample: {results[0].page_content[:100]}...")
        # Capture an ID for Test 2
        test_id = results[0].metadata.get('assistant_id')
        print(f"Captured Assistant ID: {test_id}")
    else:
        print("No documents found. Index might be empty or not active.")
        test_id = None
except Exception as e:
    print(f"Search Failed: {e}")
    test_id = None

# 2. Search WITH Filter
if test_id:
    print(f"\n--- TEST 2: Search WITH Filter (assistant_id='{test_id}') ---")
    try:
        # Pre-filter using MQL
        filter_dict = {"assistant_id": {"$eq": test_id}}
        results_f = vector_store.similarity_search("car model", k=2, pre_filter=filter_dict)
        print(f"Found {len(results_f)} documents.")
        if len(results_f) == 0:
            print("!!! FAILURE !!! Filter blocked results. ATLAS INDEX DEFINITION IS WRONG.")
            print("You MUST add 'assistant_id' to your Atlas Search Index definition as type 'filter'.")
        else:
            print("SUCCESS! Filtering works.")
    except Exception as e:
        print(f"Filtered Search Failed: {e}")
