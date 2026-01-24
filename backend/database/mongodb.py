from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
import os
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)

MONGODB_URL = os.getenv("MONGODB_URL", "mongodb+srv://viswanthchirumamilla:7HV63Eswef8h1PXF@cluster0.v0rkh.mongodb.net/?appName=Cluster0")
DATABASE_NAME = os.getenv("DATABASE_NAME", "dynamic_assistant_db")

# Async MongoDB client for FastAPI
async_client: AsyncIOMotorClient = None
async_db = None

# Sync client for initialization checks
sync_client: MongoClient = None


async def connect_to_mongo():
    """Connect to MongoDB on startup"""
    global async_client, async_db
    try:
        async_client = AsyncIOMotorClient(MONGODB_URL)
        async_db = async_client[DATABASE_NAME]
        
        # Test connection
        await async_client.admin.command('ping')
        
        # Mask password for logging
        masked_url = MONGODB_URL
        if "@" in MONGODB_URL:
            part1 = MONGODB_URL.split("@")[1]
            masked_url = f"mongodb+srv://****@{part1}"
            
        logger.info(f"Connected to MongoDB: {DATABASE_NAME} at {masked_url}")
        
        # Create indexes
        await create_indexes()
        
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {str(e)}")
        raise


async def close_mongo_connection():
    """Close MongoDB connection on shutdown"""
    global async_client
    if async_client:
        async_client.close()
        logger.info("Closed MongoDB connection")


async def create_indexes():
    """Create database indexes for better performance"""
    try:
        # Users collection indexes
        await async_db.users.create_index("email", unique=True)
        
        # Assistants collection indexes
        await async_db.assistants.create_index("user_id")
        await async_db.assistants.create_index("assistant_id", unique=True)
        
        # Chat history indexes (optional)
        await async_db.chat_history.create_index([("user_id", 1), ("assistant_id", 1)])
        
        logger.info("Database indexes created successfully")
    except Exception as e:
        logger.warning(f"Error creating indexes: {str(e)}")


def get_database():
    """Get database instance"""
    return async_db
