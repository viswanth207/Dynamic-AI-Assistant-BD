from backend.database.mongodb import get_database
from backend.database.models import UserCreate, UserInDB, AssistantInDB, ChatHistoryInDB
from datetime import datetime
from bson import ObjectId
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


# User CRUD
async def create_user(user: UserCreate, password_hash: str) -> UserInDB:
    """Create a new user in the database"""
    db = get_database()
    
    user_doc = {
        "email": user.email,
        "password_hash": password_hash,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    
    result = await db.users.insert_one(user_doc)
    user_doc["_id"] = str(result.inserted_id)
    
    return UserInDB(**user_doc)


async def get_user_by_email(email: str) -> Optional[UserInDB]:
    """Get user by email"""
    db = get_database()
    user_doc = await db.users.find_one({"email": email})
    
    if user_doc:
        user_doc["_id"] = str(user_doc["_id"])
        return UserInDB(**user_doc)
    return None


async def get_user_by_id(user_id: str) -> Optional[UserInDB]:
    """Get user by ID"""
    db = get_database()
    user_doc = await db.users.find_one({"_id": ObjectId(user_id)})
    
    if user_doc:
        user_doc["_id"] = str(user_doc["_id"])
        return UserInDB(**user_doc)
    return None


# Assistant CRUD
async def create_assistant(assistant_data: dict) -> AssistantInDB:
    """Create a new assistant in the database"""
    db = get_database()
    
    assistant_doc = {
        **assistant_data,
        "created_at": datetime.utcnow()
    }
    
    result = await db.assistants.insert_one(assistant_doc)
    assistant_doc["_id"] = str(result.inserted_id)
    
    return AssistantInDB(**assistant_doc)


async def get_assistant_by_id(assistant_id: str, user_id: str) -> Optional[AssistantInDB]:
    """Get assistant by ID and verify ownership"""
    db = get_database()
    assistant_doc = await db.assistants.find_one({
        "assistant_id": assistant_id,
        "user_id": user_id
    })
    
    if assistant_doc:
        assistant_doc["_id"] = str(assistant_doc["_id"])
        return AssistantInDB(**assistant_doc)
    return None


async def get_user_assistants(user_id: str) -> List[AssistantInDB]:
    """Get all assistants for a user"""
    db = get_database()
    cursor = db.assistants.find({"user_id": user_id}).sort("created_at", -1)
    
    assistants = []
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        assistants.append(AssistantInDB(**doc))
    
    return assistants


async def delete_assistant(assistant_id: str, user_id: str) -> bool:
    """Delete an assistant"""
    db = get_database()
    result = await db.assistants.delete_one({
        "assistant_id": assistant_id,
        "user_id": user_id
    })
    return result.deleted_count > 0


# Chat History CRUD
async def save_chat_message(user_id: str, assistant_id: str, role: str, content: str):
    """Save a chat message to history"""
    db = get_database()
    
    message = {
        "role": role,
        "content": content,
        "timestamp": datetime.utcnow()
    }
    
    # Update or insert chat history
    await db.chat_history.update_one(
        {"user_id": user_id, "assistant_id": assistant_id},
        {
            "$push": {"messages": message},
            "$set": {"updated_at": datetime.utcnow()},
            "$setOnInsert": {"created_at": datetime.utcnow()}
        },
        upsert=True
    )


async def get_chat_history(user_id: str, assistant_id: str, limit: int = 50) -> List[dict]:
    """Get chat history for an assistant"""
    db = get_database()
    
    chat_doc = await db.chat_history.find_one(
        {"user_id": user_id, "assistant_id": assistant_id}
    )
    
    if chat_doc and "messages" in chat_doc:
        # Return last N messages
        return chat_doc["messages"][-limit:]
    return []
