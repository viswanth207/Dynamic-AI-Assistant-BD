from pydantic import BaseModel, EmailStr, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from bson import ObjectId


class PyObjectId(ObjectId):
    """Custom ObjectId type for Pydantic"""
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type="string")


# User Models
class UserBase(BaseModel):
    email: EmailStr


class UserCreate(UserBase):
    password: str = Field(..., min_length=6, max_length=72)


class UserLogin(BaseModel):
    email: EmailStr
    password: str = Field(..., max_length=72)


class UserInDB(UserBase):
    id: str = Field(alias="_id")
    password_hash: str
    created_at: datetime
    updated_at: datetime

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class UserResponse(BaseModel):
    id: str
    email: str
    created_at: datetime

    class Config:
        from_attributes = True


# Assistant Models
class AssistantInDB(BaseModel):
    id: Optional[str] = Field(alias="_id", default=None)
    user_id: str
    assistant_id: str
    name: str
    data_source_type: str
    data_source_url: Optional[str] = None
    custom_instructions: str
    enable_statistics: bool = False
    enable_alerts: bool = False
    enable_recommendations: bool = False
    documents_count: int
    vector_store_path: str
    attributes: List[str] = []
    sample_questions: List[str] = []
    graph_data: Dict[str, Any] = {}
    created_at: datetime

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class AssistantResponse(BaseModel):
    assistant_id: str
    name: str
    data_source_type: str
    documents_count: int
    created_at: datetime


# Chat History Models
class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime


class ChatHistoryInDB(BaseModel):
    id: Optional[str] = Field(alias="_id", default=None)
    user_id: str
    assistant_id: str
    messages: List[ChatMessage] = []
    created_at: datetime
    updated_at: datetime

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


# Token Models
class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    user_id: Optional[str] = None
    email: Optional[str] = None
