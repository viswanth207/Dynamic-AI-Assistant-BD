from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from typing import Dict, Optional
import uuid
import os
import logging
from datetime import datetime
from dotenv import load_dotenv
import shutil
from sse_starlette.sse import EventSourceResponse

from backend.models import (
    AssistantCreateRequest,
    AssistantCreateResponse,
    ChatRequest,
    ChatResponse,
    AssistantInfo,
    ErrorResponse,
    HealthResponse,
    DataSourceType
)
from backend.assistant_engine import AssistantEngine
from backend.data_loader import DataLoader
from backend.vector_store import VectorStoreManager
from backend.database.mongodb import connect_to_mongo, close_mongo_connection
from backend.database import crud
from backend.auth.dependencies import get_current_user
from backend.routes import auth
from backend.database.models import UserInDB

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Dynamic AI Assistant API",
    description="Create and chat with custom AI assistants dynamically",
    version="1.0.0"
)

# Add MongoDB lifecycle events
@app.on_event("startup")
async def startup_db_client():
    await connect_to_mongo()

@app.on_event("shutdown")
async def shutdown_db_client():
    await close_mongo_connection()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://3.80.23.96:8000",
        "https://rag.neuraltrixai.app",
        "https://dynamic-ai-assistant-bd.onrender.com",
        "https://vercel.app",
        "https://data-mind-theta.vercel.app"
    ],
    allow_origin_regex="https://.*\\.vercel\\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include auth router
app.include_router(auth.router)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY not found in environment variables")
    raise ValueError("GROQ_API_KEY must be set in .env file")

try:
    assistant_engine = AssistantEngine(
        groq_api_key=GROQ_API_KEY,
        model_name=GROQ_MODEL_NAME
    )
    data_loader = DataLoader()
    vector_store_manager = VectorStoreManager()
    logger.info("Assistant engine initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize assistant engine: {str(e)}")
    raise

assistants_store: Dict[str, Dict] = {}

# Serve React build in production, fallback to old frontend for development
FRONTEND_BUILD_DIR = "frontend/dist"
FRONTEND_DEV_DIR = "frontend"

if os.path.exists(FRONTEND_BUILD_DIR):
    # Production: Serve React build
    app.mount("/static", StaticFiles(directory=os.path.join(FRONTEND_BUILD_DIR, "static")), name="static")
    
    @app.get("/")
    async def serve_frontend():
        return FileResponse(os.path.join(FRONTEND_BUILD_DIR, "index.html"))
elif os.path.exists(FRONTEND_DEV_DIR):
    # Development: Serve old HTML files
    app.mount("/static", StaticFiles(directory=FRONTEND_DEV_DIR), name="static")
    
    @app.get("/")
    async def serve_frontend():
        return FileResponse(os.path.join(FRONTEND_DEV_DIR, "index.html"))
else:
    logger.warning("Frontend directory not found. Skipping static file serving. This is expected in API-only mode (e.g. AWS Backend + Vercel Frontend).")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat()
    )


@app.post("/api/assistants/create", response_model=AssistantCreateResponse)
async def create_assistant(
    background_tasks: BackgroundTasks,
    name: str = Form(...),
    data_source_type: str = Form(..., regex="^(csv|json|url)$"),
    data_source_url: Optional[str] = Form(None),
    custom_instructions: str = Form(
        "You are a helpful AI assistant. Analyze the data, identify patterns, and answer questions. You can make predictions based on data patterns when asked about hypothetical scenarios."
    ),
    enable_statistics: bool = Form(False),
    enable_alerts: bool = Form(False),
    enable_recommendations: bool = Form(False),
    file: Optional[UploadFile] = File(None),
    current_user: UserInDB = Depends(get_current_user)
):
    try:
        logger.info(f"Creating assistant: {name} for user: {current_user.email}")
        
        if data_source_type not in ["csv", "json", "url"]:
            raise HTTPException(400, "Invalid data_source_type")
        
        assistant_id = str(uuid.uuid4())
        
        documents = []
        file_path = None
        
        if data_source_type == "url":
            if not data_source_url:
                raise HTTPException(400, "data_source_url required for URL type")
            
            logger.info(f"Loading data from URL: {data_source_url}")
            documents = DataLoader.load_from_url(data_source_url)
        
        else:
            if not file:
                raise HTTPException(400, "File required for CSV/JSON type")
            
            file_size = 0
            content = await file.read()
            file_size = len(content) / (1024 * 1024)
            
            if file_size > MAX_FILE_SIZE_MB:
                raise HTTPException(
                    400, 
                    f"File size exceeds {MAX_FILE_SIZE_MB}MB limit"
                )
            
            # Create user-specific upload directory
            user_upload_dir = os.path.join(UPLOAD_DIR, current_user.id)
            os.makedirs(user_upload_dir, exist_ok=True)
            
            file_path = os.path.join(user_upload_dir, f"{assistant_id}_{file.filename}")
            with open(file_path, "wb") as f:
                f.write(content)
            
            logger.info(f"File saved: {file_path}")
            
            if data_source_type == "csv":
                documents = DataLoader.load_from_csv(file_path)
            elif data_source_type == "json":
                documents = DataLoader.load_from_json(file_path)
        
        if not documents:
            raise HTTPException(400, "No data could be loaded from the source")
        
        logger.info(f"Loaded {len(documents)} documents")
        
        # Extract attributes from metadata
        attributes = []
        if documents:
            # Get keys from the first few documents' metadata, excluding internal keys
            exclude_keys = {'source', 'row_number', 'item_number', 'chunk', 'type', 'title', 'heading'}
            all_keys = set()
            for doc in documents[:5]:
                all_keys.update(doc.metadata.keys())
            attributes = sorted([k for k in all_keys if k not in exclude_keys])

        # Generate sample questions based on attributes
        sample_questions = []
        if attributes:
            if data_source_type in ["csv", "json"]:
                sample_questions = [
                    f"What can you tell me about the {attributes[0]} in this data?",
                    f"Which item has the highest {attributes[1 if len(attributes) > 1 else 0]}?",
                    f"Compare the different values of {attributes[0]}.",
                    f"Give me a summary of the dataset based on {', '.join(attributes[:2])}."
                ]
            else: # URL
                sample_questions = [
                    "What is the main topic of this page?",
                    "Can you summarize the key points?",
                    "What are the specific details mentioned about this topic?",
                    "Who are the main people or entities mentioned?"
                ]
        else:
            sample_questions = [
                "Summarize the provided data.",
                "What are the key insights from this dataset?",
                "Are there any notable patterns in the data?",
                "What's the most interesting thing you found?"
            ]

        # Prepare initial assistant data (without heavy vector store yet)
        # Note: We save it to DB immediately so it appears in the list.
        # The vector store will be built in background.
        
        graph_data = DataLoader.generate_graph_insights(file_path, data_source_type)

        assistant_data = {
            "user_id": current_user.id,
            "assistant_id": assistant_id,
            "name": name,
            "data_source_type": data_source_type,
            "data_source_url": data_source_url,
            "custom_instructions": custom_instructions,
            "enable_statistics": enable_statistics,
            "enable_alerts": enable_alerts,
            "enable_recommendations": enable_recommendations,
            "documents_count": len(documents),
            "vector_store_path": user_upload_dir if data_source_type != "url" else "",
            "attributes": attributes,
            "sample_questions": sample_questions,
            "graph_data": graph_data,
            "created_at": datetime.utcnow().isoformat()
        }
        await crud.create_assistant(assistant_data)

        # Helper to run heavy task in background
        def process_heavy_task(assistant_id, name, documents, custom_instructions, enable_statistics, enable_alerts, enable_recommendations):
             try:
                 logger.info(f"Background: Starting vector store creation for {assistant_id}...")
                 assistant_config = assistant_engine.create_assistant(
                    assistant_id=assistant_id,
                    name=name,
                    documents=documents,
                    custom_instructions=custom_instructions,
                    enable_statistics=enable_statistics,
                    enable_alerts=enable_alerts,
                    enable_recommendations=enable_recommendations
                 )
                 assistants_store[assistant_id] = assistant_config
                 logger.info(f"Background: Vector store created for {assistant_id}")
             except Exception as e:
                 logger.error(f"Background Task Failed for {assistant_id}: {str(e)}")

        # Start Background Task
        background_tasks.add_task(
            process_heavy_task,
            assistant_id, name, documents, custom_instructions, 
            enable_statistics, enable_alerts, enable_recommendations
        )
        
        logger.info(f"Assistant created (Processing in background): {assistant_id}")
        
        logger.info(f"Assistant created: {assistant_id}")
        
        return AssistantCreateResponse(
            assistant_id=assistant_id,
            name=name,
            data_source_type=data_source_type,
            documents_loaded=len(documents),
            created_at=assistant_data["created_at"],
            message="Assistant created successfully! You can now start chatting."
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating assistant: {str(e)}")
        raise HTTPException(500, f"Failed to create assistant: {str(e)}")


@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_assistant(
    request: ChatRequest,
    current_user: UserInDB = Depends(get_current_user)
):
    import time
    start_time = time.time()
    
    try:
        logger.info(f"Chat request for assistant: {request.assistant_id} by user: {current_user.email}")
        
        # Verify user owns this assistant
        assistant_db = await crud.get_assistant_by_id(request.assistant_id, current_user.id)
        if not assistant_db:
            raise HTTPException(404, "Assistant not found or access denied")
        
        # Load assistant config if not in memory
        if request.assistant_id not in assistants_store:
            load_start = time.time()
            logger.info(f"Loading assistant {request.assistant_id} from database...")
            
            # Get the user's upload directory
            user_upload_dir = os.path.join(UPLOAD_DIR, current_user.id)
            
            # Find files associated with this assistant
            assistant_files = []
            if os.path.exists(user_upload_dir):
                for file in os.listdir(user_upload_dir):
                    if file.startswith(f"{request.assistant_id}_"):
                        assistant_files.append(os.path.join(user_upload_dir, file))
            
            if not assistant_files:
                raise HTTPException(400, "Assistant files not found. Please recreate the assistant.")
            
            # Load documents from all assistant files
            documents = []
            for file_path in assistant_files:
                file_ext = os.path.splitext(file_path)[1].lower()
                try:
                    if file_ext == '.csv':
                        docs = DataLoader.load_from_csv(file_path)
                    elif file_ext == '.json':
                        docs = DataLoader.load_from_json(file_path)
                    else:
                        logger.warning(f"Unsupported file type: {file_ext}")
                        continue
                    documents.extend(docs)
                except Exception as e:
                    logger.error(f"Error loading file {file_path}: {str(e)}")
                    
            if not documents:
                raise HTTPException(400, "Failed to load assistant documents.")
            
            # Create vector store
            vector_store = vector_store_manager.create_vector_store(documents)
            
            # Build system instructions using assistant engine's method
            system_instructions = assistant_engine._build_system_instructions(
                custom_instructions=assistant_db.custom_instructions,
                enable_statistics=assistant_db.enable_statistics,
                enable_alerts=assistant_db.enable_alerts,
                enable_recommendations=assistant_db.enable_recommendations
            )
            
            # Restore assistant config
            assistant_config = {
                "assistant_id": request.assistant_id,
                "name": assistant_db.name,
                "custom_instructions": assistant_db.custom_instructions,
                "system_instructions": system_instructions,
                "vector_store": vector_store,
                "documents_count": len(documents),
                "enable_statistics": assistant_db.enable_statistics,
                "enable_alerts": assistant_db.enable_alerts,
                "enable_recommendations": assistant_db.enable_recommendations,
                "created_at": assistant_db.created_at
            }
            
            assistants_store[request.assistant_id] = assistant_config
            load_time = time.time() - load_start
            logger.info(f"Assistant {request.assistant_id} loaded in {load_time:.2f}s")
        else:
            logger.info(f"Using cached assistant {request.assistant_id}")
        
        assistant_config = assistants_store[request.assistant_id]
        
        # Call LLM
        llm_start = time.time()
        result = assistant_engine.chat(
            assistant_config=assistant_config,
            user_message=request.message
        )
        llm_time = time.time() - llm_start
        logger.info(f"LLM response received in {llm_time:.2f}s")
        
        # Save chat history
        await crud.save_chat_message(current_user.id, request.assistant_id, "user", request.message)
        await crud.save_chat_message(current_user.id, request.assistant_id, "assistant", result["response"])
        
        total_time = time.time() - start_time
        logger.info(f"Total chat request time: {total_time:.2f}s")
        
        return ChatResponse(
            assistant_id=request.assistant_id,
            user_message=request.message,
            assistant_response=result["response"],
            sources_used=result["sources_used"],
            timestamp=result["timestamp"]
        )
    
    except Exception as e:
        logger.error(f"Error during chat: {str(e)}")
        raise HTTPException(500, f"Chat failed: {str(e)}")


@app.post("/api/chat/stream")
async def chat_stream(
    request: ChatRequest,
    current_user: UserInDB = Depends(get_current_user)
):
    try:
        logger.info(f"Stream chat request for assistant: {request.assistant_id}")
        
        # Verify user owns this assistant
        assistant_db = await crud.get_assistant_by_id(request.assistant_id, current_user.id)
        if not assistant_db:
            raise HTTPException(404, "Assistant not found or access denied")
            
        # Load assistant config if not in memory (same logic as regular chat)
        if request.assistant_id not in assistants_store:
            # ... (Reuse loading logic or extract to helper if preferred, strictly copying for safety now)
            user_upload_dir = os.path.join(UPLOAD_DIR, current_user.id)
            assistant_files = []
            if os.path.exists(user_upload_dir):
                for file in os.listdir(user_upload_dir):
                    if file.startswith(f"{request.assistant_id}_"):
                        assistant_files.append(os.path.join(user_upload_dir, file))
            
            if not assistant_files:
                raise HTTPException(400, "Assistant files not found")
                
            documents = []
            for file_path in assistant_files:
                file_ext = os.path.splitext(file_path)[1].lower()
                try:
                    if file_ext == '.csv':
                        docs = DataLoader.load_from_csv(file_path)
                    elif file_ext == '.json':
                        docs = DataLoader.load_from_json(file_path)
                    documents.extend(docs)
                except Exception:
                    continue
            
            if not documents:
                raise HTTPException(400, "Failed to load assistant documents")
                
            vector_store = vector_store_manager.create_vector_store(documents)
            system_instructions = assistant_engine._build_system_instructions(
                custom_instructions=assistant_db.custom_instructions,
                enable_statistics=assistant_db.enable_statistics,
                enable_alerts=assistant_db.enable_alerts,
                enable_recommendations=assistant_db.enable_recommendations
            )
            
            assistants_store[request.assistant_id] = {
                "assistant_id": request.assistant_id,
                "name": assistant_db.name,
                "vector_store": vector_store,
                "system_instructions": system_instructions,
                "documents_count": len(documents),
                "created_at": assistant_db.created_at
            }
        
        assistant_config = assistants_store[request.assistant_id]
        
        async def wrap_generator():
            full_response = ""
            # Save User Message immediately
            await crud.save_chat_message(current_user.id, request.assistant_id, "user", request.message)
            
            async for chunk in assistant_engine.chat_stream(
                assistant_config=assistant_config,
                user_message=request.message
            ):
                import json
                try:
                    data = json.loads(chunk.strip())
                    if data.get("type") == "content":
                        full_response += data.get("data", "")
                except:
                    pass
                yield chunk
            
            # Save Assistant Message once stream is complete
            if full_response:
                await crud.save_chat_message(current_user.id, request.assistant_id, "assistant", full_response)

        return EventSourceResponse(wrap_generator())
            
    except Exception as e:
        logger.error(f"Error during stream chat: {str(e)}")
        raise HTTPException(500, str(e))


@app.get("/api/assistants/{assistant_id}", response_model=AssistantInfo)
async def get_assistant_info(
    assistant_id: str,
    current_user: UserInDB = Depends(get_current_user)
):
    try:
        # Verify user owns this assistant
        assistant_db = await crud.get_assistant_by_id(assistant_id, current_user.id)
        if not assistant_db:
            raise HTTPException(404, "Assistant not found")
            
        # Lazy load graph data if missing (for legacy assistants)
        graph_data = getattr(assistant_db, 'graph_data', {})
        if not graph_data:
            try:
                # Try to regenerate from source file
                user_upload_dir = os.path.join(UPLOAD_DIR, current_user.id)
                if os.path.exists(user_upload_dir):
                     for file in os.listdir(user_upload_dir):
                        if file.startswith(f"{assistant_id}_"):
                            file_path = os.path.join(user_upload_dir, file)
                            graph_data = DataLoader.generate_graph_insights(file_path, assistant_db.data_source_type)
                            if graph_data:
                                # Optional: Persist back to DB to save effort next time
                                # await crud.update_assistant(assistant_id, {"graph_data": graph_data})
                                pass
                            break
            except Exception as e:
                logger.warning(f"Failed to regenerate graph data: {e}")

        # Fallback to Mock Data if still empty
        if not graph_data:
             graph_data = {
                "bar_chart": {
                    "title": "Value Distribution",
                    "labels": ["Mock A", "Mock B", "Mock C"],
                    "values": [30, 50, 20]
                },
                "donut_chart": {
                    "title": "Category Breakdown",
                    "center_label": "3",
                    "center_text": "Types",
                    "labels": ["Type X", "Type Y", "Type Z"],
                    "values": [30, 50, 20]
                },
                 "line_chart": {
                    "title": "Data Trend",
                    "avg_value": "0.0",
                    "trend_label": "Avg Value",
                    "trend_change": "+0.0%",
                    "data_points": [10, 30, 15, 40, 20, 50, 30, 60]
                }
            }

        return AssistantInfo(
            assistant_id=assistant_db.assistant_id,
            name=assistant_db.name,
            data_source_type=assistant_db.data_source_type,
            custom_instructions=assistant_db.custom_instructions,
            documents_count=assistant_db.documents_count,
            enable_statistics=assistant_db.enable_statistics,
            enable_alerts=assistant_db.enable_alerts,
            enable_recommendations=assistant_db.enable_recommendations,
            attributes=getattr(assistant_db, 'attributes', []),
            sample_questions=getattr(assistant_db, 'sample_questions', []),
            graph_data=graph_data,
            created_at=assistant_db.created_at.isoformat()
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting assistant info: {str(e)}")
        raise HTTPException(500, f"Failed to get assistant info: {str(e)}")


@app.get("/api/assistants")
async def list_assistants(current_user: UserInDB = Depends(get_current_user)):
    try:
        # Get assistants from MongoDB for this user
        assistants = await crud.get_user_assistants(current_user.id)
        
        assistants_list = [
            {
                "assistant_id": asst.assistant_id,
                "name": asst.name,
                "documents_count": asst.documents_count,
                "data_source_type": asst.data_source_type,
                "created_at": asst.created_at.isoformat()
            }
            for asst in assistants
        ]
        
        return {"assistants": assistants_list, "count": len(assistants_list)}
    
    except Exception as e:
        logger.error(f"Error listing assistants: {str(e)}")
        raise HTTPException(500, f"Failed to list assistants: {str(e)}")


@app.delete("/api/assistants/{assistant_id}")
async def delete_assistant(
    assistant_id: str,
    current_user: UserInDB = Depends(get_current_user)
):
    try:
        # Delete from MongoDB
        deleted = await crud.delete_assistant(assistant_id, current_user.id)
        if not deleted:
            raise HTTPException(404, "Assistant not found or access denied")
        
        # Remove from memory store
        if assistant_id in assistants_store:
            del assistants_store[assistant_id]
        
        # Clean up user files
        user_upload_dir = os.path.join(UPLOAD_DIR, current_user.id)
        if os.path.exists(user_upload_dir):
            for file in os.listdir(user_upload_dir):
                if file.startswith(assistant_id):
                    os.remove(os.path.join(user_upload_dir, file))
        
        logger.info(f"Assistant deleted: {assistant_id}")
        
        return {"message": "Assistant deleted successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting assistant: {str(e)}")
        raise HTTPException(500, f"Failed to delete assistant: {str(e)}")


@app.get("/api/assistants/{assistant_id}/chat-history")
async def get_assistant_chat_history(
    assistant_id: str,
    limit: int = 50,
    current_user: UserInDB = Depends(get_current_user)
):
    try:
        # Verify user owns this assistant
        assistant_db = await crud.get_assistant_by_id(assistant_id, current_user.id)
        if not assistant_db:
            raise HTTPException(404, "Assistant not found or access denied")
        
        # Get chat history
        messages = await crud.get_chat_history(current_user.id, assistant_id, limit)
        
        return {
            "assistant_id": assistant_id,
            "messages": messages,
            "total": len(messages)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving chat history: {str(e)}")
        raise HTTPException(500, f"Failed to retrieve chat history: {str(e)}")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
