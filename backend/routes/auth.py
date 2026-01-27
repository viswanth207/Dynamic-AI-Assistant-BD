from fastapi import APIRouter, HTTPException, status, Response
from fastapi.responses import JSONResponse
from backend.database.models import UserCreate, UserLogin, UserResponse, Token
from backend.database import crud
from backend.auth.utils import verify_password, get_password_hash, create_access_token
from backend.auth.dependencies import get_current_user, get_current_user_optional
from fastapi import Depends
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


@router.post("/signup", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def signup(user: UserCreate):
    """
    Register a new user
    """
    try:
        # Truncate password to 72 bytes (bcrypt limitation)
        password = user.password
        if len(password.encode('utf-8')) > 72:
            password = password.encode('utf-8')[:72].decode('utf-8', errors='ignore')
        
        # Check if user already exists
        existing_user = await crud.get_user_by_email(user.email)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Hash password and create user
        password_hash = get_password_hash(password)
        db_user = await crud.create_user(user, password_hash)
        
        logger.info(f"New user registered: {user.email}")
        
        return UserResponse(
            id=db_user.id,
            email=db_user.email,
            created_at=db_user.created_at
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Signup error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user"
        )


@router.post("/login")
async def login(user_credentials: UserLogin, response: Response):
    """
    Authenticate user and set HTTP-only cookie
    """
    try:
        # Truncate password to 72 bytes (bcrypt limitation)
        password = user_credentials.password
        if len(password.encode('utf-8')) > 72:
            password = password.encode('utf-8')[:72].decode('utf-8', errors='ignore')
        
        # Get user from database
        user = await crud.get_user_by_email(user_credentials.email)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password"
            )
        
        # Verify password
        if not verify_password(password, user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password"
            )
        
        # Create access token
        access_token = create_access_token(data={"sub": user.id, "email": user.email})
        
        # Set HTTP-only cookie
        response.set_cookie(
            key="access_token",
            value=access_token,
            httponly=True,
            max_age=30 * 24 * 60 * 60,  # 30 days
            secure=True,  # Required for SameSite=None
            samesite="none"  # Required for cross-origin requests
        )
        
        logger.info(f"User logged in: {user.email}")
        
        return {
            "message": "Login successful",
            "user": {
                "id": user.id,
                "email": user.email
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.post("/logout")
async def logout(response: Response):
    """
    Logout user by clearing the cookie
    """
    response.delete_cookie(key="access_token")
    return {"message": "Logged out successfully"}


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user = Depends(get_current_user)):
    """
    Get current authenticated user info
    """
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        created_at=current_user.created_at
    )


@router.get("/check")
async def check_auth(current_user = Depends(get_current_user_optional)):
    """
    Check if user is authenticated (doesn't throw error)
    """
    if current_user:
        return {
            "authenticated": True,
            "user": {
                "id": current_user.id,
                "email": current_user.email
            }
        }
    return {"authenticated": False}
