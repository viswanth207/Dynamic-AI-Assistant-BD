from fastapi import Depends, HTTPException, status, Cookie
from typing import Optional
from backend.auth.utils import decode_access_token
from backend.database import crud
from backend.database.models import UserInDB
import logging

logger = logging.getLogger(__name__)


async def get_current_user(access_token: Optional[str] = Cookie(None)) -> UserInDB:
    """
    Dependency to get the current authenticated user from cookie.
    Raises HTTPException if token is invalid or user not found.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    if not access_token:
        raise credentials_exception
    
    # Decode token
    payload = decode_access_token(access_token)
    if payload is None:
        raise credentials_exception
    
    user_id: str = payload.get("sub")
    if user_id is None:
        raise credentials_exception
    
    # Get user from database
    user = await crud.get_user_by_id(user_id)
    if user is None:
        raise credentials_exception
    
    return user


async def get_current_user_optional(access_token: Optional[str] = Cookie(None)) -> Optional[UserInDB]:
    """
    Optional dependency to get current user.
    Returns None if not authenticated instead of raising exception.
    """
    if not access_token:
        return None
    
    payload = decode_access_token(access_token)
    if payload is None:
        return None
    
    user_id: str = payload.get("sub")
    if user_id is None:
        return None
    
    user = await crud.get_user_by_id(user_id)
    return user
