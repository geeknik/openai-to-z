"""
Caching utilities to minimize repeated API calls and computation.
This helps reduce costs for both OpenAI API usage and computational resources.
"""

import os
import json
import pickle
import hashlib
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union
from functools import wraps

from ..config import CACHE_DIR, ENABLE_CACHE, CACHE_EXPIRE_DAYS

def _get_cache_path(cache_key: str, extension: str = ".pkl") -> Path:
    """Get the path to a cache file."""
    return CACHE_DIR / f"{cache_key}{extension}"

def _hash_args(*args, **kwargs) -> str:
    """Create a hash from function arguments."""
    args_str = str(args) + str(sorted(kwargs.items()))
    return hashlib.md5(args_str.encode()).hexdigest()

def cache_result(
    expire_seconds: Optional[int] = None, 
    use_json: bool = False
) -> Callable:
    """
    Decorator to cache function results to disk.
    
    Args:
        expire_seconds: Cache expiration time in seconds (None for no expiration)
        use_json: Whether to use JSON instead of pickle (more portable but limited types)
    """
    if expire_seconds is None and CACHE_EXPIRE_DAYS > 0:
        expire_seconds = CACHE_EXPIRE_DAYS * 86400  # Convert days to seconds
    
    extension = ".json" if use_json else ".pkl"
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            if not ENABLE_CACHE:
                return func(*args, **kwargs)
                
            # Create cache key from function name and arguments
            cache_key = f"{func.__name__}_{_hash_args(*args, **kwargs)}"
            cache_path = _get_cache_path(cache_key, extension)
            
            # Check if cached result exists and is not expired
            if cache_path.exists():
                # Check expiration if applicable
                if expire_seconds is not None:
                    modification_time = cache_path.stat().st_mtime
                    if time.time() - modification_time > expire_seconds:
                        # Expired, remove cache
                        cache_path.unlink()
                    else:
                        # Load cache
                        try:
                            if use_json:
                                with open(cache_path, 'r') as f:
                                    return json.load(f)
                            else:
                                with open(cache_path, 'rb') as f:
                                    return pickle.load(f)
                        except (json.JSONDecodeError, pickle.PickleError):
                            # Invalid cache, remove and continue
                            cache_path.unlink()
                else:
                    # No expiration, load cache
                    try:
                        if use_json:
                            with open(cache_path, 'r') as f:
                                return json.load(f)
                        else:
                            with open(cache_path, 'rb') as f:
                                return pickle.load(f)
                    except (json.JSONDecodeError, pickle.PickleError):
                        # Invalid cache, remove and continue
                        cache_path.unlink()
            
            # Cache miss or expired, call function
            result = func(*args, **kwargs)
            
            # Save result to cache
            try:
                if use_json:
                    with open(cache_path, 'w') as f:
                        json.dump(result, f)
                else:
                    with open(cache_path, 'wb') as f:
                        pickle.dump(result, f)
            except (TypeError, pickle.PickleError):
                # If caching fails, just return the result
                pass
                
            return result
        return wrapper
    return decorator

def clear_cache(prefix: Optional[str] = None, force: bool = False) -> int:
    """
    Clear cached results.
    
    Args:
        prefix: Optional prefix to selectively clear cache
        force: Whether to bypass confirmation for complete cache clear
        
    Returns:
        Number of cache files removed
    """
    if not CACHE_DIR.exists():
        return 0
        
    if prefix:
        files = list(CACHE_DIR.glob(f"{prefix}*"))
    else:
        files = list(CACHE_DIR.glob("*"))
        
    if not files:
        return 0
        
    # Delete files
    for file_path in files:
        file_path.unlink()
        
    return len(files)

def cache_openai_response(model: str, prompt: str, max_age_days: Optional[int] = None) -> Dict:
    """
    Check for cached OpenAI response before making API call.
    This is specifically designed to reduce OpenAI API costs.
    
    Args:
        model: OpenAI model name
        prompt: Prompt text
        max_age_days: Maximum age of cache in days (None for default)
        
    Returns:
        Cached response or None if not in cache
    """
    if not ENABLE_CACHE:
        return None
        
    # Use model and prompt hash as key
    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
    model_hash = hashlib.md5(model.encode()).hexdigest()
    cache_key = f"openai_{model_hash}_{prompt_hash}"
    cache_path = _get_cache_path(cache_key, ".json")
    
    # Set expiration time
    if max_age_days is None:
        max_age_days = CACHE_EXPIRE_DAYS
    expire_seconds = max_age_days * 86400 if max_age_days > 0 else None
    
    # Check if cached response exists and is not expired
    if cache_path.exists() and (expire_seconds is None or 
                               time.time() - cache_path.stat().st_mtime <= expire_seconds):
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            # Invalid cache, remove
            cache_path.unlink()
    
    return None

def save_openai_response(model: str, prompt: str, response: Dict) -> None:
    """
    Save OpenAI response to cache.
    
    Args:
        model: OpenAI model name
        prompt: Prompt text
        response: Response from OpenAI API
    """
    if not ENABLE_CACHE:
        return
        
    # Use model and prompt hash as key
    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
    model_hash = hashlib.md5(model.encode()).hexdigest()
    cache_key = f"openai_{model_hash}_{prompt_hash}"
    cache_path = _get_cache_path(cache_key, ".json")
    
    try:
        with open(cache_path, 'w') as f:
            json.dump(response, f)
    except (TypeError, json.JSONDecodeError):
        # If saving fails, just continue
        pass 