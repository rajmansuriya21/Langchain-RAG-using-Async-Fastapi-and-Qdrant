import aiosqlite
import json
import time
from functools import wraps
from services.logger import logger
import os
import asyncio
from typing import Any, Optional

class SQLiteCache:
    def __init__(self, db_path: str = "cache.db"):
        self.db_path = db_path
        self._initialize_task = None

    async def initialize(self):
        """Initialize the SQLite cache database"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS cache (
                        key TEXT PRIMARY KEY,
                        value TEXT,
                        expiry FLOAT
                    )
                """)
                await db.commit()
            logger.info("SQLite cache initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize SQLite cache: {e}")
            raise

    async def get(self, key: str) -> Optional[Any]:
        """Get a value from cache"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                current_time = time.time()
                cursor = await db.execute(
                    "SELECT value FROM cache WHERE key = ? AND (expiry IS NULL OR expiry > ?)",
                    (key, current_time)
                )
                result = await cursor.fetchone()
                if result:
                    return json.loads(result[0])
                return None
        except Exception as e:
            logger.warning(f"Cache get operation failed: {e}")
            return None

    async def set(self, key: str, value: Any, expire: Optional[int] = None):
        """Set a value in cache with optional expiration"""
        try:
            expiry = time.time() + expire if expire else None
            value_json = json.dumps(value)
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    "INSERT OR REPLACE INTO cache (key, value, expiry) VALUES (?, ?, ?)",
                    (key, value_json, expiry)
                )
                await db.commit()
        except Exception as e:
            logger.warning(f"Cache set operation failed: {e}")

    async def cleanup(self):
        """Remove expired cache entries"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                current_time = time.time()
                await db.execute("DELETE FROM cache WHERE expiry < ?", (current_time,))
                await db.commit()
        except Exception as e:
            logger.warning(f"Cache cleanup failed: {e}")

cache = SQLiteCache()

def cached_embedding():
    """Cache decorator for embeddings with 1 hour expiration"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache_key = f"embedding:{args[1]}"  # args[1] should be the text to embed
            try:
                cached_result = await cache.get(cache_key)
                if cached_result:
                    logger.info("Cache hit for embedding")
                    return cached_result
                
                result = await func(*args, **kwargs)
                await cache.set(cache_key, result, expire=3600)  # 1 hour cache
                return result
            except Exception as e:
                logger.warning(f"Cache operation failed, falling back to direct computation: {e}")
                return await func(*args, **kwargs)
        return wrapper
    return decorator

def cached_retrieval():
    """Cache decorator for document retrieval with 5 minute expiration"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            query = kwargs.get('refined_query', '')
            cache_key = f"retrieval:{query}"
            try:
                cached_result = await cache.get(cache_key)
                if cached_result:
                    logger.info("Cache hit for document retrieval")
                    return cached_result
                
                result = await func(*args, **kwargs)
                await cache.set(cache_key, result, expire=300)  # 5 minutes cache
                return result
            except Exception as e:
                logger.warning(f"Cache operation failed, falling back to direct retrieval: {e}")
                return await func(*args, **kwargs)
        return wrapper
    return decorator

async def initialize_cache():
    """Initialize the cache system"""
    await cache.initialize()
    asyncio.create_task(periodic_cleanup())

async def periodic_cleanup():
    """Periodically clean up expired cache entries"""
    while True:
        try:
            await cache.cleanup()
        except Exception as e:
            logger.error(f"Cache cleanup error: {e}")
        await asyncio.sleep(300)  # Run cleanup every 5 minutes