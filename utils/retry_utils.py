import asyncio
from functools import wraps
from typing import Type, Callable, Any, Optional
import time
from services.logger import logger

class RetryError(Exception):
    """Custom error for retry failures"""
    pass

def async_retry(
    retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
    fallback: Optional[Callable] = None
):
    """
    Retry decorator for async functions with exponential backoff
    
    Args:
        retries: Maximum number of retries
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch
        fallback: Optional fallback function to call if all retries fail
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay

            for attempt in range(retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == retries:
                        break
                    
                    logger.warning(
                        f"Attempt {attempt + 1}/{retries} failed for {func.__name__}: {str(e)}. "
                        f"Retrying in {current_delay:.1f}s..."
                    )
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff

            if fallback:
                try:
                    logger.info(f"All retries failed for {func.__name__}, using fallback")
                    return await fallback(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Fallback also failed for {func.__name__}: {str(e)}")
                    raise RetryError(f"Both main function and fallback failed: {str(e)}")
            
            raise RetryError(f"All {retries} retries failed: {str(last_exception)}")
        
        return wrapper
    return decorator

async def simple_fallback(*args, **kwargs):
    """Simple fallback that returns a default error message"""
    return "I apologize, but I'm having trouble processing your request right now. Please try again in a moment."

class CircuitBreaker:
    """Circuit breaker pattern implementation"""
    def __init__(self, failure_threshold: int = 5, reset_timeout: float = 60):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, or half-open

    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker pattern"""
        current_time = time.time()

        # Check if circuit should be reset
        if self.state == "open" and current_time - self.last_failure_time >= self.reset_timeout:
            self.state = "half-open"
            logger.info("Circuit breaker state changed to half-open")

        if self.state == "open":
            raise RetryError("Circuit breaker is open")

        try:
            result = await func(*args, **kwargs)
            if self.state == "half-open":
                self.state = "closed"
                self.failures = 0
                logger.info("Circuit breaker state changed to closed")
            return result

        except Exception as e:
            self.failures += 1
            self.last_failure_time = current_time

            if self.failures >= self.failure_threshold:
                self.state = "open"
                logger.warning(f"Circuit breaker opened after {self.failures} failures")

            raise e