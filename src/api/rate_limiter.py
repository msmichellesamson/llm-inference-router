from typing import Optional, Dict, Any
import time
import logging
from dataclasses import dataclass
from redis import Redis
from src.core.exceptions import RateLimitExceeded

logger = logging.getLogger(__name__)

@dataclass
class RateLimitConfig:
    requests_per_minute: int = 100
    burst_limit: int = 10
    window_size: int = 60

class RateLimiter:
    def __init__(self, redis_client: Redis, config: RateLimitConfig):
        self.redis = redis_client
        self.config = config
        
    async def check_rate_limit(self, client_id: str) -> Dict[str, Any]:
        """Check if request is within rate limits using sliding window"""
        current_time = int(time.time())
        window_start = current_time - self.config.window_size
        
        key = f"rate_limit:{client_id}"
        
        try:
            # Remove expired entries
            self.redis.zremrangebyscore(key, 0, window_start)
            
            # Get current request count
            current_count = self.redis.zcard(key)
            
            if current_count >= self.config.requests_per_minute:
                logger.warning(f"Rate limit exceeded for client {client_id}")
                raise RateLimitExceeded(f"Rate limit of {self.config.requests_per_minute}/min exceeded")
            
            # Add current request
            self.redis.zadd(key, {str(current_time): current_time})
            self.redis.expire(key, self.config.window_size)
            
            remaining = self.config.requests_per_minute - current_count - 1
            reset_time = current_time + self.config.window_size
            
            return {
                "allowed": True,
                "remaining": remaining,
                "reset_time": reset_time,
                "retry_after": None
            }
            
        except Exception as e:
            logger.error(f"Rate limiting error: {e}")
            # Fail open - allow request if Redis is down
            return {
                "allowed": True,
                "remaining": self.config.requests_per_minute,
                "reset_time": current_time + self.config.window_size,
                "retry_after": None
            }

    def get_rate_limit_headers(self, client_id: str) -> Dict[str, str]:
        """Get rate limit headers for response"""
        try:
            result = self.check_rate_limit(client_id)
            return {
                "X-RateLimit-Limit": str(self.config.requests_per_minute),
                "X-RateLimit-Remaining": str(result["remaining"]),
                "X-RateLimit-Reset": str(result["reset_time"])
            }
        except RateLimitExceeded:
            return {
                "X-RateLimit-Limit": str(self.config.requests_per_minute),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(int(time.time()) + self.config.window_size),
                "Retry-After": str(self.config.window_size)
            }
