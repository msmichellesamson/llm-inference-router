from typing import Optional, Dict, Any
import hashlib
import json
import time
from functools import wraps
from ..database.redis_cache import RedisCache
from ..core.metrics import MetricsCollector


class QueryCacheMiddleware:
    """Caches LLM responses based on query hash with configurable TTL."""
    
    def __init__(self, redis_cache: RedisCache, metrics: MetricsCollector):
        self.redis = redis_cache
        self.metrics = metrics
        self.default_ttl = 3600  # 1 hour
        self.cache_prefix = "llm_query:"
    
    def _generate_cache_key(self, query: str, model: str, temperature: float) -> str:
        """Generate deterministic cache key from query parameters."""
        cache_data = {
            "query": query.strip().lower(),
            "model": model,
            "temperature": round(temperature, 2)
        }
        hash_input = json.dumps(cache_data, sort_keys=True)
        query_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]
        return f"{self.cache_prefix}{query_hash}"
    
    def get_cached_response(self, query: str, model: str, temperature: float) -> Optional[Dict[str, Any]]:
        """Retrieve cached response if available."""
        cache_key = self._generate_cache_key(query, model, temperature)
        
        try:
            cached_data = self.redis.get(cache_key)
            if cached_data:
                self.metrics.increment("cache_hit")
                return {
                    "response": cached_data["response"],
                    "model_used": cached_data["model_used"],
                    "cached_at": cached_data["timestamp"],
                    "from_cache": True
                }
            else:
                self.metrics.increment("cache_miss")
                return None
        except Exception as e:
            self.metrics.increment("cache_error")
            return None
    
    def cache_response(self, query: str, model: str, temperature: float, 
                      response: str, model_used: str, ttl: Optional[int] = None) -> bool:
        """Cache the LLM response with TTL."""
        cache_key = self._generate_cache_key(query, model, temperature)
        ttl = ttl or self.default_ttl
        
        cache_data = {
            "response": response,
            "model_used": model_used,
            "timestamp": int(time.time())
        }
        
        try:
            self.redis.setex(cache_key, ttl, cache_data)
            self.metrics.increment("cache_store")
            return True
        except Exception as e:
            self.metrics.increment("cache_store_error")
            return False
    
    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching pattern."""
        try:
            keys = self.redis.scan_iter(match=f"{self.cache_prefix}{pattern}*")
            deleted = 0
            for key in keys:
                if self.redis.delete(key):
                    deleted += 1
            self.metrics.increment("cache_invalidation", deleted)
            return deleted
        except Exception:
            return 0
