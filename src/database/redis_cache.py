import redis
import redis.connection
import logging
from typing import Optional, Any, Dict
from contextlib import contextmanager
import json
import time

logger = logging.getLogger(__name__)

class RedisCache:
    def __init__(self, host: str = 'redis', port: int = 6379, db: int = 0, max_connections: int = 10):
        self.pool = redis.ConnectionPool(
            host=host,
            port=port,
            db=db,
            max_connections=max_connections,
            retry_on_timeout=True,
            socket_keepalive=True,
            socket_keepalive_options={},
            health_check_interval=30
        )
        self.redis = redis.Redis(connection_pool=self.pool)
        self._circuit_breaker_failures = 0
        self._circuit_breaker_threshold = 5
        self._circuit_breaker_timeout = 30
        self._circuit_breaker_last_failure = 0
    
    @contextmanager
    def _handle_redis_errors(self):
        try:
            yield
        except redis.ConnectionError as e:
            self._circuit_breaker_failures += 1
            self._circuit_breaker_last_failure = time.time()
            logger.error(f"Redis connection error: {e}")
            raise
        except redis.TimeoutError as e:
            logger.warning(f"Redis timeout: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected redis error: {e}")
            raise
    
    def _is_circuit_open(self) -> bool:
        if self._circuit_breaker_failures >= self._circuit_breaker_threshold:
            if time.time() - self._circuit_breaker_last_failure < self._circuit_breaker_timeout:
                return True
            else:
                # Reset circuit breaker after timeout
                self._circuit_breaker_failures = 0
        return False
    
    def get(self, key: str) -> Optional[Any]:
        if self._is_circuit_open():
            logger.warning("Redis circuit breaker is open, skipping cache")
            return None
            
        with self._handle_redis_errors():
            value = self.redis.get(key)
            if value:
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON in cache key {key}")
                    return None
            return None
    
    def set(self, key: str, value: Any, ttl: int = 300) -> bool:
        if self._is_circuit_open():
            logger.warning("Redis circuit breaker is open, skipping cache")
            return False
            
        with self._handle_redis_errors():
            serialized = json.dumps(value, default=str)
            result = self.redis.setex(key, ttl, serialized)
            return bool(result)
    
    def delete(self, key: str) -> bool:
        if self._is_circuit_open():
            return False
            
        with self._handle_redis_errors():
            return bool(self.redis.delete(key))
    
    def health_check(self) -> Dict[str, Any]:
        try:
            start_time = time.time()
            self.redis.ping()
            latency = (time.time() - start_time) * 1000
            
            info = self.redis.info()
            return {
                "status": "healthy",
                "latency_ms": round(latency, 2),
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_human": info.get("used_memory_human", "unknown"),
                "circuit_breaker_failures": self._circuit_breaker_failures
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "circuit_breaker_failures": self._circuit_breaker_failures
            }
    
    def close(self):
        """Close connection pool"""
        self.pool.disconnect()