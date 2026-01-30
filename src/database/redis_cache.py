import asyncio
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict

import redis.asyncio as redis
import structlog
from pydantic import BaseModel

from src.core.models import LLMResponse, ModelMetadata
from src.core.metrics import MetricsCollector


logger = structlog.get_logger(__name__)


class CacheError(Exception):
    """Base exception for cache operations."""
    pass


class CacheConnectionError(CacheError):
    """Raised when Redis connection fails."""
    pass


class CacheSerializationError(CacheError):
    """Raised when data serialization/deserialization fails."""
    pass


@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int
    misses: int
    evictions: int
    memory_usage: int
    key_count: int
    hit_rate: float


@dataclass
class CacheEntry:
    """Individual cache entry with metadata."""
    data: Dict[str, Any]
    created_at: datetime
    expires_at: datetime
    access_count: int
    model_name: str
    complexity_score: float
    response_time_ms: int


class RedisCache:
    """Production Redis cache for LLM responses and model metadata."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        max_connections: int = 20,
        socket_timeout: int = 5,
        socket_connect_timeout: int = 5,
        default_ttl: int = 3600,  # 1 hour
        response_ttl: int = 1800,  # 30 minutes
        metadata_ttl: int = 7200,  # 2 hours
        max_retries: int = 3,
        metrics_collector: Optional[MetricsCollector] = None
    ) -> None:
        """Initialize Redis cache connection.
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password if auth enabled
            max_connections: Maximum connection pool size
            socket_timeout: Socket timeout in seconds
            socket_connect_timeout: Connection timeout in seconds
            default_ttl: Default TTL for cache entries in seconds
            response_ttl: TTL for LLM response cache in seconds
            metadata_ttl: TTL for model metadata cache in seconds
            max_retries: Maximum retry attempts for operations
            metrics_collector: Metrics collector instance
        """
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.default_ttl = default_ttl
        self.response_ttl = response_ttl
        self.metadata_ttl = metadata_ttl
        self.max_retries = max_retries
        self.metrics = metrics_collector
        
        self.pool = redis.ConnectionPool(
            host=host,
            port=port,
            db=db,
            password=password,
            max_connections=max_connections,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_connect_timeout,
            retry_on_timeout=True,
            health_check_interval=30
        )
        
        self._client: Optional[redis.Redis] = None
        self._connected = False
        
        # Cache prefixes for different data types
        self.RESPONSE_PREFIX = "llm:response:"
        self.METADATA_PREFIX = "llm:metadata:"
        self.STATS_PREFIX = "llm:stats:"
        self.HEALTH_KEY = "llm:health"
        
    async def connect(self) -> None:
        """Establish Redis connection with health check."""
        try:
            self._client = redis.Redis(connection_pool=self.pool)
            
            # Test connection
            await self._client.ping()
            self._connected = True
            
            # Set health indicator
            await self._client.setex(
                self.HEALTH_KEY,
                60,
                json.dumps({
                    "status": "healthy",
                    "timestamp": datetime.utcnow().isoformat(),
                    "version": "1.0.0"
                })
            )
            
            logger.info(
                "Redis connection established",
                host=self.host,
                port=self.port,
                db=self.db
            )
            
            if self.metrics:
                self.metrics.cache_connections_total.inc()
                
        except Exception as e:
            self._connected = False
            logger.error(
                "Failed to connect to Redis",
                error=str(e),
                host=self.host,
                port=self.port
            )
            
            if self.metrics:
                self.metrics.cache_errors_total.labels(operation="connect").inc()
                
            raise CacheConnectionError(f"Redis connection failed: {e}")
    
    async def disconnect(self) -> None:
        """Close Redis connection gracefully."""
        if self._client:
            await self._client.aclose()
            self._connected = False
            logger.info("Redis connection closed")
    
    def _ensure_connected(self) -> None:
        """Ensure Redis connection is active."""
        if not self._connected or not self._client:
            raise CacheConnectionError("Redis not connected")
    
    def _generate_cache_key(self, prompt: str, model_name: str, params: Dict[str, Any]) -> str:
        """Generate deterministic cache key for request."""
        # Create deterministic hash from prompt, model, and parameters
        content = {
            "prompt": prompt,
            "model": model_name,
            "params": sorted(params.items())
        }
        
        content_str = json.dumps(content, sort_keys=True)
        hash_digest = hashlib.sha256(content_str.encode()).hexdigest()
        
        return f"{self.RESPONSE_PREFIX}{model_name}:{hash_digest[:16]}"
    
    def _serialize_entry(self, entry: CacheEntry) -> str:
        """Serialize cache entry to JSON."""
        try:
            data = asdict(entry)
            data["created_at"] = entry.created_at.isoformat()
            data["expires_at"] = entry.expires_at.isoformat()
            return json.dumps(data)
        except Exception as e:
            raise CacheSerializationError(f"Failed to serialize entry: {e}")
    
    def _deserialize_entry(self, data: str) -> CacheEntry:
        """Deserialize JSON to cache entry."""
        try:
            parsed = json.loads(data)
            parsed["created_at"] = datetime.fromisoformat(parsed["created_at"])
            parsed["expires_at"] = datetime.fromisoformat(parsed["expires_at"])
            return CacheEntry(**parsed)
        except Exception as e:
            raise CacheSerializationError(f"Failed to deserialize entry: {e}")
    
    async def _execute_with_retry(self, operation: str, func, *args, **kwargs) -> Any:
        """Execute Redis operation with retry logic."""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_error = e
                logger.warning(
                    "Redis operation failed, retrying",
                    operation=operation,
                    attempt=attempt + 1,
                    error=str(e)
                )
                
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(0.1 * (2 ** attempt))  # Exponential backoff
        
        if self.metrics:
            self.metrics.cache_errors_total.labels(operation=operation).inc()
        
        raise CacheError(f"Operation {operation} failed after {self.max_retries} attempts: {last_error}")
    
    async def get_response(
        self,
        prompt: str,
        model_name: str,
        params: Dict[str, Any]
    ) -> Optional[LLMResponse]:
        """Retrieve cached LLM response."""
        self._ensure_connected()
        
        cache_key = self._generate_cache_key(prompt, model_name, params)
        
        try:
            cached_data = await self._execute_with_retry(
                "get_response",
                self._client.get,
                cache_key
            )
            
            if cached_data:
                entry = self._deserialize_entry(cached_data.decode())
                
                # Check if entry has expired
                if entry.expires_at <= datetime.utcnow():
                    await self._execute_with_retry(
                        "delete_expired",
                        self._client.delete,
                        cache_key
                    )
                    
                    if self.metrics:
                        self.metrics.cache_misses_total.labels(type="response").inc()
                    
                    return None
                
                # Update access count
                entry.access_count += 1
                await self._execute_with_retry(
                    "update_access",
                    self._client.setex,
                    cache_key,
                    self.response_ttl,
                    self._serialize_entry(entry)
                )
                
                if self.metrics:
                    self.metrics.cache_hits_total.labels(type="response").inc()
                    self.metrics.cache_response_time_ms.observe(entry.response_time_ms)
                
                logger.debug(
                    "Cache hit for response",
                    model=model_name,
                    access_count=entry.access_count
                )
                
                return LLMResponse(**entry.data)
            
            if self.metrics:
                self.metrics.cache_misses_total.labels(type="response").inc()
            
            return None
            
        except Exception as e:
            logger.error(
                "Failed to get cached response",
                error=str(e),
                model=model_name
            )
            return None
    
    async def set_response(
        self,
        prompt: str,
        model_name: str,
        params: Dict[str, Any],
        response: LLMResponse,
        complexity_score: float,
        response_time_ms: int
    ) -> bool:
        """Cache LLM response with metadata."""
        self._ensure_connected()
        
        cache_key = self._generate_cache_key(prompt, model_name, params)
        
        try:
            now = datetime.utcnow()
            expires_at = now + timedelta(seconds=self.response_ttl)
            
            entry = CacheEntry(
                data=response.dict(),
                created_at=now,
                expires_at=expires_at,
                access_count=0,
                model_name=model_name,
                complexity_score=complexity_score,
                response_time_ms=response_time_ms
            )
            
            await self._execute_with_retry(
                "set_response",
                self._client.setex,
                cache_key,
                self.response_ttl,
                self._serialize_entry(entry)
            )
            
            if self.metrics:
                self.metrics.cache_sets_total.labels(type="response").inc()
            
            logger.debug(
                "Cached response",
                model=model_name,
                complexity_score=complexity_score,
                response_time_ms=response_time_ms
            )
            
            return True
            
        except Exception as e:
            logger.error(
                "Failed to cache response",
                error=str(e),
                model=model_name
            )
            return False
    
    async def get_model_metadata(self, model_name: str) -> Optional[ModelMetadata]:
        """Retrieve cached model metadata."""
        self._ensure_connected()
        
        cache_key = f"{self.METADATA_PREFIX}{model_name}"
        
        try:
            cached_data = await self._execute_with_retry(
                "get_metadata",
                self._client.get,
                cache_key
            )
            
            if cached_data:
                data = json.loads(cached_data.decode())
                
                if self.metrics:
                    self.metrics.cache_hits_total.labels(type="metadata").inc()
                
                return ModelMetadata(**data)
            
            if self.metrics:
                self.metrics.cache_misses_total.labels(type="metadata").inc()
            
            return None
            
        except Exception as e:
            logger.error(
                "Failed to get model metadata",
                error=str(e),
                model=model_name
            )
            return None
    
    async def set_model_metadata(self, model_name: str, metadata: ModelMetadata) -> bool:
        """Cache model metadata."""
        self._ensure_connected()
        
        cache_key = f"{self.METADATA_PREFIX}{model_name}"
        
        try:
            await self._execute_with_retry(
                "set_metadata",
                self._client.setex,
                cache_key,
                self.metadata_ttl,
                json.dumps(metadata.dict())
            )
            
            if self.metrics:
                self.metrics.cache_sets_total.labels(type="metadata").inc()
            
            logger.debug("Cached model metadata", model=model_name)
            return True
            
        except Exception as e:
            logger.error(
                "Failed to cache model metadata",
                error=str(e),
                model=model_name
            )
            return False
    
    async def invalidate_model(self, model_name: str) -> int:
        """Invalidate all cached data for a specific model."""
        self._ensure_connected()
        
        try:
            # Find all keys for this model
            response_pattern = f"{self.RESPONSE_PREFIX}{model_name}:*"
            metadata_key = f"{self.METADATA_PREFIX}{model_name}"
            
            keys_to_delete = []
            
            # Scan for response keys
            async for key in self._client.scan_iter(match=response_pattern):
                keys_to_delete.append(key)
            
            # Add metadata key
            keys_to_delete.append(metadata_key)
            
            if keys_to_delete:
                deleted_count = await self._execute_with_retry(
                    "invalidate_model",
                    self._client.delete,
                    *keys_to_delete
                )
                
                if self.metrics:
                    self.metrics.cache_invalidations_total.labels(type="model").inc()
                
                logger.info(
                    "Invalidated model cache",
                    model=model_name,
                    deleted_keys=deleted_count
                )
                
                return deleted_count
            
            return 0
            
        except Exception as e:
            logger.error(
                "Failed to invalidate model cache",
                error=str(e),
                model=model_name
            )
            return 0
    
    async def get_cache_stats(self) -> CacheStats:
        """Get comprehensive cache statistics."""
        self._ensure_connected()
        
        try:
            info = await self._client.info("memory")
            keyspace = await self._client.info("keyspace")
            
            # Count keys by prefix
            response_keys = 0
            metadata_keys = 0
            
            async for key in self._client.scan_iter(match=f"{self.RESPONSE_PREFIX}*"):
                response_keys += 1
            
            async for key in self._client.scan_iter(match=f"{self.METADATA_PREFIX}*"):
                metadata_keys += 1
            
            total_keys = response_keys + metadata_keys
            
            # Get hit/miss stats from metrics if available
            hits = 0
            misses = 0
            
            if self.metrics:
                # These would need to be tracked separately in a real implementation
                # as Redis doesn't provide hit/miss stats directly
                pass
            
            hit_rate = hits / (hits + misses) if (hits + misses) > 0 else 0.0
            
            stats = CacheStats(
                hits=hits,
                misses=misses,
                evictions=info.get("evicted_keys", 0),
                memory_usage=info.get("used_memory", 0),
                key_count=total_keys,
                hit_rate=hit_rate
            )
            
            logger.debug(
                "Cache statistics retrieved",
                total_keys=total_keys,
                memory_usage=stats.memory_usage,
                hit_rate=stats.hit_rate
            )
            
            return stats
            
        except Exception as e:
            logger.error("Failed to get cache stats", error=str(e))
            return CacheStats(0, 0, 0, 0, 0, 0.0)
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        health_status = {
            "status": "unknown",
            "connected": self._connected,
            "timestamp": datetime.utcnow().isoformat(),
            "latency_ms": None,
            "memory_usage": None,
            "key_count": None,
            "errors": []
        }
        
        try:
            self._ensure_connected()
            
            # Test latency
            start_time = datetime.utcnow()
            await self._client.ping()
            latency = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            health_status["latency_ms"] = round(latency, 2)
            
            # Get basic stats
            info = await self._client.info("memory")
            keyspace = await self._client.dbsize()
            
            health_status["memory_usage"] = info.get("used_memory", 0)
            health_status["key_count"] = keyspace
            health_status["status"] = "healthy"
            
        except Exception as e:
            health_status["status"] = "unhealthy"
            health_status["errors"].append(str(e))
            logger.error("Cache health check failed", error=str(e))
        
        return health_status
    
    async def cleanup_expired(self) -> int:
        """Clean up expired cache entries manually."""
        self._ensure_connected()
        
        cleaned_count = 0
        
        try:
            # Scan all response keys
            async for key in self._client.scan_iter(match=f"{self.RESPONSE_PREFIX}*"):
                try:
                    data = await self._client.get(key)
                    if data:
                        entry = self._deserialize_entry(data.decode())
                        
                        if entry.expires_at <= datetime.utcnow():
                            await self._client.delete(key)
                            cleaned_count += 1
                            
                except Exception as e:
                    logger.warning(
                        "Failed to check/clean entry",
                        key=key.decode(),
                        error=str(e)
                    )
            
            logger.info(f"Cleaned up {cleaned_count} expired cache entries")
            return cleaned_count
            
        except Exception as e:
            logger.error("Cache cleanup failed", error=str(e))
            return 0


# Singleton instance for application use
_cache_instance: Optional[RedisCache] = None


def get_cache() -> RedisCache:
    """Get singleton cache instance."""
    global _cache_instance
    if _cache_instance is None:
        raise RuntimeError("Cache not initialized. Call init_cache() first.")
    return _cache_instance


def init_cache(
    host: str = "localhost",
    port: int = 6379,
    db: int = 0,
    password: Optional[str] = None,
    metrics_collector: Optional[MetricsCollector] = None,
    **kwargs
) -> RedisCache:
    """Initialize singleton cache instance."""
    global _cache_instance
    _cache_instance = RedisCache(
        host=host,
        port=port,
        db=db,
        password=password,
        metrics_collector=metrics_collector,
        **kwargs
    )
    return _cache_instance