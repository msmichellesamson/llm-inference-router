"""
Metrics collection and monitoring for the LLM inference router.

Tracks cost, latency, and model performance metrics with Prometheus integration.
"""

import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import json
from collections import defaultdict, deque

import structlog
from prometheus_client import (
    Counter, 
    Histogram, 
    Gauge, 
    Summary,
    generate_latest,
    CONTENT_TYPE_LATEST
)
import redis.asyncio as redis
from fastapi import HTTPException


logger = structlog.get_logger(__name__)


class ModelType(Enum):
    LOCAL = "local"
    CLOUD = "cloud"


class MetricType(Enum):
    LATENCY = "latency"
    COST = "cost"
    ERROR = "error"
    THROUGHPUT = "throughput"


class MetricsException(Exception):
    """Base exception for metrics operations."""
    pass


class RedisConnectionError(MetricsException):
    """Redis connection failed."""
    pass


@dataclass
class ModelMetrics:
    """Individual model performance metrics."""
    model_name: str
    model_type: ModelType
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency: float = 0.0
    total_cost: float = 0.0
    avg_latency: float = 0.0
    avg_cost_per_request: float = 0.0
    last_used: Optional[datetime] = None
    recent_latencies: deque = field(default_factory=lambda: deque(maxlen=100))


@dataclass
class RouteDecision:
    """Decision record for routing analysis."""
    timestamp: datetime
    query_complexity: float
    selected_model: str
    model_type: ModelType
    predicted_latency: float
    predicted_cost: float
    actual_latency: Optional[float] = None
    actual_cost: Optional[float] = None
    success: bool = True


class MetricsCollector:
    """Production-grade metrics collection with Redis persistence and Prometheus export."""
    
    def __init__(
        self, 
        redis_url: str = "redis://localhost:6379",
        retention_days: int = 30,
        enable_prometheus: bool = True
    ):
        self.redis_url = redis_url
        self.retention_days = retention_days
        self.enable_prometheus = enable_prometheus
        
        # In-memory metrics cache
        self._model_metrics: Dict[str, ModelMetrics] = {}
        self._route_decisions: deque = deque(maxlen=10000)
        
        # Redis connection
        self._redis: Optional[redis.Redis] = None
        
        # Prometheus metrics
        if enable_prometheus:
            self._setup_prometheus_metrics()
    
    def _setup_prometheus_metrics(self) -> None:
        """Initialize Prometheus metrics."""
        self.request_counter = Counter(
            'llm_requests_total',
            'Total number of LLM requests',
            ['model_name', 'model_type', 'status']
        )
        
        self.latency_histogram = Histogram(
            'llm_request_duration_seconds',
            'Request latency in seconds',
            ['model_name', 'model_type'],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        )
        
        self.cost_histogram = Histogram(
            'llm_request_cost_usd',
            'Request cost in USD',
            ['model_name', 'model_type'],
            buckets=[0.0001, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
        )
        
        self.model_availability = Gauge(
            'llm_model_availability',
            'Model availability (1=available, 0=unavailable)',
            ['model_name', 'model_type']
        )
        
        self.routing_accuracy = Gauge(
            'llm_routing_accuracy',
            'Routing decision accuracy over time'
        )
        
        self.queue_size = Gauge(
            'llm_request_queue_size',
            'Current request queue size',
            ['model_name']
        )
    
    async def initialize(self) -> None:
        """Initialize Redis connection and load existing metrics."""
        try:
            self._redis = redis.from_url(self.redis_url, decode_responses=True)
            await self._redis.ping()
            logger.info("Connected to Redis", redis_url=self.redis_url)
            
            # Load existing metrics from Redis
            await self._load_metrics_from_redis()
            
        except Exception as e:
            logger.error("Failed to connect to Redis", error=str(e))
            raise RedisConnectionError(f"Redis connection failed: {e}")
    
    async def _load_metrics_from_redis(self) -> None:
        """Load historical metrics from Redis."""
        try:
            # Load model metrics
            model_keys = await self._redis.keys("metrics:model:*")
            for key in model_keys:
                model_data = await self._redis.hgetall(key)
                if model_data:
                    model_name = key.split(":")[-1]
                    self._model_metrics[model_name] = self._deserialize_model_metrics(model_data)
            
            # Load recent routing decisions
            decisions_data = await self._redis.lrange("metrics:routing_decisions", 0, 1000)
            for decision_json in decisions_data:
                decision = RouteDecision(**json.loads(decision_json))
                self._route_decisions.append(decision)
            
            logger.info(
                "Loaded metrics from Redis",
                models=len(self._model_metrics),
                decisions=len(self._route_decisions)
            )
        
        except Exception as e:
            logger.warning("Failed to load metrics from Redis", error=str(e))
    
    def _deserialize_model_metrics(self, data: Dict[str, str]) -> ModelMetrics:
        """Convert Redis hash data to ModelMetrics object."""
        return ModelMetrics(
            model_name=data["model_name"],
            model_type=ModelType(data["model_type"]),
            total_requests=int(data.get("total_requests", 0)),
            successful_requests=int(data.get("successful_requests", 0)),
            failed_requests=int(data.get("failed_requests", 0)),
            total_latency=float(data.get("total_latency", 0.0)),
            total_cost=float(data.get("total_cost", 0.0)),
            avg_latency=float(data.get("avg_latency", 0.0)),
            avg_cost_per_request=float(data.get("avg_cost_per_request", 0.0)),
            last_used=datetime.fromisoformat(data["last_used"]) if data.get("last_used") else None
        )
    
    async def record_request_start(
        self, 
        model_name: str, 
        model_type: ModelType,
        query_complexity: float,
        predicted_latency: float,
        predicted_cost: float
    ) -> str:
        """Record the start of a request and return tracking ID."""
        request_id = f"{model_name}_{int(time.time() * 1000000)}"
        
        # Record routing decision
        decision = RouteDecision(
            timestamp=datetime.utcnow(),
            query_complexity=query_complexity,
            selected_model=model_name,
            model_type=model_type,
            predicted_latency=predicted_latency,
            predicted_cost=predicted_cost
        )
        
        self._route_decisions.append(decision)
        
        # Update queue metrics
        if self.enable_prometheus:
            self.queue_size.labels(model_name=model_name).inc()
        
        logger.debug(
            "Request started",
            request_id=request_id,
            model_name=model_name,
            complexity=query_complexity
        )
        
        return request_id
    
    async def record_request_completion(
        self,
        model_name: str,
        model_type: ModelType,
        actual_latency: float,
        actual_cost: float,
        success: bool = True,
        error_type: Optional[str] = None
    ) -> None:
        """Record completion of a request with actual metrics."""
        try:
            # Update model metrics
            if model_name not in self._model_metrics:
                self._model_metrics[model_name] = ModelMetrics(
                    model_name=model_name,
                    model_type=model_type
                )
            
            metrics = self._model_metrics[model_name]
            metrics.total_requests += 1
            metrics.last_used = datetime.utcnow()
            
            if success:
                metrics.successful_requests += 1
                metrics.total_latency += actual_latency
                metrics.total_cost += actual_cost
                metrics.recent_latencies.append(actual_latency)
                
                # Update averages
                metrics.avg_latency = metrics.total_latency / metrics.successful_requests
                metrics.avg_cost_per_request = metrics.total_cost / metrics.successful_requests
            else:
                metrics.failed_requests += 1
            
            # Update Prometheus metrics
            if self.enable_prometheus:
                status = "success" if success else "error"
                self.request_counter.labels(
                    model_name=model_name,
                    model_type=model_type.value,
                    status=status
                ).inc()
                
                if success:
                    self.latency_histogram.labels(
                        model_name=model_name,
                        model_type=model_type.value
                    ).observe(actual_latency)
                    
                    self.cost_histogram.labels(
                        model_name=model_name,
                        model_type=model_type.value
                    ).observe(actual_cost)
                
                self.queue_size.labels(model_name=model_name).dec()
            
            # Update routing decision with actual values
            if self._route_decisions:
                latest_decision = self._route_decisions[-1]
                if (latest_decision.selected_model == model_name and 
                    latest_decision.actual_latency is None):
                    latest_decision.actual_latency = actual_latency
                    latest_decision.actual_cost = actual_cost
                    latest_decision.success = success
            
            # Persist to Redis
            await self._persist_metrics()
            
            logger.info(
                "Request completed",
                model_name=model_name,
                latency=actual_latency,
                cost=actual_cost,
                success=success,
                error_type=error_type
            )
        
        except Exception as e:
            logger.error("Failed to record request completion", error=str(e))
            raise MetricsException(f"Metrics recording failed: {e}")
    
    async def _persist_metrics(self) -> None:
        """Persist current metrics to Redis."""
        if not self._redis:
            return
        
        try:
            # Use pipeline for better performance
            pipe = self._redis.pipeline()
            
            # Store model metrics
            for model_name, metrics in self._model_metrics.items():
                key = f"metrics:model:{model_name}"
                data = {
                    "model_name": metrics.model_name,
                    "model_type": metrics.model_type.value,
                    "total_requests": metrics.total_requests,
                    "successful_requests": metrics.successful_requests,
                    "failed_requests": metrics.failed_requests,
                    "total_latency": metrics.total_latency,
                    "total_cost": metrics.total_cost,
                    "avg_latency": metrics.avg_latency,
                    "avg_cost_per_request": metrics.avg_cost_per_request,
                    "last_used": metrics.last_used.isoformat() if metrics.last_used else ""
                }
                pipe.hset(key, mapping=data)
                pipe.expire(key, timedelta(days=self.retention_days).total_seconds())
            
            # Store recent routing decisions (keep only last 1000)
            if self._route_decisions:
                pipe.delete("metrics:routing_decisions")
                for decision in list(self._route_decisions)[-1000:]:
                    decision_data = {
                        "timestamp": decision.timestamp.isoformat(),
                        "query_complexity": decision.query_complexity,
                        "selected_model": decision.selected_model,
                        "model_type": decision.model_type.value,
                        "predicted_latency": decision.predicted_latency,
                        "predicted_cost": decision.predicted_cost,
                        "actual_latency": decision.actual_latency,
                        "actual_cost": decision.actual_cost,
                        "success": decision.success
                    }
                    pipe.lpush("metrics:routing_decisions", json.dumps(decision_data))
                
                pipe.expire("metrics:routing_decisions", timedelta(days=self.retention_days).total_seconds())
            
            await pipe.execute()
            
        except Exception as e:
            logger.error("Failed to persist metrics to Redis", error=str(e))
    
    def get_model_metrics(self, model_name: str) -> Optional[ModelMetrics]:
        """Get metrics for a specific model."""
        return self._model_metrics.get(model_name)
    
    def get_all_model_metrics(self) -> Dict[str, ModelMetrics]:
        """Get metrics for all models."""
        return self._model_metrics.copy()
    
    def calculate_routing_accuracy(self, time_window_minutes: int = 60) -> float:
        """Calculate routing accuracy over the specified time window."""
        cutoff_time = datetime.utcnow() - timedelta(minutes=time_window_minutes)
        
        recent_decisions = [
            d for d in self._route_decisions 
            if d.timestamp >= cutoff_time and d.actual_latency is not None
        ]
        
        if not recent_decisions:
            return 0.0
        
        accurate_predictions = 0
        total_decisions = len(recent_decisions)
        
        for decision in recent_decisions:
            # Consider prediction accurate if actual latency is within 50% of predicted
            if decision.actual_latency and decision.predicted_latency > 0:
                error_ratio = abs(decision.actual_latency - decision.predicted_latency) / decision.predicted_latency
                if error_ratio <= 0.5:
                    accurate_predictions += 1
        
        accuracy = accurate_predictions / total_decisions
        
        # Update Prometheus metric
        if self.enable_prometheus:
            self.routing_accuracy.set(accuracy)
        
        return accuracy
    
    async def get_cost_breakdown(self, time_window_hours: int = 24) -> Dict[str, float]:
        """Get cost breakdown by model over the specified time window."""
        cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)
        
        cost_breakdown = defaultdict(float)
        
        for decision in self._route_decisions:
            if (decision.timestamp >= cutoff_time and 
                decision.actual_cost is not None and 
                decision.success):
                cost_breakdown[decision.selected_model] += decision.actual_cost
        
        return dict(cost_breakdown)
    
    async def update_model_availability(self, model_name: str, model_type: ModelType, available: bool) -> None:
        """Update model availability status."""
        if self.enable_prometheus:
            self.model_availability.labels(
                model_name=model_name,
                model_type=model_type.value
            ).set(1 if available else 0)
        
        logger.info(
            "Model availability updated",
            model_name=model_name,
            model_type=model_type.value,
            available=available
        )
    
    def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        if not self.enable_prometheus:
            raise MetricsException("Prometheus metrics not enabled")
        
        return generate_latest().decode('utf-8')
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        total_requests = sum(m.total_requests for m in self._model_metrics.values())
        total_successful = sum(m.successful_requests for m in self._model_metrics.values())
        total_cost = sum(m.total_cost for m in self._model_metrics.values())
        
        success_rate = total_successful / total_requests if total_requests > 0 else 0.0
        routing_accuracy = self.calculate_routing_accuracy()
        cost_breakdown = await self.get_cost_breakdown()
        
        return {
            "total_requests": total_requests,
            "success_rate": success_rate,
            "total_cost_usd": round(total_cost, 4),
            "routing_accuracy": round(routing_accuracy, 3),
            "cost_breakdown": cost_breakdown,
            "model_count": len(self._model_metrics),
            "avg_cost_per_request": round(total_cost / total_successful, 6) if total_successful > 0 else 0.0
        }
    
    async def cleanup_old_metrics(self) -> None:
        """Clean up metrics older than retention period."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=self.retention_days)
            
            # Clean up routing decisions
            self._route_decisions = deque(
                [d for d in self._route_decisions if d.timestamp >= cutoff_time],
                maxlen=10000
            )
            
            # Clean up Redis keys
            if self._redis:
                # This is handled by Redis TTL, but we can force cleanup if needed
                await self._redis.eval(
                    """
                    local keys = redis.call('KEYS', ARGV[1])
                    local deleted = 0
                    for i=1,#keys do
                        local ttl = redis.call('TTL', keys[i])
                        if ttl == -1 then
                            redis.call('EXPIRE', keys[i], ARGV[2])
                        end
                    end
                    return deleted
                    """,
                    0,
                    "metrics:*",
                    int(timedelta(days=self.retention_days).total_seconds())
                )
            
            logger.info("Cleaned up old metrics", cutoff_time=cutoff_time.isoformat())
            
        except Exception as e:
            logger.error("Failed to cleanup old metrics", error=str(e))
    
    async def close(self) -> None:
        """Close Redis connection and cleanup resources."""
        try:
            if self._redis:
                await self._redis.close()
                logger.info("Redis connection closed")
        except Exception as e:
            logger.error("Error closing Redis connection", error=str(e))