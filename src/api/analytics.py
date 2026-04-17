from typing import Dict, List, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel

from ..core.metrics import MetricsCollector
from ..core.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/analytics", tags=["analytics"])

class ModelUsageStats(BaseModel):
    model_name: str
    request_count: int
    avg_latency_ms: float
    success_rate: float
    total_cost_usd: float
    last_used: datetime

class UsageAnalytics(BaseModel):
    time_window_hours: int
    total_requests: int
    models: List[ModelUsageStats]
    cost_savings_usd: float
    avg_routing_latency_ms: float

@router.get("/usage", response_model=UsageAnalytics)
async def get_usage_analytics(
    hours: int = Query(24, ge=1, le=168, description="Time window in hours"),
    model_filter: Optional[str] = Query(None, description="Filter by model name")
):
    """Get model usage analytics and routing performance."""
    try:
        metrics_collector = MetricsCollector()
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        # Get raw metrics
        raw_metrics = await metrics_collector.get_metrics_range(
            start_time, end_time, model_filter
        )
        
        # Calculate model stats
        model_stats = []
        total_requests = 0
        total_cost_savings = 0.0
        routing_latencies = []
        
        for model_name, metrics in raw_metrics.items():
            if metrics['request_count'] > 0:
                stats = ModelUsageStats(
                    model_name=model_name,
                    request_count=metrics['request_count'],
                    avg_latency_ms=metrics['total_latency'] / metrics['request_count'],
                    success_rate=metrics['success_count'] / metrics['request_count'],
                    total_cost_usd=metrics['total_cost'],
                    last_used=metrics['last_used']
                )
                model_stats.append(stats)
                total_requests += stats.request_count
                total_cost_savings += metrics.get('cost_savings', 0.0)
                routing_latencies.extend(metrics.get('routing_latencies', []))
        
        avg_routing_latency = (
            sum(routing_latencies) / len(routing_latencies) 
            if routing_latencies else 0.0
        )
        
        return UsageAnalytics(
            time_window_hours=hours,
            total_requests=total_requests,
            models=sorted(model_stats, key=lambda x: x.request_count, reverse=True),
            cost_savings_usd=round(total_cost_savings, 2),
            avg_routing_latency_ms=round(avg_routing_latency, 2)
        )
        
    except Exception as e:
        logger.error(f"Failed to get usage analytics: {e}")
        raise HTTPException(status_code=500, detail="Analytics unavailable")

@router.get("/models/{model_name}/performance")
async def get_model_performance(
    model_name: str,
    hours: int = Query(24, ge=1, le=168)
):
    """Get detailed performance metrics for a specific model."""
    try:
        metrics_collector = MetricsCollector()
        performance = await metrics_collector.get_model_performance(
            model_name, hours
        )
        
        if not performance:
            raise HTTPException(
                status_code=404, 
                detail=f"No performance data found for model: {model_name}"
            )
            
        return performance
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model performance for {model_name}: {e}")
        raise HTTPException(status_code=500, detail="Performance data unavailable")