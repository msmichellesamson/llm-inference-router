from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, List, Optional
import time
import asyncio
from ..core.models import ModelProvider
from ..core.metrics import MetricsCollector
from ..database.redis_cache import RedisCache

router = APIRouter(prefix="/benchmark", tags=["benchmark"])

class BenchmarkRequest(BaseModel):
    prompt: str
    models: List[str]
    iterations: int = 3
    max_tokens: int = 100

class BenchmarkResult(BaseModel):
    model: str
    avg_latency_ms: float
    avg_tokens_per_second: float
    success_rate: float
    cost_per_token: float

class BenchmarkRunner:
    def __init__(self, model_provider: ModelProvider, metrics: MetricsCollector, cache: RedisCache):
        self.model_provider = model_provider
        self.metrics = metrics
        self.cache = cache
    
    async def run_benchmark(self, request: BenchmarkRequest) -> List[BenchmarkResult]:
        results = []
        
        for model in request.models:
            latencies = []
            token_counts = []
            successes = 0
            
            for i in range(request.iterations):
                start_time = time.time()
                try:
                    response = await self.model_provider.generate(
                        model=model,
                        prompt=request.prompt,
                        max_tokens=request.max_tokens
                    )
                    
                    end_time = time.time()
                    latency_ms = (end_time - start_time) * 1000
                    latencies.append(latency_ms)
                    
                    token_count = len(response.split())
                    token_counts.append(token_count)
                    tokens_per_second = token_count / (latency_ms / 1000) if latency_ms > 0 else 0
                    
                    successes += 1
                    
                    # Record metrics
                    self.metrics.record_latency(model, latency_ms)
                    self.metrics.record_tokens_per_second(model, tokens_per_second)
                    
                except Exception as e:
                    print(f"Benchmark failed for {model}: {e}")
                    continue
            
            if latencies:
                avg_latency = sum(latencies) / len(latencies)
                avg_tps = sum(token_counts) / sum(l/1000 for l in latencies) if latencies else 0
                success_rate = successes / request.iterations
                
                # Estimate cost (placeholder - integrate with actual pricing)
                cost_per_token = self._get_model_cost(model)
                
                results.append(BenchmarkResult(
                    model=model,
                    avg_latency_ms=avg_latency,
                    avg_tokens_per_second=avg_tps,
                    success_rate=success_rate,
                    cost_per_token=cost_per_token
                ))
        
        # Cache results
        cache_key = f"benchmark:{hash(request.prompt)}:{':'.join(request.models)}"
        await self.cache.set(cache_key, [r.dict() for r in results], ttl=3600)
        
        return results
    
    def _get_model_cost(self, model: str) -> float:
        # Placeholder cost mapping - integrate with actual pricing APIs
        cost_map = {
            "gpt-3.5-turbo": 0.002,
            "gpt-4": 0.03,
            "claude-2": 0.008,
            "llama-2-7b": 0.0001  # local model
        }
        return cost_map.get(model, 0.001)

@router.post("/run", response_model=List[BenchmarkResult])
async def run_benchmark(
    request: BenchmarkRequest,
    background_tasks: BackgroundTasks,
    model_provider: ModelProvider = None,
    metrics: MetricsCollector = None,
    cache: RedisCache = None
):
    """Run performance benchmark across multiple models"""
    if not model_provider or not metrics or not cache:
        raise HTTPException(status_code=500, detail="Services not initialized")
    
    runner = BenchmarkRunner(model_provider, metrics, cache)
    return await runner.run_benchmark(request)

@router.get("/history/{benchmark_id}")
async def get_benchmark_history(benchmark_id: str, cache: RedisCache = None):
    """Get cached benchmark results"""
    if not cache:
        raise HTTPException(status_code=500, detail="Cache not initialized")
    
    results = await cache.get(benchmark_id)
    if not results:
        raise HTTPException(status_code=404, detail="Benchmark not found")
    
    return results
