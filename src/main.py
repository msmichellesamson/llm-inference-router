from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import logging
import time

from .core.router import Router
from .core.models import InferenceRequest, InferenceResponse
from .api.schemas import RouterStats
from .api.health import HealthChecker
from .api.metrics import MetricsCollector
from .api.rate_limiter import RateLimiter
from .api.batch import BatchProcessor, BatchRequest, BatchResponse
from .database.redis_cache import RedisCache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LLM Inference Router", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
redis_cache = RedisCache()
router = Router(cache=redis_cache)
batch_processor = BatchProcessor(router)
health_checker = HealthChecker(router)
metrics = MetricsCollector()
rate_limiter = RateLimiter(redis_cache, requests_per_minute=100)

@app.post("/infer", response_model=InferenceResponse)
async def infer(request: InferenceRequest, _: None = Depends(rate_limiter.check_rate_limit)):
    start_time = time.time()
    try:
        response = await router.route(request)
        metrics.record_request(response.model_used, time.time() - start_time)
        return response
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch", response_model=BatchResponse)
async def batch_infer(request: BatchRequest, _: None = Depends(rate_limiter.check_rate_limit)):
    try:
        return await batch_processor.process_batch(request)
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return await health_checker.check_health()

@app.get("/stats", response_model=RouterStats)
async def stats():
    return await router.get_stats()

@app.get("/metrics")
async def get_metrics():
    return metrics.get_prometheus_metrics()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)