from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging

from .core.router import InferenceRouter
from .api.health import router as health_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    logger.info("Starting LLM Inference Router")
    yield
    logger.info("Shutting down LLM Inference Router")


app = FastAPI(
    title="LLM Inference Router",
    description="Multi-model LLM router with cost and latency optimization",
    version="1.0.0",
    lifespan=lifespan
)

# Include routers
app.include_router(health_router, prefix="/api/v1", tags=["health"])

# Global router instance
router = InferenceRouter()


@app.post("/api/v1/inference")
async def inference(request: dict):
    """Main inference endpoint."""
    try:
        result = await router.route_request(request)
        return result
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        raise


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
