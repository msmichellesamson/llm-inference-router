from fastapi import FastAPI
from src.api.tracing import TracingMiddleware
from src.api.health import router as health_router
from src.api.preprocessing import router as preprocess_router
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = FastAPI(
    title="LLM Inference Router",
    description="Multi-model router with intelligent query routing",
    version="1.0.0"
)

# Add tracing middleware
app.add_middleware(TracingMiddleware)

# Include routers
app.include_router(health_router, prefix="/health", tags=["health"])
app.include_router(preprocess_router, prefix="/api/v1", tags=["routing"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
