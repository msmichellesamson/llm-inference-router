from fastapi import FastAPI, HTTPException
from src.api.health import router as health_router
from src.api.metrics import router as metrics_router
from src.core.router import LLMRouter
from src.core.models import QueryRequest, QueryResponse
from src.api.metrics import metrics
import uuid
import logging

app = FastAPI(title="LLM Inference Router", version="1.0.0")

# Include routers
app.include_router(health_router, prefix="/health")
app.include_router(metrics_router, prefix="/monitoring")

# Initialize LLM router
llm_router = LLMRouter()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.post("/query", response_model=QueryResponse)
async def route_query(request: QueryRequest):
    """Route LLM query to optimal model based on complexity analysis"""
    request_id = str(uuid.uuid4())
    
    try:
        # Analyze complexity and route request
        routing_decision = await llm_router.route_request(request)
        model = routing_decision.selected_model
        complexity = routing_decision.complexity_level
        reason = routing_decision.routing_reason
        
        # Record metrics
        metrics.record_routing_decision(model, complexity, reason)
        metrics.start_request_timer(request_id, model)
        
        # Process request (placeholder for actual model inference)
        response = QueryResponse(
            content=f"Response from {model}",
            model=model,
            tokens_used=150,
            latency_ms=routing_decision.estimated_latency
        )
        
        metrics.end_request_timer(request_id, model, "success")
        
        logger.info(
            f"Routed query to {model} (complexity: {complexity}, reason: {reason})"
        )
        
        return response
        
    except Exception as e:
        metrics.end_request_timer(request_id, "unknown", "error")
        logger.error(f"Query routing failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)