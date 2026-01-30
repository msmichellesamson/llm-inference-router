import asyncio
import os
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import structlog
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import redis.asyncio as redis

from .complexity_analyzer import ComplexityAnalyzer, QueryComplexity
from .model_router import ModelRouter, ModelTarget, RoutingDecision
from .models.local import LocalModelManager
from .models.cloud import CloudModelManager
from .metrics import MetricsCollector
from .exceptions import (
    RouterError,
    ModelUnavailableError,
    ComplexityAnalysisError,
    ConfigurationError
)

logger = structlog.get_logger()

# Prometheus metrics
REQUEST_COUNT = Counter('llm_router_requests_total', 'Total requests', ['model', 'status'])
REQUEST_DURATION = Histogram('llm_router_request_duration_seconds', 'Request duration', ['model'])
ROUTING_DECISIONS = Counter('llm_router_routing_decisions_total', 'Routing decisions', ['target', 'reason'])

class QueryRequest(BaseModel):
    """Request model for LLM inference."""
    query: str = Field(..., min_length=1, max_length=8192, description="Query text")
    max_tokens: Optional[int] = Field(512, ge=1, le=4096, description="Maximum tokens in response")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    priority: Optional[str] = Field("normal", regex="^(low|normal|high|critical)$", description="Query priority")
    user_id: Optional[str] = Field(None, description="User identifier for tracking")

class QueryResponse(BaseModel):
    """Response model for LLM inference."""
    response: str = Field(..., description="Generated response")
    model_used: str = Field(..., description="Model that generated the response")
    tokens_used: int = Field(..., ge=0, description="Tokens consumed")
    latency_ms: int = Field(..., ge=0, description="Response latency in milliseconds")
    cost_estimate: float = Field(..., ge=0.0, description="Estimated cost in USD")
    routing_reason: str = Field(..., description="Why this model was selected")

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    models_available: Dict[str, bool] = Field(..., description="Model availability")
    redis_connected: bool = Field(..., description="Redis connection status")
    uptime_seconds: int = Field(..., description="Service uptime")

class RouterService:
    """Main router service handling model selection and inference."""
    
    def __init__(self):
        self.start_time = time.time()
        self.complexity_analyzer: Optional[ComplexityAnalyzer] = None
        self.model_router: Optional[ModelRouter] = None
        self.local_manager: Optional[LocalModelManager] = None
        self.cloud_manager: Optional[CloudModelManager] = None
        self.redis_client: Optional[redis.Redis] = None
        self.metrics_collector: Optional[MetricsCollector] = None

    async def initialize(self) -> None:
        """Initialize all service components."""
        try:
            logger.info("Initializing router service")
            
            # Initialize Redis connection
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            await self.redis_client.ping()
            logger.info("Redis connection established")
            
            # Initialize metrics collector
            self.metrics_collector = MetricsCollector(self.redis_client)
            
            # Initialize complexity analyzer
            self.complexity_analyzer = ComplexityAnalyzer()
            await self.complexity_analyzer.initialize()
            
            # Initialize model managers
            self.local_manager = LocalModelManager()
            await self.local_manager.initialize()
            
            self.cloud_manager = CloudModelManager(
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
            await self.cloud_manager.initialize()
            
            # Initialize router with managers
            self.model_router = ModelRouter(
                local_manager=self.local_manager,
                cloud_manager=self.cloud_manager,
                metrics_collector=self.metrics_collector
            )
            
            logger.info("Router service initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize router service", error=str(e))
            raise ConfigurationError(f"Service initialization failed: {e}") from e

    async def shutdown(self) -> None:
        """Cleanup service resources."""
        logger.info("Shutting down router service")
        
        if self.redis_client:
            await self.redis_client.close()
        
        if self.local_manager:
            await self.local_manager.cleanup()
        
        if self.cloud_manager:
            await self.cloud_manager.cleanup()

    async def process_query(self, request: QueryRequest) -> QueryResponse:
        """Process a query through the routing pipeline."""
        start_time = time.time()
        
        try:
            # Analyze query complexity
            if not self.complexity_analyzer:
                raise RouterError("Complexity analyzer not initialized")
            
            complexity = await self.complexity_analyzer.analyze(request.query)
            logger.info("Query complexity analyzed", 
                       complexity=complexity.level.value,
                       confidence=complexity.confidence)
            
            # Get routing decision
            if not self.model_router:
                raise RouterError("Model router not initialized")
            
            routing_decision = await self.model_router.route_query(
                complexity=complexity,
                max_tokens=request.max_tokens,
                priority=request.priority,
                user_id=request.user_id
            )
            
            ROUTING_DECISIONS.labels(
                target=routing_decision.target.value,
                reason=routing_decision.reason
            ).inc()
            
            logger.info("Routing decision made",
                       target=routing_decision.target.value,
                       model=routing_decision.model_name,
                       reason=routing_decision.reason)
            
            # Execute inference
            result = await self._execute_inference(request, routing_decision)
            
            # Record metrics
            latency_ms = int((time.time() - start_time) * 1000)
            REQUEST_COUNT.labels(model=result.model_used, status="success").inc()
            REQUEST_DURATION.labels(model=result.model_used).observe(time.time() - start_time)
            
            # Update result with timing
            result.latency_ms = latency_ms
            
            logger.info("Query processed successfully",
                       model=result.model_used,
                       tokens=result.tokens_used,
                       latency_ms=latency_ms)
            
            return result
            
        except Exception as e:
            REQUEST_COUNT.labels(model="unknown", status="error").inc()
            logger.error("Query processing failed", error=str(e))
            raise

    async def _execute_inference(self, request: QueryRequest, decision: RoutingDecision) -> QueryResponse:
        """Execute inference using the selected model."""
        if decision.target == ModelTarget.LOCAL:
            if not self.local_manager:
                raise ModelUnavailableError("Local model manager not available")
            result = await self.local_manager.generate(
                model_name=decision.model_name,
                prompt=request.query,
                max_tokens=request.max_tokens,
                temperature=request.temperature
            )
        elif decision.target == ModelTarget.CLOUD:
            if not self.cloud_manager:
                raise ModelUnavailableError("Cloud model manager not available")
            result = await self.cloud_manager.generate(
                model_name=decision.model_name,
                prompt=request.query,
                max_tokens=request.max_tokens,
                temperature=request.temperature
            )
        else:
            raise RouterError(f"Unknown model target: {decision.target}")
        
        return QueryResponse(
            response=result.text,
            model_used=decision.model_name,
            tokens_used=result.tokens_used,
            latency_ms=0,  # Will be set by caller
            cost_estimate=result.cost_estimate,
            routing_reason=decision.reason
        )

    async def get_health(self) -> HealthResponse:
        """Get service health status."""
        models_status = {}
        redis_connected = False
        
        # Check Redis connection
        try:
            if self.redis_client:
                await self.redis_client.ping()
                redis_connected = True
        except Exception:
            pass
        
        # Check model availability
        if self.local_manager:
            local_models = await self.local_manager.get_available_models()
            for model in local_models:
                models_status[f"local/{model}"] = True
        
        if self.cloud_manager:
            cloud_models = await self.cloud_manager.get_available_models()
            for model in cloud_models:
                models_status[f"cloud/{model}"] = True
        
        uptime = int(time.time() - self.start_time)
        
        return HealthResponse(
            status="healthy" if models_status and redis_connected else "degraded",
            models_available=models_status,
            redis_connected=redis_connected,
            uptime_seconds=uptime
        )

# Global service instance
router_service = RouterService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    await router_service.initialize()
    yield
    # Shutdown
    await router_service.shutdown()

# FastAPI application
app = FastAPI(
    title="LLM Inference Router",
    description="Multi-model LLM router optimizing cost and latency",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Request logging middleware."""
    start_time = time.time()
    
    # Log request
    logger.info("Request received",
               method=request.method,
               path=request.url.path,
               client_ip=request.client.host if request.client else "unknown")
    
    response = await call_next(request)
    
    # Log response
    duration_ms = int((time.time() - start_time) * 1000)
    logger.info("Request completed",
               method=request.method,
               path=request.url.path,
               status_code=response.status_code,
               duration_ms=duration_ms)
    
    return response

@app.post("/v1/query", response_model=QueryResponse)
async def query_llm(request: QueryRequest) -> QueryResponse:
    """Process a query through the LLM router."""
    try:
        return await router_service.process_query(request)
    except ComplexityAnalysisError as e:
        logger.error("Complexity analysis failed", error=str(e))
        raise HTTPException(status_code=422, detail=f"Query analysis failed: {e}")
    except ModelUnavailableError as e:
        logger.error("Model unavailable", error=str(e))
        raise HTTPException(status_code=503, detail=f"Service unavailable: {e}")
    except RouterError as e:
        logger.error("Router error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")
    except Exception as e:
        logger.error("Unexpected error", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Get service health status."""
    return await router_service.get_health()

@app.get("/metrics")
async def metrics() -> Response:
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/models")
async def list_models() -> Dict[str, List[str]]:
    """List available models by category."""
    local_models = []
    cloud_models = []
    
    if router_service.local_manager:
        local_models = await router_service.local_manager.get_available_models()
    
    if router_service.cloud_manager:
        cloud_models = await router_service.cloud_manager.get_available_models()
    
    return {
        "local": local_models,
        "cloud": cloud_models
    }

@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint."""
    return {
        "service": "LLM Inference Router",
        "version": "1.0.0",
        "status": "running"
    }

if __name__ == "__main__":
    import uvicorn
    
    # Configure structured logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Run server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        workers=1,
        log_config=None,  # Use structlog
        access_log=False,  # Use our middleware
    )