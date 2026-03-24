from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import logging

from ..core.complexity_analyzer import ComplexityAnalyzer
from ..core.metrics import MetricsCollector

router = APIRouter(prefix="/preprocessing", tags=["preprocessing"])
logger = logging.getLogger(__name__)

class QueryPreprocessRequest(BaseModel):
    query: str
    context: Dict[str, Any] = {}
    user_id: str = "anonymous"

class QueryPreprocessResponse(BaseModel):
    complexity_score: float
    recommended_model: str
    estimated_cost: float
    estimated_latency_ms: int
    preprocessing_time_ms: float

@router.post("/analyze", response_model=QueryPreprocessResponse)
async def preprocess_query(request: QueryPreprocessRequest):
    """Analyze query complexity and recommend routing without executing."""
    try:
        analyzer = ComplexityAnalyzer()
        metrics = MetricsCollector()
        
        start_time = metrics._get_current_time_ms()
        
        # Analyze query complexity
        complexity = analyzer.analyze_complexity(
            query=request.query,
            context=request.context
        )
        
        # Recommend model based on complexity
        if complexity.score < 0.3:
            model = "local-small"
            cost = 0.001
            latency = 50
        elif complexity.score < 0.7:
            model = "local-medium" 
            cost = 0.005
            latency = 150
        else:
            model = "cloud-gpt4"
            cost = 0.03
            latency = 800
            
        processing_time = metrics._get_current_time_ms() - start_time
        
        logger.info(f"Query preprocessed: complexity={complexity.score:.3f}, model={model}")
        
        return QueryPreprocessResponse(
            complexity_score=complexity.score,
            recommended_model=model,
            estimated_cost=cost,
            estimated_latency_ms=latency,
            preprocessing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))