from typing import Dict, Any, Optional
from dataclasses import dataclass
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..core.complexity_analyzer import ComplexityAnalyzer
from ..core.models import ModelConfig, get_model_config


@dataclass
class CostEstimate:
    estimated_tokens: int
    estimated_cost_usd: float
    recommended_model: str
    confidence: float


class CostEstimationRequest(BaseModel):
    query: str
    user_budget: Optional[float] = None


class CostEstimationResponse(BaseModel):
    estimated_tokens: int
    estimated_cost_usd: float
    recommended_model: str
    confidence: float
    alternatives: list[Dict[str, Any]]


router = APIRouter(prefix="/cost", tags=["cost-estimation"])


@router.post("/estimate", response_model=CostEstimationResponse)
async def estimate_query_cost(request: CostEstimationRequest):
    """Estimate cost and recommend optimal model for a query."""
    try:
        analyzer = ComplexityAnalyzer()
        complexity = analyzer.analyze_query(request.query)
        
        # Get available models and their costs
        models = {
            "gpt-4": {"cost_per_1k": 0.03, "latency_ms": 2000},
            "gpt-3.5-turbo": {"cost_per_1k": 0.002, "latency_ms": 800},
            "local-llama": {"cost_per_1k": 0.0001, "latency_ms": 1200}
        }
        
        # Estimate tokens (rough heuristic)
        estimated_tokens = max(50, len(request.query.split()) * 1.3)
        
        # Find best model based on complexity and budget
        best_model = "local-llama"
        if complexity.score > 0.7:
            best_model = "gpt-4"
        elif complexity.score > 0.4:
            best_model = "gpt-3.5-turbo"
            
        # Apply budget constraint if provided
        if request.user_budget:
            affordable_models = [
                (name, config) for name, config in models.items()
                if (estimated_tokens / 1000) * config["cost_per_1k"] <= request.user_budget
            ]
            if affordable_models:
                best_model = min(affordable_models, key=lambda x: x[1]["cost_per_1k"])[0]
        
        best_config = models[best_model]
        estimated_cost = (estimated_tokens / 1000) * best_config["cost_per_1k"]
        
        # Generate alternatives
        alternatives = [
            {
                "model": name,
                "cost_usd": (estimated_tokens / 1000) * config["cost_per_1k"],
                "latency_ms": config["latency_ms"]
            }
            for name, config in models.items() if name != best_model
        ]
        
        return CostEstimationResponse(
            estimated_tokens=int(estimated_tokens),
            estimated_cost_usd=round(estimated_cost, 6),
            recommended_model=best_model,
            confidence=complexity.confidence,
            alternatives=alternatives
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cost estimation failed: {str(e)}")
