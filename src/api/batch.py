from typing import List, Dict, Any
from pydantic import BaseModel
from fastapi import HTTPException
import asyncio

from ..core.router import Router
from ..core.models import InferenceRequest, InferenceResponse
from ..core.exceptions import RouterError

class BatchRequest(BaseModel):
    queries: List[InferenceRequest]
    max_concurrency: int = 5

class BatchResponse(BaseModel):
    results: List[InferenceResponse]
    failed_count: int
    success_count: int

class BatchProcessor:
    def __init__(self, router: Router, max_batch_size: int = 50):
        self.router = router
        self.max_batch_size = max_batch_size
    
    async def process_batch(self, batch_request: BatchRequest) -> BatchResponse:
        if len(batch_request.queries) > self.max_batch_size:
            raise HTTPException(400, f"Batch size exceeds limit of {self.max_batch_size}")
        
        semaphore = asyncio.Semaphore(batch_request.max_concurrency)
        
        async def process_single(request: InferenceRequest) -> InferenceResponse:
            async with semaphore:
                try:
                    return await self.router.route(request)
                except RouterError as e:
                    return InferenceResponse(
                        text="Error processing request",
                        model_used="error",
                        latency=0.0,
                        cost=0.0,
                        error=str(e)
                    )
        
        results = await asyncio.gather(
            *[process_single(req) for req in batch_request.queries],
            return_exceptions=False
        )
        
        failed_count = sum(1 for r in results if hasattr(r, 'error') and r.error)
        success_count = len(results) - failed_count
        
        return BatchResponse(
            results=results,
            failed_count=failed_count,
            success_count=success_count
        )