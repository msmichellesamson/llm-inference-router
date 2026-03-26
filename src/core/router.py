# Update imports
from .retry import RetryHandler, RetryConfig

# Add to Router class __init__:
self.retry_handler = RetryHandler(
    RetryConfig(max_attempts=3, base_delay=0.5, max_delay=30.0)
)

# Update route_request method to use retry:
async def route_request(self, query: str, user_id: str = None) -> ModelResponse:
    """Route request to optimal model with retry logic."""
    complexity = self.complexity_analyzer.analyze(query)
    selected_model = await self._select_model(complexity, user_id)
    
    try:
        # Use retry handler for model requests
        response = await self.retry_handler.retry_async(
            self._make_request,
            selected_model,
            query,
            user_id
        )
        
        self.metrics.record_request(
            model=selected_model.name,
            latency=response.latency,
            success=True
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Request failed after retries: {e}")
        self.metrics.record_request(
            model=selected_model.name,
            success=False,
            error=str(e)
        )
        raise