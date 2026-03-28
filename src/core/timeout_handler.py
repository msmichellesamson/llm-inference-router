"""Request timeout handling with circuit breaker integration."""

import asyncio
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass
from .exceptions import TimeoutError, ModelUnavailableError
from .circuit_breaker import CircuitBreaker


@dataclass
class TimeoutConfig:
    """Configuration for request timeouts."""
    default_timeout: float = 30.0
    local_model_timeout: float = 10.0
    cloud_model_timeout: float = 45.0
    complexity_based_scaling: bool = True


class TimeoutHandler:
    """Handles request timeouts with intelligent scaling based on complexity."""
    
    def __init__(self, config: TimeoutConfig, circuit_breaker: CircuitBreaker):
        self.config = config
        self.circuit_breaker = circuit_breaker
        self._timeout_history: Dict[str, list] = {}
    
    def calculate_timeout(self, model_type: str, complexity_score: float) -> float:
        """Calculate timeout based on model type and complexity."""
        if model_type == "local":
            base_timeout = self.config.local_model_timeout
        elif model_type == "cloud":
            base_timeout = self.config.cloud_model_timeout
        else:
            base_timeout = self.config.default_timeout
        
        if self.config.complexity_based_scaling:
            # Scale timeout based on complexity (0.1-1.0 -> 0.8x-1.5x multiplier)
            multiplier = 0.8 + (complexity_score * 0.7)
            return base_timeout * multiplier
        
        return base_timeout
    
    async def execute_with_timeout(
        self, 
        coro,
        model_name: str,
        timeout: float,
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Execute coroutine with timeout and circuit breaker integration."""
        start_time = time.time()
        
        try:
            result = await asyncio.wait_for(coro, timeout=timeout)
            
            # Record successful execution time
            execution_time = time.time() - start_time
            self._record_execution_time(model_name, execution_time)
            
            return result
            
        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            
            # Notify circuit breaker of timeout
            self.circuit_breaker.record_failure(model_name)
            
            raise TimeoutError(
                f"Request to {model_name} timed out after {execution_time:.2f}s",
                model_name=model_name,
                timeout=timeout,
                context=context or {}
            )
    
    def _record_execution_time(self, model_name: str, execution_time: float):
        """Record execution time for timeout optimization."""
        if model_name not in self._timeout_history:
            self._timeout_history[model_name] = []
        
        # Keep last 50 execution times
        history = self._timeout_history[model_name]
        history.append(execution_time)
        if len(history) > 50:
            history.pop(0)
    
    def get_avg_execution_time(self, model_name: str) -> Optional[float]:
        """Get average execution time for a model."""
        if model_name not in self._timeout_history:
            return None
        
        history = self._timeout_history[model_name]
        return sum(history) / len(history) if history else None
