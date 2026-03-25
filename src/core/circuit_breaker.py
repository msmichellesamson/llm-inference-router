"""Circuit breaker implementation for model endpoints."""

import time
from enum import Enum
from typing import Dict, Optional
from dataclasses import dataclass
import logging

from .exceptions import CircuitOpenError, ModelUnavailableError, ErrorCode

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitConfig:
    failure_threshold: int = 5
    timeout: int = 60
    recovery_timeout: int = 30


class CircuitBreaker:
    """Circuit breaker for model endpoints."""
    
    def __init__(self, config: CircuitConfig = None):
        self.config = config or CircuitConfig()
        self.circuits: Dict[str, dict] = {}
    
    def call(self, model_id: str, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        circuit = self._get_circuit(model_id)
        
        if circuit["state"] == CircuitState.OPEN:
            if time.time() - circuit["opened_at"] > self.config.recovery_timeout:
                logger.info(f"Circuit breaker transitioning to half-open: {model_id}")
                circuit["state"] = CircuitState.HALF_OPEN
            else:
                retry_after = int(self.config.recovery_timeout - (time.time() - circuit["opened_at"]))
                raise CircuitOpenError(model_id, retry_after)
        
        try:
            result = func(*args, **kwargs)
            self._on_success(model_id)
            return result
        except Exception as e:
            self._on_failure(model_id, e)
            raise
    
    def _get_circuit(self, model_id: str) -> dict:
        """Get or create circuit state for model."""
        if model_id not in self.circuits:
            self.circuits[model_id] = {
                "state": CircuitState.CLOSED,
                "failure_count": 0,
                "opened_at": None,
                "last_failure": None
            }
        return self.circuits[model_id]
    
    def _on_success(self, model_id: str):
        """Handle successful call."""
        circuit = self.circuits[model_id]
        if circuit["state"] == CircuitState.HALF_OPEN:
            logger.info(f"Circuit breaker closed after recovery: {model_id}")
        circuit.update({
            "state": CircuitState.CLOSED,
            "failure_count": 0,
            "opened_at": None
        })
    
    def _on_failure(self, model_id: str, error: Exception):
        """Handle failed call."""
        circuit = self.circuits[model_id]
        circuit["failure_count"] += 1
        circuit["last_failure"] = str(error)
        
        if circuit["failure_count"] >= self.config.failure_threshold:
            if circuit["state"] != CircuitState.OPEN:
                logger.warning(f"Circuit breaker opened for {model_id} after {circuit['failure_count']} failures")
                circuit["state"] = CircuitState.OPEN
                circuit["opened_at"] = time.time()
