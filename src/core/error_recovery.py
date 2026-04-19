from typing import Optional, Callable, Any, Dict
import asyncio
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from .exceptions import LLMInferenceError, ModelTimeoutError, ModelOverloadedError
from .metrics import MetricsCollector

logger = logging.getLogger(__name__)


class RecoveryStrategy(Enum):
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    IMMEDIATE_RETRY = "immediate_retry"
    CIRCUIT_BREAK = "circuit_break"


@dataclass
class RecoveryAttempt:
    attempt_number: int
    strategy: RecoveryStrategy
    delay_seconds: float
    timestamp: datetime
    error_type: str


class ErrorRecoveryManager:
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.recovery_history: Dict[str, list[RecoveryAttempt]] = {}
        self.max_attempts = 3
        self.base_delay = 1.0
        self.max_delay = 30.0
    
    async def recover_with_strategy(
        self,
        operation: Callable,
        error: Exception,
        context: str = "unknown"
    ) -> Any:
        """Execute recovery strategy based on error type."""
        strategy = self._select_strategy(error)
        attempt_count = len(self.recovery_history.get(context, []))
        
        if attempt_count >= self.max_attempts:
            logger.error(f"Max recovery attempts exceeded for {context}")
            self.metrics.increment_counter("error_recovery_exhausted_total", {"context": context})
            raise error
        
        delay = self._calculate_delay(strategy, attempt_count)
        
        # Record attempt
        attempt = RecoveryAttempt(
            attempt_number=attempt_count + 1,
            strategy=strategy,
            delay_seconds=delay,
            timestamp=datetime.utcnow(),
            error_type=type(error).__name__
        )
        
        if context not in self.recovery_history:
            self.recovery_history[context] = []
        self.recovery_history[context].append(attempt)
        
        logger.info(
            f"Attempting recovery {attempt_count + 1}/{self.max_attempts} "
            f"for {context} using {strategy.value} (delay: {delay}s)"
        )
        
        await asyncio.sleep(delay)
        
        try:
            result = await operation()
            # Recovery successful - clear history
            self.recovery_history.pop(context, None)
            self.metrics.increment_counter("error_recovery_success_total", {
                "context": context,
                "strategy": strategy.value,
                "attempt": str(attempt_count + 1)
            })
            return result
        except Exception as retry_error:
            self.metrics.increment_counter("error_recovery_failed_total", {
                "context": context,
                "error_type": type(retry_error).__name__
            })
            raise retry_error
    
    def _select_strategy(self, error: Exception) -> RecoveryStrategy:
        """Select recovery strategy based on error type."""
        if isinstance(error, ModelTimeoutError):
            return RecoveryStrategy.LINEAR_BACKOFF
        elif isinstance(error, ModelOverloadedError):
            return RecoveryStrategy.EXPONENTIAL_BACKOFF
        elif isinstance(error, ConnectionError):
            return RecoveryStrategy.EXPONENTIAL_BACKOFF
        else:
            return RecoveryStrategy.IMMEDIATE_RETRY
    
    def _calculate_delay(self, strategy: RecoveryStrategy, attempt: int) -> float:
        """Calculate delay based on strategy and attempt number."""
        if strategy == RecoveryStrategy.EXPONENTIAL_BACKOFF:
            delay = min(self.base_delay * (2 ** attempt), self.max_delay)
        elif strategy == RecoveryStrategy.LINEAR_BACKOFF:
            delay = min(self.base_delay * (attempt + 1), self.max_delay)
        else:  # IMMEDIATE_RETRY
            delay = 0.1
        
        return delay
    
    def get_recovery_stats(self, context: str) -> Optional[Dict[str, Any]]:
        """Get recovery statistics for a context."""
        if context not in self.recovery_history:
            return None
        
        attempts = self.recovery_history[context]
        return {
            "total_attempts": len(attempts),
            "strategies_used": [a.strategy.value for a in attempts],
            "total_delay": sum(a.delay_seconds for a in attempts),
            "last_attempt": attempts[-1].timestamp.isoformat() if attempts else None
        }
