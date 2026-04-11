import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from .exceptions import ModelHealthError
from .models import ModelProvider


@dataclass
class HealthStatus:
    """Model health status with metrics."""
    is_healthy: bool
    response_time: Optional[float] = None
    error_rate: float = 0.0
    last_check: Optional[datetime] = None
    consecutive_failures: int = 0


class ModelHealthChecker:
    """Monitors model health and triggers automatic failover."""
    
    def __init__(self, check_interval: int = 30, max_failures: int = 3):
        self.check_interval = check_interval
        self.max_failures = max_failures
        self.health_status: Dict[ModelProvider, HealthStatus] = {}
        self.logger = logging.getLogger(__name__)
        self._running = False
    
    async def start_monitoring(self, providers: List[ModelProvider]):
        """Start health monitoring for providers."""
        for provider in providers:
            self.health_status[provider] = HealthStatus(is_healthy=True)
        
        self._running = True
        asyncio.create_task(self._monitor_loop())
        self.logger.info(f"Started health monitoring for {len(providers)} providers")
    
    async def _monitor_loop(self):
        """Background health check loop."""
        while self._running:
            await asyncio.sleep(self.check_interval)
            await self._check_all_models()
    
    async def _check_all_models(self):
        """Check health of all registered models."""
        tasks = [self._check_model_health(provider) for provider in self.health_status]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _check_model_health(self, provider: ModelProvider):
        """Perform health check on a single model."""
        start_time = datetime.now()
        
        try:
            # Simple health check with minimal prompt
            response_time = await self._ping_model(provider)
            
            # Update health status on success
            self.health_status[provider] = HealthStatus(
                is_healthy=True,
                response_time=response_time,
                last_check=start_time,
                consecutive_failures=0
            )
            
        except Exception as e:
            # Update health status on failure
            current = self.health_status[provider]
            failures = current.consecutive_failures + 1
            
            self.health_status[provider] = HealthStatus(
                is_healthy=failures < self.max_failures,
                last_check=start_time,
                consecutive_failures=failures
            )
            
            if failures >= self.max_failures:
                self.logger.warning(f"Model {provider.name} marked unhealthy after {failures} failures")
    
    async def _ping_model(self, provider: ModelProvider) -> float:
        """Send health check request to model."""
        start = datetime.now()
        
        # Mock health check - in real implementation, send minimal request
        await asyncio.sleep(0.1)  # Simulate network call
        
        return (datetime.now() - start).total_seconds()
    
    def get_healthy_providers(self) -> List[ModelProvider]:
        """Return list of currently healthy providers."""
        return [
            provider for provider, status in self.health_status.items()
            if status.is_healthy
        ]
    
    def is_provider_healthy(self, provider: ModelProvider) -> bool:
        """Check if specific provider is healthy."""
        return self.health_status.get(provider, HealthStatus(False)).is_healthy
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self._running = False
        self.logger.info("Stopped health monitoring")
