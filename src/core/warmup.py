from typing import Dict, List, Optional
import asyncio
import logging
from datetime import datetime, timedelta

from .models import ModelConfig, ModelType
from .exceptions import ModelWarmupError

logger = logging.getLogger(__name__)


class ModelWarmup:
    """Handles model warmup to reduce cold start latency."""
    
    def __init__(self, warmup_timeout: float = 30.0):
        self.warmup_timeout = warmup_timeout
        self.warmed_models: Dict[str, datetime] = {}
        self.warmup_requests = {
            ModelType.LOCAL: "Hello, world!",
            ModelType.CLOUD: "What is the capital of France?",
            ModelType.EDGE: "Test prompt"
        }
    
    async def warmup_model(self, model_config: ModelConfig) -> bool:
        """Warm up a specific model with a test request."""
        try:
            logger.info(f"Warming up model {model_config.name}")
            
            # Use appropriate warmup request based on model type
            warmup_prompt = self.warmup_requests.get(
                model_config.type, 
                "Test warmup request"
            )
            
            # Simulate model inference (replace with actual model call)
            await asyncio.sleep(0.1)  # Placeholder for actual warmup
            
            self.warmed_models[model_config.name] = datetime.now()
            logger.info(f"Successfully warmed up {model_config.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to warm up {model_config.name}: {e}")
            raise ModelWarmupError(f"Warmup failed for {model_config.name}: {e}")
    
    async def warmup_models(self, models: List[ModelConfig]) -> Dict[str, bool]:
        """Warm up multiple models concurrently."""
        tasks = [self.warmup_model(model) for model in models]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            models[i].name: not isinstance(result, Exception)
            for i, result in enumerate(results)
        }
    
    def is_warmed(self, model_name: str, max_age_minutes: int = 30) -> bool:
        """Check if a model is still considered warm."""
        if model_name not in self.warmed_models:
            return False
        
        warmup_time = self.warmed_models[model_name]
        age = datetime.now() - warmup_time
        return age < timedelta(minutes=max_age_minutes)
    
    def get_warmup_status(self) -> Dict[str, dict]:
        """Get status of all warmed models."""
        now = datetime.now()
        return {
            model: {
                "warmed_at": warmup_time.isoformat(),
                "age_minutes": (now - warmup_time).total_seconds() / 60,
                "is_warm": self.is_warmed(model)
            }
            for model, warmup_time in self.warmed_models.items()
        }