from fastapi import HTTPException
from typing import Dict, List, Optional
import logging
from ..core.models import ModelProvider, QueryRequest, QueryResponse
from ..core.exceptions import ModelUnavailableError

logger = logging.getLogger(__name__)

class FallbackHandler:
    """Handles fallback routing when primary models fail"""
    
    def __init__(self):
        self.fallback_chain = [
            ModelProvider.OLLAMA_LLAMA2,
            ModelProvider.OPENAI_GPT35,
            ModelProvider.CLAUDE_HAIKU
        ]
    
    async def get_fallback_model(self, 
                               failed_model: ModelProvider,
                               complexity_score: float) -> Optional[ModelProvider]:
        """Get next available fallback model based on complexity"""
        
        # Remove failed model from chain
        available_models = [m for m in self.fallback_chain if m != failed_model]
        
        if not available_models:
            return None
            
        # For high complexity queries, prefer cloud models
        if complexity_score > 0.7:
            cloud_models = [m for m in available_models 
                          if m in [ModelProvider.OPENAI_GPT35, ModelProvider.CLAUDE_HAIKU]]
            return cloud_models[0] if cloud_models else available_models[0]
        
        # For low complexity, prefer local models
        local_models = [m for m in available_models 
                       if m == ModelProvider.OLLAMA_LLAMA2]
        return local_models[0] if local_models else available_models[0]
    
    def update_fallback_chain(self, new_chain: List[ModelProvider]) -> None:
        """Update the fallback priority chain"""
        self.fallback_chain = new_chain
        logger.info(f"Updated fallback chain: {[m.value for m in new_chain]}")

fallback_handler = FallbackHandler()