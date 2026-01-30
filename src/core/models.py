from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any, AsyncGenerator
import asyncio
import json
import time
from pathlib import Path

import structlog
import aiohttp
import numpy as np
from prometheus_client import Counter, Histogram, Gauge
import onnxruntime as ort

from .exceptions import ModelLoadError, InferenceError, ModelTimeoutError


logger = structlog.get_logger(__name__)

# Prometheus metrics
MODEL_REQUESTS = Counter(
    "llm_router_model_requests_total",
    "Total requests to models",
    ["model_type", "model_name", "status"]
)

MODEL_LATENCY = Histogram(
    "llm_router_model_latency_seconds",
    "Model inference latency",
    ["model_type", "model_name"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

MODEL_QUEUE_SIZE = Gauge(
    "llm_router_model_queue_size",
    "Current model queue size",
    ["model_type", "model_name"]
)


class ModelType(str, Enum):
    LOCAL_ONNX = "local_onnx"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"


@dataclass
class ModelConfig:
    """Configuration for a model instance."""
    name: str
    model_type: ModelType
    cost_per_token: float
    max_tokens: int
    timeout_seconds: float
    config: Dict[str, Any]


@dataclass
class InferenceRequest:
    """Request for model inference."""
    prompt: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stream: bool = False
    metadata: Dict[str, Any] = None


@dataclass
class InferenceResponse:
    """Response from model inference."""
    text: str
    tokens_used: int
    latency_ms: float
    model_name: str
    cost_estimate: float
    metadata: Dict[str, Any] = None


class BaseModelProvider(ABC):
    """Abstract base class for all model providers."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.logger = logger.bind(
            model_type=config.model_type,
            model_name=config.name
        )
        self._queue_size = 0
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the model provider."""
        pass
    
    @abstractmethod
    async def inference(self, request: InferenceRequest) -> InferenceResponse:
        """Perform inference on the model."""
        pass
    
    @abstractmethod
    async def stream_inference(
        self, 
        request: InferenceRequest
    ) -> AsyncGenerator[str, None]:
        """Perform streaming inference on the model."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the model is healthy."""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Cleanup model resources."""
        pass
    
    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars per token)."""
        return len(text) // 4
    
    def calculate_cost(self, tokens: int) -> float:
        """Calculate estimated cost for tokens."""
        return tokens * self.config.cost_per_token
    
    async def _track_request(self, func, *args, **kwargs):
        """Wrapper to track metrics for requests."""
        start_time = time.time()
        self._queue_size += 1
        MODEL_QUEUE_SIZE.labels(
            model_type=self.config.model_type,
            model_name=self.config.name
        ).set(self._queue_size)
        
        try:
            result = await func(*args, **kwargs)
            MODEL_REQUESTS.labels(
                model_type=self.config.model_type,
                model_name=self.config.name,
                status="success"
            ).inc()
            return result
        except Exception as e:
            MODEL_REQUESTS.labels(
                model_type=self.config.model_type,
                model_name=self.config.name,
                status="error"
            ).inc()
            raise
        finally:
            latency = time.time() - start_time
            MODEL_LATENCY.labels(
                model_type=self.config.model_type,
                model_name=self.config.name
            ).observe(latency)
            self._queue_size = max(0, self._queue_size - 1)
            MODEL_QUEUE_SIZE.labels(
                model_type=self.config.model_type,
                model_name=self.config.name
            ).set(self._queue_size)


class ONNXModelProvider(BaseModelProvider):
    """Local ONNX model provider for fast inference."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.session: Optional[ort.InferenceSession] = None
        self.tokenizer = None
        self._semaphore = asyncio.Semaphore(
            config.config.get("max_concurrent", 4)
        )
    
    async def initialize(self) -> None:
        """Initialize ONNX runtime session."""
        try:
            model_path = Path(self.config.config["model_path"])
            if not model_path.exists():
                raise ModelLoadError(
                    f"ONNX model not found: {model_path}"
                )
            
            # Configure ONNX session
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = self.config.config.get(
                "num_threads", 4
            )
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            
            providers = self.config.config.get("providers", ["CPUExecutionProvider"])
            
            self.session = ort.InferenceSession(
                str(model_path),
                sess_options,
                providers=providers
            )
            
            self.logger.info(
                "ONNX model initialized",
                model_path=str(model_path),
                providers=providers
            )
            
        except Exception as e:
            raise ModelLoadError(f"Failed to initialize ONNX model: {e}")
    
    async def inference(self, request: InferenceRequest) -> InferenceResponse:
        """Perform ONNX model inference."""
        return await self._track_request(self._inference_impl, request)
    
    async def _inference_impl(self, request: InferenceRequest) -> InferenceResponse:
        if not self.session:
            raise InferenceError("Model not initialized")
        
        async with self._semaphore:
            start_time = time.time()
            
            try:
                # Prepare inputs (simplified - would need proper tokenization)
                input_ids = self._tokenize(request.prompt)
                max_length = min(
                    request.max_tokens or self.config.max_tokens,
                    self.config.max_tokens
                )
                
                # Run inference
                inputs = {
                    "input_ids": input_ids,
                    "attention_mask": np.ones_like(input_ids),
                    "max_length": np.array([max_length], dtype=np.int64)
                }
                
                # Execute with timeout
                try:
                    result = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda: self.session.run(None, inputs)
                        ),
                        timeout=self.config.timeout_seconds
                    )
                except asyncio.TimeoutError:
                    raise ModelTimeoutError(
                        f"ONNX inference timeout after {self.config.timeout_seconds}s"
                    )
                
                # Process output
                output_text = self._detokenize(result[0])
                tokens_used = len(result[0][0])
                latency_ms = (time.time() - start_time) * 1000
                
                return InferenceResponse(
                    text=output_text,
                    tokens_used=tokens_used,
                    latency_ms=latency_ms,
                    model_name=self.config.name,
                    cost_estimate=self.calculate_cost(tokens_used),
                    metadata={"provider": "onnx"}
                )
                
            except Exception as e:
                self.logger.error("ONNX inference failed", error=str(e))
                raise InferenceError(f"ONNX inference failed: {e}")
    
    async def stream_inference(
        self, 
        request: InferenceRequest
    ) -> AsyncGenerator[str, None]:
        """ONNX doesn't support native streaming, simulate with chunks."""
        response = await self.inference(request)
        
        # Simulate streaming by chunking the response
        chunk_size = 10
        words = response.text.split()
        
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            if i + chunk_size < len(words):
                chunk += " "
            yield chunk
            await asyncio.sleep(0.1)  # Simulate processing time
    
    async def health_check(self) -> bool:
        """Check if ONNX model is ready."""
        try:
            if not self.session:
                return False
            
            # Simple test inference
            test_input = np.array([[1, 2, 3]], dtype=np.int64)
            test_inputs = {
                "input_ids": test_input,
                "attention_mask": np.ones_like(test_input)
            }
            
            await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.session.run(None, test_inputs)
                ),
                timeout=5.0
            )
            return True
            
        except Exception as e:
            self.logger.error("ONNX health check failed", error=str(e))
            return False
    
    async def shutdown(self) -> None:
        """Clean up ONNX resources."""
        if self.session:
            self.session = None
        self.logger.info("ONNX model shutdown complete")
    
    def _tokenize(self, text: str) -> np.ndarray:
        """Simple tokenization (would use proper tokenizer in production)."""
        # Placeholder - would use actual tokenizer
        tokens = [ord(c) % 1000 for c in text[:512]]  # Simple char-based
        return np.array([tokens], dtype=np.int64)
    
    def _detokenize(self, tokens: np.ndarray) -> str:
        """Simple detokenization."""
        # Placeholder - would use actual detokenizer
        chars = [chr(min(max(int(t) % 128, 32), 126)) for t in tokens[0]]
        return "Generated response: " + "".join(chars[:100])


class OpenAIModelProvider(BaseModelProvider):
    """OpenAI API model provider."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.api_key = config.config.get("api_key", "")
        self.base_url = config.config.get(
            "base_url", 
            "https://api.openai.com/v1"
        )
        self.session: Optional[aiohttp.ClientSession] = None
        self._semaphore = asyncio.Semaphore(
            config.config.get("max_concurrent", 10)
        )
    
    async def initialize(self) -> None:
        """Initialize HTTP session."""
        timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        )
        
        self.logger.info("OpenAI provider initialized")
    
    async def inference(self, request: InferenceRequest) -> InferenceResponse:
        """Perform OpenAI API inference."""
        return await self._track_request(self._inference_impl, request)
    
    async def _inference_impl(self, request: InferenceRequest) -> InferenceResponse:
        if not self.session:
            raise InferenceError("OpenAI session not initialized")
        
        async with self._semaphore:
            start_time = time.time()
            
            payload = {
                "model": self.config.config["model"],
                "messages": [{"role": "user", "content": request.prompt}],
                "max_tokens": request.max_tokens or self.config.max_tokens,
                "temperature": request.temperature or 0.7,
                "stream": False
            }
            
            try:
                async with self.session.post(
                    f"{self.base_url}/chat/completions",
                    json=payload
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise InferenceError(
                            f"OpenAI API error {response.status}: {error_text}"
                        )
                    
                    data = await response.json()
                    
                    if "error" in data:
                        raise InferenceError(f"OpenAI error: {data['error']}")
                    
                    choice = data["choices"][0]
                    tokens_used = data["usage"]["total_tokens"]
                    latency_ms = (time.time() - start_time) * 1000
                    
                    return InferenceResponse(
                        text=choice["message"]["content"],
                        tokens_used=tokens_used,
                        latency_ms=latency_ms,
                        model_name=self.config.name,
                        cost_estimate=self.calculate_cost(tokens_used),
                        metadata={
                            "provider": "openai",
                            "finish_reason": choice["finish_reason"]
                        }
                    )
                    
            except aiohttp.ClientError as e:
                raise InferenceError(f"OpenAI request failed: {e}")
            except asyncio.TimeoutError:
                raise ModelTimeoutError(
                    f"OpenAI request timeout after {self.config.timeout_seconds}s"
                )
    
    async def stream_inference(
        self, 
        request: InferenceRequest
    ) -> AsyncGenerator[str, None]:
        """Perform streaming OpenAI inference."""
        if not self.session:
            raise InferenceError("OpenAI session not initialized")
        
        async with self._semaphore:
            payload = {
                "model": self.config.config["model"],
                "messages": [{"role": "user", "content": request.prompt}],
                "max_tokens": request.max_tokens or self.config.max_tokens,
                "temperature": request.temperature or 0.7,
                "stream": True
            }
            
            try:
                async with self.session.post(
                    f"{self.base_url}/chat/completions",
                    json=payload
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise InferenceError(
                            f"OpenAI API error {response.status}: {error_text}"
                        )
                    
                    async for line in response.content:
                        line = line.decode().strip()
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str == "[DONE]":
                                break
                            
                            try:
                                data = json.loads(data_str)
                                if "choices" in data and data["choices"]:
                                    delta = data["choices"][0].get("delta", {})
                                    if "content" in delta:
                                        yield delta["content"]
                            except json.JSONDecodeError:
                                continue
                                
            except Exception as e:
                self.logger.error("OpenAI streaming failed", error=str(e))
                raise InferenceError(f"OpenAI streaming failed: {e}")
    
    async def health_check(self) -> bool:
        """Check OpenAI API health."""
        if not self.session:
            return False
        
        try:
            async with self.session.get(f"{self.base_url}/models") as response:
                return response.status == 200
        except Exception:
            return False
    
    async def shutdown(self) -> None:
        """Close HTTP session."""
        if self.session:
            await self.session.close()
        self.logger.info("OpenAI provider shutdown complete")


class ModelRegistry:
    """Registry for managing multiple model providers."""
    
    def __init__(self):
        self.providers: Dict[str, BaseModelProvider] = {}
        self.logger = logger.bind(component="model_registry")
    
    async def register_model(self, config: ModelConfig) -> None:
        """Register a new model provider."""
        try:
            if config.model_type == ModelType.LOCAL_ONNX:
                provider = ONNXModelProvider(config)
            elif config.model_type == ModelType.OPENAI:
                provider = OpenAIModelProvider(config)
            else:
                raise ValueError(f"Unsupported model type: {config.model_type}")
            
            await provider.initialize()
            self.providers[config.name] = provider
            
            self.logger.info(
                "Model registered",
                model_name=config.name,
                model_type=config.model_type
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to register model",
                model_name=config.name,
                error=str(e)
            )
            raise ModelLoadError(f"Failed to register model {config.name}: {e}")
    
    def get_provider(self, model_name: str) -> Optional[BaseModelProvider]:
        """Get a model provider by name."""
        return self.providers.get(model_name)
    
    def list_models(self) -> List[str]:
        """List all registered models."""
        return list(self.providers.keys())
    
    async def health_check_all(self) -> Dict[str, bool]:
        """Check health of all registered models."""
        results = {}
        for name, provider in self.providers.items():
            try:
                results[name] = await provider.health_check()
            except Exception as e:
                self.logger.error(
                    "Health check failed",
                    model_name=name,
                    error=str(e)
                )
                results[name] = False
        return results
    
    async def shutdown_all(self) -> None:
        """Shutdown all model providers."""
        for name, provider in self.providers.items():
            try:
                await provider.shutdown()
                self.logger.info("Model shutdown", model_name=name)
            except Exception as e:
                self.logger.error(
                    "Model shutdown failed",
                    model_name=name,
                    error=str(e)
                )
        
        self.providers.clear()