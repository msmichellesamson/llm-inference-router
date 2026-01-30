import asyncio
import hashlib
import time
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

import structlog
import redis.asyncio as redis
from prometheus_client import Counter, Histogram, Gauge
import tiktoken
import re
import numpy as np

logger = structlog.get_logger()

# Metrics
ROUTING_DECISIONS = Counter('routing_decisions_total', 'Total routing decisions', ['model_type', 'complexity'])
QUERY_COMPLEXITY_HISTOGRAM = Histogram('query_complexity_score', 'Query complexity analysis scores')
MODEL_LATENCY = Histogram('model_latency_seconds', 'Model inference latency', ['model_name'])
ACTIVE_REQUESTS = Gauge('active_requests', 'Currently processing requests', ['model_name'])
CACHE_HITS = Counter('cache_hits_total', 'Cache hit/miss statistics', ['status'])


class RoutingError(Exception):
    """Base exception for routing errors"""
    pass


class ComplexityAnalysisError(RoutingError):
    """Error during query complexity analysis"""
    pass


class ModelUnavailableError(RoutingError):
    """Model is currently unavailable"""
    pass


class ModelType(Enum):
    LOCAL_SMALL = "local_small"
    LOCAL_MEDIUM = "local_medium"
    CLOUD_GPT4 = "cloud_gpt4"
    CLOUD_CLAUDE = "cloud_claude"


class ComplexityLevel(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"


@dataclass
class QueryMetrics:
    """Metrics extracted from a query for complexity analysis"""
    token_count: int
    sentence_count: int
    technical_terms: int
    code_blocks: int
    math_expressions: int
    question_depth: int
    domain_specificity: float
    reasoning_complexity: float


@dataclass
class RoutingDecision:
    """Result of routing analysis"""
    model_type: ModelType
    complexity_level: ComplexityLevel
    confidence_score: float
    estimated_cost: float
    estimated_latency: float
    reasoning: str
    cache_key: Optional[str] = None


@dataclass
class ModelConfig:
    """Configuration for a specific model"""
    name: str
    model_type: ModelType
    max_tokens: int
    cost_per_1k_tokens: float
    avg_latency_ms: float
    availability_threshold: float
    complexity_range: Tuple[ComplexityLevel, ComplexityLevel]


class ComplexityAnalyzer:
    """Analyzes query complexity using multiple heuristics"""
    
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.technical_patterns = self._load_technical_patterns()
        self.code_pattern = re.compile(r'```[\s\S]*?```|`[^`]+`')
        self.math_pattern = re.compile(r'\$[^$]+\$|\\\([^)]+\\\)|\\\[[^\]]+\\\]')
        
    def _load_technical_patterns(self) -> List[re.Pattern]:
        """Load regex patterns for technical term detection"""
        patterns = [
            r'\b(?:algorithm|optimization|complexity|distributed|microservice)\b',
            r'\b(?:kubernetes|terraform|docker|prometheus|grafana)\b',
            r'\b(?:postgresql|redis|mongodb|elasticsearch)\b',
            r'\b(?:machine learning|deep learning|neural network|transformer)\b',
            r'\b(?:api|rest|grpc|graphql|webhook)\b',
            r'\b(?:cloud|aws|gcp|azure|serverless)\b',
        ]
        return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    
    async def analyze_query(self, query: str) -> QueryMetrics:
        """Analyze query and extract complexity metrics"""
        try:
            start_time = time.time()
            
            # Basic metrics
            tokens = self.tokenizer.encode(query)
            token_count = len(tokens)
            sentences = query.split('.')
            sentence_count = len([s for s in sentences if s.strip()])
            
            # Technical content detection
            technical_terms = sum(len(pattern.findall(query)) for pattern in self.technical_patterns)
            code_blocks = len(self.code_pattern.findall(query))
            math_expressions = len(self.math_pattern.findall(query))
            
            # Advanced analysis
            question_depth = await self._analyze_question_depth(query)
            domain_specificity = await self._calculate_domain_specificity(query)
            reasoning_complexity = await self._analyze_reasoning_complexity(query)
            
            metrics = QueryMetrics(
                token_count=token_count,
                sentence_count=sentence_count,
                technical_terms=technical_terms,
                code_blocks=code_blocks,
                math_expressions=math_expressions,
                question_depth=question_depth,
                domain_specificity=domain_specificity,
                reasoning_complexity=reasoning_complexity
            )
            
            analysis_time = time.time() - start_time
            logger.info("Query complexity analyzed", 
                       token_count=token_count,
                       technical_terms=technical_terms,
                       analysis_time=analysis_time)
            
            return metrics
            
        except Exception as e:
            logger.error("Failed to analyze query complexity", error=str(e))
            raise ComplexityAnalysisError(f"Complexity analysis failed: {e}") from e
    
    async def _analyze_question_depth(self, query: str) -> int:
        """Analyze the depth/complexity of questions being asked"""
        question_indicators = ['why', 'how', 'what if', 'explain', 'compare', 'analyze']
        nested_questions = query.count('?')
        depth_indicators = sum(1 for indicator in question_indicators if indicator in query.lower())
        
        # Multi-part questions increase depth
        conjunctions = len(re.findall(r'\b(?:and|or|but|however|furthermore|moreover)\b', query, re.IGNORECASE))
        
        return min(nested_questions + depth_indicators + conjunctions // 2, 10)
    
    async def _calculate_domain_specificity(self, query: str) -> float:
        """Calculate how domain-specific the query is (0.0 to 1.0)"""
        # Simple heuristic based on technical term density
        words = query.split()
        if not words:
            return 0.0
        
        technical_word_count = sum(len(pattern.findall(query)) for pattern in self.technical_patterns)
        domain_score = min(technical_word_count / len(words), 1.0)
        
        # Boost for code or math content
        if self.code_pattern.search(query) or self.math_pattern.search(query):
            domain_score = min(domain_score + 0.3, 1.0)
        
        return domain_score
    
    async def _analyze_reasoning_complexity(self, query: str) -> float:
        """Analyze complexity of reasoning required (0.0 to 1.0)"""
        reasoning_keywords = [
            'analyze', 'synthesize', 'evaluate', 'critique', 'design', 'architect',
            'optimize', 'troubleshoot', 'debug', 'implement', 'integrate'
        ]
        
        reasoning_score = 0.0
        query_lower = query.lower()
        
        for keyword in reasoning_keywords:
            if keyword in query_lower:
                reasoning_score += 0.1
        
        # Multi-step reasoning indicators
        step_indicators = ['first', 'then', 'next', 'finally', 'step by step', 'approach']
        reasoning_score += sum(0.1 for indicator in step_indicators if indicator in query_lower)
        
        return min(reasoning_score, 1.0)


class RouterCache:
    """Redis-based cache for routing decisions"""
    
    def __init__(self, redis_client: redis.Redis, ttl: int = 3600):
        self.redis = redis_client
        self.ttl = ttl
    
    def _generate_cache_key(self, query: str, model_configs: Dict[str, ModelConfig]) -> str:
        """Generate cache key for routing decision"""
        config_hash = hashlib.md5(str(sorted(model_configs.keys())).encode()).hexdigest()[:8]
        query_hash = hashlib.md5(query.encode()).hexdigest()
        return f"routing:{query_hash}:{config_hash}"
    
    async def get_routing_decision(self, query: str, model_configs: Dict[str, ModelConfig]) -> Optional[RoutingDecision]:
        """Get cached routing decision"""
        try:
            cache_key = self._generate_cache_key(query, model_configs)
            cached_data = await self.redis.hgetall(cache_key)
            
            if not cached_data:
                CACHE_HITS.labels(status="miss").inc()
                return None
            
            CACHE_HITS.labels(status="hit").inc()
            
            return RoutingDecision(
                model_type=ModelType(cached_data['model_type'].decode()),
                complexity_level=ComplexityLevel(cached_data['complexity_level'].decode()),
                confidence_score=float(cached_data['confidence_score']),
                estimated_cost=float(cached_data['estimated_cost']),
                estimated_latency=float(cached_data['estimated_latency']),
                reasoning=cached_data['reasoning'].decode(),
                cache_key=cache_key
            )
            
        except Exception as e:
            logger.warning("Cache retrieval failed", error=str(e))
            return None
    
    async def cache_routing_decision(self, query: str, model_configs: Dict[str, ModelConfig], 
                                   decision: RoutingDecision) -> None:
        """Cache routing decision"""
        try:
            cache_key = self._generate_cache_key(query, model_configs)
            
            data = {
                'model_type': decision.model_type.value,
                'complexity_level': decision.complexity_level.value,
                'confidence_score': decision.confidence_score,
                'estimated_cost': decision.estimated_cost,
                'estimated_latency': decision.estimated_latency,
                'reasoning': decision.reasoning
            }
            
            await self.redis.hset(cache_key, mapping=data)
            await self.redis.expire(cache_key, self.ttl)
            
        except Exception as e:
            logger.warning("Cache storage failed", error=str(e))


class IntelligentRouter:
    """Core routing engine that makes intelligent model selection decisions"""
    
    def __init__(self, redis_client: redis.Redis, model_configs: Dict[str, ModelConfig]):
        self.complexity_analyzer = ComplexityAnalyzer()
        self.cache = RouterCache(redis_client)
        self.model_configs = model_configs
        self.model_health = {name: 1.0 for name in model_configs.keys()}
        
        # Complexity thresholds for routing decisions
        self.complexity_thresholds = {
            ComplexityLevel.SIMPLE: 0.2,
            ComplexityLevel.MODERATE: 0.5,
            ComplexityLevel.COMPLEX: 0.8,
            ComplexityLevel.EXPERT: 1.0
        }
    
    async def route_query(self, query: str, preferences: Optional[Dict[str, Any]] = None) -> RoutingDecision:
        """Route query to optimal model based on complexity and constraints"""
        start_time = time.time()
        
        try:
            # Check cache first
            cached_decision = await self.cache.get_routing_decision(query, self.model_configs)
            if cached_decision:
                logger.info("Using cached routing decision", model=cached_decision.model_type.value)
                return cached_decision
            
            # Analyze query complexity
            metrics = await self.complexity_analyzer.analyze_query(query)
            complexity_score = self._calculate_complexity_score(metrics)
            complexity_level = self._determine_complexity_level(complexity_score)
            
            QUERY_COMPLEXITY_HISTOGRAM.observe(complexity_score)
            
            # Apply preferences and constraints
            preferences = preferences or {}
            max_cost = preferences.get('max_cost', float('inf'))
            max_latency = preferences.get('max_latency_ms', float('inf'))
            preferred_models = preferences.get('preferred_models', [])
            
            # Find suitable models
            suitable_models = await self._find_suitable_models(
                complexity_level, max_cost, max_latency, preferred_models
            )
            
            if not suitable_models:
                raise ModelUnavailableError("No suitable models available for query requirements")
            
            # Select optimal model
            decision = await self._select_optimal_model(
                suitable_models, complexity_score, complexity_level, metrics
            )
            
            # Cache the decision
            await self.cache.cache_routing_decision(query, self.model_configs, decision)
            
            routing_time = time.time() - start_time
            ROUTING_DECISIONS.labels(
                model_type=decision.model_type.value,
                complexity=complexity_level.value
            ).inc()
            
            logger.info("Query routed successfully",
                       model=decision.model_type.value,
                       complexity=complexity_level.value,
                       confidence=decision.confidence_score,
                       routing_time=routing_time)
            
            return decision
            
        except Exception as e:
            logger.error("Routing failed", error=str(e), query_length=len(query))
            raise
    
    def _calculate_complexity_score(self, metrics: QueryMetrics) -> float:
        """Calculate overall complexity score from metrics"""
        # Weighted combination of various factors
        weights = {
            'token_count': 0.15,
            'technical_terms': 0.20,
            'code_blocks': 0.15,
            'math_expressions': 0.10,
            'question_depth': 0.15,
            'domain_specificity': 0.15,
            'reasoning_complexity': 0.10
        }
        
        # Normalize metrics to 0-1 scale
        normalized_metrics = {
            'token_count': min(metrics.token_count / 4000, 1.0),
            'technical_terms': min(metrics.technical_terms / 20, 1.0),
            'code_blocks': min(metrics.code_blocks / 5, 1.0),
            'math_expressions': min(metrics.math_expressions / 10, 1.0),
            'question_depth': min(metrics.question_depth / 10, 1.0),
            'domain_specificity': metrics.domain_specificity,
            'reasoning_complexity': metrics.reasoning_complexity
        }
        
        score = sum(normalized_metrics[key] * weights[key] for key in weights)
        return min(score, 1.0)
    
    def _determine_complexity_level(self, complexity_score: float) -> ComplexityLevel:
        """Determine complexity level from score"""
        if complexity_score <= self.complexity_thresholds[ComplexityLevel.SIMPLE]:
            return ComplexityLevel.SIMPLE
        elif complexity_score <= self.complexity_thresholds[ComplexityLevel.MODERATE]:
            return ComplexityLevel.MODERATE
        elif complexity_score <= self.complexity_thresholds[ComplexityLevel.COMPLEX]:
            return ComplexityLevel.COMPLEX
        else:
            return ComplexityLevel.EXPERT
    
    async def _find_suitable_models(self, complexity_level: ComplexityLevel, 
                                   max_cost: float, max_latency: float,
                                   preferred_models: List[str]) -> List[ModelConfig]:
        """Find models suitable for the complexity level and constraints"""
        suitable_models = []
        
        for name, config in self.model_configs.items():
            # Check if model can handle complexity level
            min_complexity, max_complexity = config.complexity_range
            complexity_values = [level.value for level in ComplexityLevel]
            
            if not (complexity_values.index(min_complexity.value) <= 
                   complexity_values.index(complexity_level.value) <= 
                   complexity_values.index(max_complexity.value)):
                continue
            
            # Check cost and latency constraints
            if config.cost_per_1k_tokens > max_cost:
                continue
            
            if config.avg_latency_ms > max_latency:
                continue
            
            # Check model health
            if self.model_health.get(name, 0) < config.availability_threshold:
                continue
            
            suitable_models.append(config)
        
        # Prioritize preferred models
        if preferred_models:
            suitable_models.sort(
                key=lambda m: (m.name not in preferred_models, m.cost_per_1k_tokens)
            )
        else:
            # Sort by cost-effectiveness
            suitable_models.sort(key=lambda m: m.cost_per_1k_tokens)
        
        return suitable_models
    
    async def _select_optimal_model(self, suitable_models: List[ModelConfig],
                                   complexity_score: float, complexity_level: ComplexityLevel,
                                   metrics: QueryMetrics) -> RoutingDecision:
        """Select the optimal model from suitable candidates"""
        best_model = suitable_models[0]  # Already sorted by preference
        
        # Calculate confidence based on model capability vs requirement
        confidence = self._calculate_routing_confidence(best_model, complexity_score)
        
        # Estimate actual cost and latency
        estimated_tokens = max(metrics.token_count, 100)  # Minimum estimation
        estimated_cost = (estimated_tokens / 1000) * best_model.cost_per_1k_tokens
        estimated_latency = best_model.avg_latency_ms
        
        # Generate reasoning
        reasoning = self._generate_routing_reasoning(
            best_model, complexity_level, confidence, suitable_models
        )
        
        return RoutingDecision(
            model_type=best_model.model_type,
            complexity_level=complexity_level,
            confidence_score=confidence,
            estimated_cost=estimated_cost,
            estimated_latency=estimated_latency,
            reasoning=reasoning
        )
    
    def _calculate_routing_confidence(self, model: ModelConfig, complexity_score: float) -> float:
        """Calculate confidence score for routing decision"""
        # Base confidence on model health
        base_confidence = self.model_health.get(model.name, 0)
        
        # Adjust based on complexity match
        min_complexity, max_complexity = model.complexity_range
        complexity_values = [level.value for level in ComplexityLevel]
        min_idx = complexity_values.index(min_complexity.value)
        max_idx = complexity_values.index(max_complexity.value)
        
        # Optimal range is middle of model's capability
        optimal_range = (max_idx + min_idx) / 2
        current_complexity_idx = complexity_score * (len(complexity_values) - 1)
        
        complexity_match = 1.0 - abs(current_complexity_idx - optimal_range) / len(complexity_values)
        
        return min(base_confidence * (0.7 + 0.3 * complexity_match), 1.0)
    
    def _generate_routing_reasoning(self, selected_model: ModelConfig,
                                  complexity_level: ComplexityLevel,
                                  confidence: float,
                                  alternatives: List[ModelConfig]) -> str:
        """Generate human-readable reasoning for routing decision"""
        reasoning_parts = [
            f"Selected {selected_model.model_type.value} for {complexity_level.value} complexity query"
        ]
        
        if confidence > 0.8:
            reasoning_parts.append("High confidence match for model capabilities")
        elif confidence > 0.6:
            reasoning_parts.append("Good match for model capabilities")
        else:
            reasoning_parts.append("Acceptable match with some limitations")
        
        if len(alternatives) > 1:
            reasoning_parts.append(f"Chosen from {len(alternatives)} suitable models based on cost-effectiveness")
        
        reasoning_parts.append(f"Estimated cost: ${selected_model.cost_per_1k_tokens:.4f}/1k tokens")
        
        return ". ".join(reasoning_parts)
    
    async def update_model_health(self, model_name: str, health_score: float) -> None:
        """Update model health score for routing decisions"""
        if model_name in self.model_configs:
            self.model_health[model_name] = max(0.0, min(health_score, 1.0))
            logger.info("Model health updated", model=model_name, health=health_score)
    
    async def get_routing_stats(self) -> Dict[str, Any]:
        """Get current routing statistics"""
        return {
            'model_health': self.model_health.copy(),
            'total_models': len(self.model_configs),
            'healthy_models': sum(1 for h in self.model_health.values() if h > 0.5),
            'complexity_thresholds': {k.value: v for k, v in self.complexity_thresholds.items()}
        }