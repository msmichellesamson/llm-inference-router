from typing import Dict, List, Tuple
import re
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ComplexityScore:
    score: float  # 0.0 = simple, 1.0 = complex
    token_count: int
    reasoning_detected: bool
    code_detected: bool
    
class ComplexityAnalyzer:
    """Analyzes query complexity to inform routing decisions."""
    
    def __init__(self):
        self.code_patterns = [
            r'```[\w]*\n[\s\S]*?```',  # Code blocks
            r'def\s+\w+\(',  # Function definitions
            r'class\s+\w+',  # Class definitions
            r'import\s+\w+',  # Imports
        ]
        
        self.reasoning_patterns = [
            r'explain|analyze|compare|evaluate',
            r'why|how|what.*difference',
            r'pros.*cons|advantages.*disadvantages',
            r'step.*step|walkthrough|breakdown',
        ]
    
    def analyze(self, query: str) -> ComplexityScore:
        """Analyze query complexity based on multiple factors."""
        try:
            token_count = self._estimate_tokens(query)
            code_detected = self._has_code(query)
            reasoning_detected = self._requires_reasoning(query)
            
            # Calculate complexity score
            score = self._calculate_score(
                token_count, code_detected, reasoning_detected
            )
            
            return ComplexityScore(
                score=score,
                token_count=token_count,
                reasoning_detected=reasoning_detected,
                code_detected=code_detected
            )
        except Exception as e:
            logger.error(f"Error analyzing complexity: {e}")
            # Default to medium complexity on error
            return ComplexityScore(0.5, len(query.split()), False, False)
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (1 token ≈ 4 characters)."""
        return len(text) // 4
    
    def _has_code(self, text: str) -> bool:
        """Detect if query contains code."""
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in self.code_patterns)
    
    def _requires_reasoning(self, text: str) -> bool:
        """Detect if query requires complex reasoning."""
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in self.reasoning_patterns)
    
    def _calculate_score(self, tokens: int, has_code: bool, needs_reasoning: bool) -> float:
        """Calculate final complexity score."""
        score = 0.0
        
        # Token-based scoring
        if tokens > 500:
            score += 0.4
        elif tokens > 200:
            score += 0.2
        
        # Pattern-based scoring
        if has_code:
            score += 0.3
        if needs_reasoning:
            score += 0.3
        
        return min(score, 1.0)
