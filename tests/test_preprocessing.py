import pytest
from unittest.mock import Mock, patch
from src.api.preprocessing import preprocess_query, validate_query_format, sanitize_input
from src.core.exceptions import ValidationError


class TestQueryPreprocessing:
    """Test query preprocessing functionality."""
    
    def test_preprocess_query_basic(self):
        """Test basic query preprocessing."""
        query = "  What is the capital of France?  "
        result = preprocess_query(query)
        
        assert result["original"] == query
        assert result["cleaned"] == "What is the capital of France?"
        assert result["token_count"] > 0
        assert "complexity_score" in result
    
    def test_preprocess_query_empty(self):
        """Test preprocessing empty query raises validation error."""
        with pytest.raises(ValidationError, match="Query cannot be empty"):
            preprocess_query("")
        
        with pytest.raises(ValidationError, match="Query cannot be empty"):
            preprocess_query("   ")
    
    def test_validate_query_format_valid(self):
        """Test valid query format validation."""
        query = "Tell me about machine learning"
        assert validate_query_format(query) is True
    
    def test_validate_query_format_invalid(self):
        """Test invalid query format validation."""
        # Too long
        long_query = "x" * 10001
        assert validate_query_format(long_query) is False
        
        # Contains invalid characters
        invalid_query = "What is\x00this?"
        assert validate_query_format(invalid_query) is False
    
    def test_sanitize_input(self):
        """Test input sanitization removes dangerous content."""
        malicious = "<script>alert('xss')</script>What is AI?"
        result = sanitize_input(malicious)
        
        assert "<script>" not in result
        assert "What is AI?" in result
        
    def test_token_counting_accuracy(self):
        """Test token counting gives reasonable estimates."""
        simple_query = "Hello world"
        complex_query = "Explain the intricacies of quantum computing and its applications in cryptography"
        
        simple_result = preprocess_query(simple_query)
        complex_result = preprocess_query(complex_query)
        
        assert simple_result["token_count"] < complex_result["token_count"]
        assert simple_result["complexity_score"] < complex_result["complexity_score"]
