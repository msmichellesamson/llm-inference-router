import pytest
from unittest.mock import Mock, patch
from src.core.complexity_analyzer import ComplexityAnalyzer, QueryComplexity


class TestComplexityAnalyzer:
    def setup_method(self):
        self.analyzer = ComplexityAnalyzer()
    
    def test_simple_query_classification(self):
        """Test classification of simple queries"""
        queries = [
            "What is 2+2?",
            "Hello",
            "Define AI"
        ]
        
        for query in queries:
            complexity = self.analyzer.analyze_query(query)
            assert complexity.level == QueryComplexity.SIMPLE
            assert complexity.confidence > 0.7
    
    def test_medium_query_classification(self):
        """Test classification of medium complexity queries"""
        queries = [
            "Explain the differences between REST and GraphQL APIs with examples",
            "How do I implement caching in a distributed system?",
            "What are the trade-offs between SQL and NoSQL databases?"
        ]
        
        for query in queries:
            complexity = self.analyzer.analyze_query(query)
            assert complexity.level == QueryComplexity.MEDIUM
    
    def test_complex_query_classification(self):
        """Test classification of complex queries"""
        queries = [
            "Design a distributed machine learning system that can handle real-time inference with fault tolerance and auto-scaling capabilities",
            "Analyze the performance implications of different consensus algorithms in blockchain networks"
        ]
        
        for query in queries:
            complexity = self.analyzer.analyze_query(query)
            assert complexity.level == QueryComplexity.COMPLEX
    
    def test_feature_extraction(self):
        """Test query feature extraction"""
        query = "How do I optimize PostgreSQL queries for better performance?"
        features = self.analyzer._extract_features(query)
        
        assert features['word_count'] > 0
        assert features['technical_terms'] > 0
        assert 'question_complexity' in features
        assert 'domain_specificity' in features
    
    def test_model_routing_recommendation(self):
        """Test model routing recommendations based on complexity"""
        simple_query = "Hello world"
        complex_query = "Design a microservices architecture with event sourcing"
        
        simple_result = self.analyzer.analyze_query(simple_query)
        complex_result = self.analyzer.analyze_query(complex_query)
        
        assert simple_result.recommended_model_tier == "local"
        assert complex_result.recommended_model_tier == "cloud"
    
    @patch('src.core.complexity_analyzer.time.time')
    def test_analysis_performance(self, mock_time):
        """Test that analysis completes within acceptable time"""
        mock_time.side_effect = [0.0, 0.05]  # 50ms
        
        query = "Test query for performance measurement"
        result = self.analyzer.analyze_query(query)
        
        assert result.analysis_time_ms <= 100  # Should be under 100ms
    
    def test_batch_analysis(self):
        """Test batch processing of multiple queries"""
        queries = [
            "Simple query",
            "Medium complexity query about databases and caching strategies",
            "Complex architectural question about distributed systems"
        ]
        
        results = self.analyzer.analyze_batch(queries)
        
        assert len(results) == 3
        assert results[0].level == QueryComplexity.SIMPLE
        assert results[1].level == QueryComplexity.MEDIUM
        assert results[2].level == QueryComplexity.COMPLEX
