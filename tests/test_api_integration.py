import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from src.main import app
from src.core.models import QueryRequest, ModelConfig, ModelType

client = TestClient(app)

class TestPreprocessingIntegration:
    """Integration tests for preprocessing pipeline"""
    
    @pytest.fixture
    def mock_complexity_analyzer(self):
        with patch('src.api.preprocessing.ComplexityAnalyzer') as mock:
            analyzer = mock.return_value
            analyzer.analyze_complexity.return_value = {
                'complexity_score': 0.3,
                'reasoning': 'Simple query',
                'features': {'length': 10, 'entities': 0}
            }
            yield analyzer
    
    def test_preprocess_simple_query(self, mock_complexity_analyzer):
        """Test preprocessing pipeline with simple query"""
        payload = {
            "query": "What is Python?",
            "user_id": "test_user",
            "context": {"session_id": "123"}
        }
        
        response = client.post("/api/preprocess", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["complexity_score"] == 0.3
        assert data["recommended_model"] == "local"
        assert "features" in data
    
    def test_preprocess_complex_query(self, mock_complexity_analyzer):
        """Test preprocessing with complex query triggers cloud routing"""
        # Mock high complexity
        mock_complexity_analyzer.analyze_complexity.return_value = {
            'complexity_score': 0.8,
            'reasoning': 'Complex analysis required',
            'features': {'length': 200, 'entities': 5}
        }
        
        payload = {
            "query": "Analyze the socioeconomic implications of AI in healthcare",
            "user_id": "test_user"
        }
        
        response = client.post("/api/preprocess", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["complexity_score"] == 0.8
        assert data["recommended_model"] == "cloud"

class TestTemperatureControlIntegration:
    """Integration tests for temperature control system"""
    
    def test_temperature_adjustment_simple(self):
        """Test temperature is lowered for simple queries"""
        payload = {
            "query": "What is 2+2?",
            "base_temperature": 0.7,
            "complexity_score": 0.2
        }
        
        response = client.post("/api/temperature/adjust", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["adjusted_temperature"] < 0.7
        assert data["reasoning"] == "Low complexity - reduced temperature for consistency"
    
    def test_temperature_adjustment_complex(self):
        """Test temperature is maintained for complex queries"""
        payload = {
            "query": "Discuss quantum computing implications",
            "base_temperature": 0.7,
            "complexity_score": 0.9
        }
        
        response = client.post("/api/temperature/adjust", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["adjusted_temperature"] >= 0.7
        assert "creativity" in data["reasoning"]
    
    def test_temperature_validation(self):
        """Test temperature bounds validation"""
        payload = {
            "query": "test",
            "base_temperature": 2.0,  # Invalid
            "complexity_score": 0.5
        }
        
        response = client.post("/api/temperature/adjust", json=payload)
        assert response.status_code == 422