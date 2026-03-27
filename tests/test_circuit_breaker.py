import pytest
import time
from unittest.mock import Mock, patch
from src.core.circuit_breaker import CircuitBreaker, CircuitState
from src.core.exceptions import CircuitBreakerOpenError, ModelProviderError


class TestCircuitBreaker:
    def test_initial_state_closed(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_success_resets_failure_count(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
        cb._failure_count = 2
        
        with cb:
            pass  # Success
        
        assert cb.failure_count == 0
        assert cb.state == CircuitState.CLOSED

    def test_failures_increment_count(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
        
        for i in range(2):
            with pytest.raises(ModelProviderError):
                with cb:
                    raise ModelProviderError("Test error")
        
        assert cb.failure_count == 2
        assert cb.state == CircuitState.CLOSED

    def test_threshold_opens_circuit(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
        
        for i in range(3):
            with pytest.raises(ModelProviderError):
                with cb:
                    raise ModelProviderError("Test error")
        
        assert cb.state == CircuitState.OPEN
        assert cb.failure_count == 3

    def test_open_circuit_raises_immediately(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=60)
        
        # Trigger open state
        with pytest.raises(ModelProviderError):
            with cb:
                raise ModelProviderError("Test error")
        
        # Should now raise CircuitBreakerOpenError immediately
        with pytest.raises(CircuitBreakerOpenError):
            with cb:
                pass

    @patch('time.time')
    def test_half_open_after_timeout(self, mock_time):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=60)
        
        # Set initial time
        mock_time.return_value = 1000
        
        # Trigger open state
        with pytest.raises(ModelProviderError):
            with cb:
                raise ModelProviderError("Test error")
        
        assert cb.state == CircuitState.OPEN
        
        # Move time forward past recovery timeout
        mock_time.return_value = 1070
        
        # Should be half-open now
        assert cb.state == CircuitState.HALF_OPEN

    @patch('time.time')
    def test_half_open_success_closes_circuit(self, mock_time):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=60)
        mock_time.return_value = 1000
        
        # Open circuit
        with pytest.raises(ModelProviderError):
            with cb:
                raise ModelProviderError("Test error")
        
        # Move to half-open
        mock_time.return_value = 1070
        assert cb.state == CircuitState.HALF_OPEN
        
        # Successful call should close circuit
        with cb:
            pass
        
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    @patch('time.time')
    def test_half_open_failure_reopens_circuit(self, mock_time):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=60)
        mock_time.return_value = 1000
        
        # Open circuit
        with pytest.raises(ModelProviderError):
            with cb:
                raise ModelProviderError("Test error")
        
        # Move to half-open
        mock_time.return_value = 1070
        
        # Failed call should reopen circuit
        with pytest.raises(ModelProviderError):
            with cb:
                raise ModelProviderError("Test error")
        
        assert cb.state == CircuitState.OPEN
        assert cb.failure_count == 1