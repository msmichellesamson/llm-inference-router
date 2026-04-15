import asyncio
import pytest
from unittest.mock import AsyncMock, patch
from src.core.timeout_handler import TimeoutHandler
from src.core.exceptions import TimeoutError, ModelError


class TestTimeoutHandler:
    @pytest.fixture
    def handler(self):
        return TimeoutHandler(default_timeout=5.0)
    
    @pytest.mark.asyncio
    async def test_successful_execution(self, handler):
        async def mock_task():
            await asyncio.sleep(0.1)
            return "success"
        
        result = await handler.execute(mock_task(), timeout=1.0)
        assert result == "success"
    
    @pytest.mark.asyncio
    async def test_timeout_raises_exception(self, handler):
        async def slow_task():
            await asyncio.sleep(2.0)
            return "never reached"
        
        with pytest.raises(TimeoutError) as exc_info:
            await handler.execute(slow_task(), timeout=0.5)
        
        assert "Operation timed out after 0.5s" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_uses_default_timeout(self, handler):
        async def task():
            return "done"
        
        with patch('asyncio.wait_for') as mock_wait:
            mock_wait.return_value = "done"
            await handler.execute(task())
            mock_wait.assert_called_once()
            args = mock_wait.call_args
            assert args[1]['timeout'] == 5.0
    
    @pytest.mark.asyncio
    async def test_cancellation_cleanup(self, handler):
        cancelled = False
        
        async def cancellable_task():
            nonlocal cancelled
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancelled = True
                raise
        
        with pytest.raises(TimeoutError):
            await handler.execute(cancellable_task(), timeout=0.1)
        
        await asyncio.sleep(0.1)  # Allow cleanup
        assert cancelled
    
    @pytest.mark.asyncio
    async def test_model_error_propagation(self, handler):
        async def failing_task():
            raise ModelError("Model failed")
        
        with pytest.raises(ModelError) as exc_info:
            await handler.execute(failing_task(), timeout=1.0)
        
        assert "Model failed" in str(exc_info.value)
    
    def test_invalid_timeout_raises_error(self, handler):
        async def task():
            return "done"
        
        with pytest.raises(ValueError):
            asyncio.run(handler.execute(task(), timeout=-1.0))
