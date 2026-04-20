"""Graceful shutdown handler for clean resource cleanup."""

import asyncio
import signal
from typing import List, Callable, Optional
from contextlib import asynccontextmanager

from .logger import get_logger
from .exceptions import ShutdownError

logger = get_logger(__name__)


class ShutdownHandler:
    """Manages graceful shutdown with proper resource cleanup."""
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.shutdown_callbacks: List[Callable] = []
        self.is_shutting_down = False
        self._shutdown_event = asyncio.Event()
    
    def register_callback(self, callback: Callable) -> None:
        """Register cleanup callback to run during shutdown."""
        self.shutdown_callbacks.append(callback)
        logger.info(f"Registered shutdown callback: {callback.__name__}")
    
    async def _signal_handler(self, signum: int) -> None:
        """Handle shutdown signals."""
        signal_name = signal.Signals(signum).name
        logger.info(f"Received {signal_name}, initiating graceful shutdown")
        await self.shutdown()
    
    async def shutdown(self) -> None:
        """Perform graceful shutdown."""
        if self.is_shutting_down:
            return
        
        self.is_shutting_down = True
        logger.info("Starting graceful shutdown...")
        
        try:
            # Run all cleanup callbacks with timeout
            await asyncio.wait_for(
                self._run_callbacks(),
                timeout=self.timeout
            )
            logger.info("Graceful shutdown completed")
        except asyncio.TimeoutError:
            logger.warning(f"Shutdown timeout after {self.timeout}s, forcing exit")
            raise ShutdownError("Shutdown timeout exceeded")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            raise ShutdownError(f"Shutdown failed: {e}")
        finally:
            self._shutdown_event.set()
    
    async def _run_callbacks(self) -> None:
        """Run all registered callbacks concurrently."""
        if not self.shutdown_callbacks:
            return
        
        tasks = []
        for callback in self.shutdown_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    task = asyncio.create_task(callback())
                else:
                    task = asyncio.create_task(asyncio.to_thread(callback))
                tasks.append(task)
            except Exception as e:
                logger.error(f"Failed to create shutdown task for {callback.__name__}: {e}")
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        for sig in [signal.SIGTERM, signal.SIGINT]:
            signal.signal(sig, lambda s, f: asyncio.create_task(self._signal_handler(s)))
    
    @asynccontextmanager
    async def lifecycle(self):
        """Context manager for application lifecycle."""
        self.setup_signal_handlers()
        try:
            yield self
        finally:
            if not self.is_shutting_down:
                await self.shutdown()
    
    async def wait_for_shutdown(self) -> None:
        """Wait for shutdown to complete."""
        await self._shutdown_event.wait()
