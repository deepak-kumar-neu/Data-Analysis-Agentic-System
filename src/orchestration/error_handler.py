"""
Advanced error handling with retry mechanisms, circuit breakers, and fallback strategies.

This module implements sophisticated error handling patterns for production-ready systems.
"""

import time
import functools
from typing import Any, Callable, Optional, Type, Tuple, Dict
from enum import Enum
import logging


class ErrorStrategy(Enum):
    """Error handling strategies."""
    FAIL = "fail"                          # Fail immediately
    RETRY = "retry"                        # Retry with exponential backoff
    RETRY_WITH_FALLBACK = "retry_with_fallback"  # Retry then use fallback
    CONTINUE = "continue"                  # Log error and continue


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """
    Circuit breaker pattern implementation.
    
    Prevents cascading failures by stopping requests to failing services.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: Type[Exception] = Exception
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            expected_exception: Exception type to catch
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Call function through circuit breaker.
        
        Args:
            func: Function to call
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If circuit is open or function fails
        """
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self.logger.info("Circuit breaker entering HALF_OPEN state")
            else:
                raise Exception("Circuit breaker is OPEN - service unavailable")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        return (time.time() - self.last_failure_time) >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            self.logger.info("Circuit breaker CLOSED - service recovered")
    
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            self.logger.error(
                f"Circuit breaker OPEN - {self.failure_count} failures exceeded threshold"
            )


class ErrorHandler:
    """
    Advanced error handler with multiple strategies and patterns.
    """
    
    def __init__(
        self,
        strategy: ErrorStrategy = ErrorStrategy.RETRY_WITH_FALLBACK,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        exponential_backoff: bool = True,
        backoff_factor: float = 2.0,
        circuit_breaker_threshold: int = 5
    ):
        """
        Initialize error handler.
        
        Args:
            strategy: Error handling strategy
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries (seconds)
            exponential_backoff: Use exponential backoff for retries
            backoff_factor: Multiplier for exponential backoff
            circuit_breaker_threshold: Failures before circuit breaker opens
        """
        self.strategy = strategy
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.exponential_backoff = exponential_backoff
        self.backoff_factor = backoff_factor
        
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.circuit_breaker_threshold = circuit_breaker_threshold
        
        self.error_counts: Dict[str, int] = {}
        self.recovery_counts: Dict[str, int] = {}
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def handle(
        self,
        func: Callable,
        *args,
        context: Optional[str] = None,
        fallback: Optional[Callable] = None,
        use_circuit_breaker: bool = False,
        **kwargs
    ) -> Tuple[Any, bool, Optional[Exception]]:
        """
        Handle function execution with error handling.
        
        Args:
            func: Function to execute
            *args: Function arguments
            context: Context identifier for tracking
            fallback: Fallback function if main function fails
            use_circuit_breaker: Use circuit breaker pattern
            **kwargs: Function keyword arguments
            
        Returns:
            Tuple of (result, success, error)
        """
        context = context or func.__name__
        
        # Use circuit breaker if requested
        if use_circuit_breaker:
            return self._execute_with_circuit_breaker(
                func, context, fallback, *args, **kwargs
            )
        
        # Execute based on strategy
        if self.strategy == ErrorStrategy.FAIL:
            return self._execute_fail_fast(func, context, *args, **kwargs)
        
        elif self.strategy == ErrorStrategy.RETRY:
            return self._execute_with_retry(func, context, *args, **kwargs)
        
        elif self.strategy == ErrorStrategy.RETRY_WITH_FALLBACK:
            return self._execute_with_retry_and_fallback(
                func, context, fallback, *args, **kwargs
            )
        
        elif self.strategy == ErrorStrategy.CONTINUE:
            return self._execute_continue_on_error(func, context, *args, **kwargs)
        
        else:
            return self._execute_fail_fast(func, context, *args, **kwargs)
    
    def _execute_fail_fast(
        self, func: Callable, context: str, *args, **kwargs
    ) -> Tuple[Any, bool, Optional[Exception]]:
        """Execute with fail-fast strategy."""
        try:
            result = func(*args, **kwargs)
            return result, True, None
        except Exception as e:
            self.logger.error(f"[{context}] Error: {str(e)}")
            self._increment_error_count(context)
            return None, False, e
    
    def _execute_with_retry(
        self, func: Callable, context: str, *args, **kwargs
    ) -> Tuple[Any, bool, Optional[Exception]]:
        """Execute with retry strategy."""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                result = func(*args, **kwargs)
                
                if attempt > 0:
                    self.logger.info(f"[{context}] Succeeded on retry {attempt}")
                    self._increment_recovery_count(context)
                
                return result, True, None
                
            except Exception as e:
                last_exception = e
                self._increment_error_count(context)
                
                if attempt < self.max_retries:
                    delay = self._calculate_delay(attempt)
                    self.logger.warning(
                        f"[{context}] Attempt {attempt + 1}/{self.max_retries + 1} failed: {str(e)}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    self.logger.error(
                        f"[{context}] All {self.max_retries + 1} attempts failed: {str(e)}"
                    )
        
        return None, False, last_exception
    
    def _execute_with_retry_and_fallback(
        self, func: Callable, context: str, fallback: Optional[Callable], *args, **kwargs
    ) -> Tuple[Any, bool, Optional[Exception]]:
        """Execute with retry and fallback strategy."""
        result, success, error = self._execute_with_retry(func, context, *args, **kwargs)
        
        if not success and fallback is not None:
            self.logger.info(f"[{context}] Attempting fallback strategy")
            try:
                fallback_result = fallback(*args, **kwargs)
                self.logger.info(f"[{context}] Fallback succeeded")
                return fallback_result, True, None
            except Exception as fallback_error:
                self.logger.error(f"[{context}] Fallback also failed: {str(fallback_error)}")
                return None, False, fallback_error
        
        return result, success, error
    
    def _execute_continue_on_error(
        self, func: Callable, context: str, *args, **kwargs
    ) -> Tuple[Any, bool, Optional[Exception]]:
        """Execute with continue-on-error strategy."""
        try:
            result = func(*args, **kwargs)
            return result, True, None
        except Exception as e:
            self.logger.warning(f"[{context}] Error (continuing): {str(e)}")
            self._increment_error_count(context)
            return None, False, e
    
    def _execute_with_circuit_breaker(
        self, func: Callable, context: str, fallback: Optional[Callable], *args, **kwargs
    ) -> Tuple[Any, bool, Optional[Exception]]:
        """Execute with circuit breaker pattern."""
        # Get or create circuit breaker for this context
        if context not in self.circuit_breakers:
            self.circuit_breakers[context] = CircuitBreaker(
                failure_threshold=self.circuit_breaker_threshold
            )
        
        breaker = self.circuit_breakers[context]
        
        try:
            result = breaker.call(func, *args, **kwargs)
            return result, True, None
        except Exception as e:
            self.logger.error(f"[{context}] Circuit breaker caught error: {str(e)}")
            self._increment_error_count(context)
            
            # Try fallback if available
            if fallback is not None:
                self.logger.info(f"[{context}] Attempting fallback")
                try:
                    return fallback(*args, **kwargs), True, None
                except Exception as fallback_error:
                    return None, False, fallback_error
            
            return None, False, e
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate retry delay with optional exponential backoff."""
        if self.exponential_backoff:
            return self.retry_delay * (self.backoff_factor ** attempt)
        return self.retry_delay
    
    def _increment_error_count(self, context: str):
        """Increment error count for context."""
        self.error_counts[context] = self.error_counts.get(context, 0) + 1
    
    def _increment_recovery_count(self, context: str):
        """Increment recovery count for context."""
        self.recovery_counts[context] = self.recovery_counts.get(context, 0) + 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get error handling statistics."""
        return {
            "error_counts": self.error_counts.copy(),
            "recovery_counts": self.recovery_counts.copy(),
            "circuit_breakers": {
                context: breaker.state.value
                for context, breaker in self.circuit_breakers.items()
            }
        }
    
    def reset_stats(self):
        """Reset all statistics."""
        self.error_counts.clear()
        self.recovery_counts.clear()


def with_error_handling(
    max_retries: int = 3,
    retry_delay: float = 2.0,
    fallback: Optional[Callable] = None
):
    """
    Decorator for adding error handling to functions.
    
    Args:
        max_retries: Maximum retry attempts
        retry_delay: Delay between retries
        fallback: Fallback function
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            handler = ErrorHandler(
                strategy=ErrorStrategy.RETRY_WITH_FALLBACK,
                max_retries=max_retries,
                retry_delay=retry_delay
            )
            result, success, error = handler.handle(func, *args, fallback=fallback, **kwargs)
            
            if not success and error:
                raise error
            
            return result
        
        return wrapper
    return decorator
