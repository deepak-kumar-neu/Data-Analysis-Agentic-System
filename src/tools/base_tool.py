"""
Base tool class providing common functionality for all tools.

This module defines the base class that all tools inherit from,
providing common utilities, validation, and error handling.
"""

from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Optional, Any, Dict
from abc import ABC, abstractmethod
import time

from src.utils.logger import LoggerMixin
from src.utils.validators import ValidationError


class BaseToolInput(BaseModel):
    """Base input model for tools."""
    pass


class BaseCustomTool(BaseTool, LoggerMixin, ABC):
    """
    Base class for all custom tools in the system.
    
    Provides common functionality including:
    - Input validation
    - Error handling
    - Performance tracking
    - Logging
    - Result formatting
    """
    
    # Metadata
    return_direct: bool = False
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
        extra = 'allow'  # Allow extra fields
    
    def __init__(self, **kwargs):
        """Initialize base tool."""
        super().__init__(**kwargs)
        # Initialize metrics as instance variable (not Pydantic field)
        if not hasattr(self, '_metrics'):
            self._metrics = {
                "execution_count": 0,
                "success_count": 0,
                "error_count": 0,
                "total_execution_time": 0.0,
                "average_execution_time": 0.0
            }
    
    @property
    def metrics(self) -> Dict[str, Any]:
        """Get metrics dictionary."""
        if not hasattr(self, '_metrics'):
            self._metrics = {
                "execution_count": 0,
                "success_count": 0,
                "error_count": 0,
                "total_execution_time": 0.0,
                "average_execution_time": 0.0
            }
        return self._metrics
    
    def _run(self, *args, **kwargs) -> str:
        """
        Execute the tool with error handling and metrics tracking.
        
        Returns:
            JSON string with results
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Executing {self.name}...")
            
            # Execute the actual tool logic
            result = self.execute(*args, **kwargs)
            
            # Track success metrics
            execution_time = time.time() - start_time
            self._record_success(execution_time)
            
            self.logger.info(f"{self.name} completed in {execution_time:.2f}s")
            
            return self._format_success_result(result, execution_time)
            
        except ValidationError as e:
            execution_time = time.time() - start_time
            self._record_error()
            self.logger.error(f"{self.name} validation error: {str(e)}")
            return self._format_error_result(str(e), "validation_error", execution_time)
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._record_error()
            self.logger.error(f"{self.name} error: {str(e)}", exc_info=True)
            return self._format_error_result(str(e), "execution_error", execution_time)
    
    async def _arun(self, *args, **kwargs) -> str:
        """Async execution (defaults to sync for now)."""
        return self._run(*args, **kwargs)
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Execute the core tool logic.
        
        Must be implemented by subclasses.
        
        Returns:
            Dictionary with tool results
        """
        pass
    
    def validate_input(self, **kwargs) -> bool:
        """
        Validate input parameters.
        
        Args:
            **kwargs: Input parameters
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If validation fails
        """
        # Subclasses can override for custom validation
        return True
    
    def _format_success_result(self, result: Dict[str, Any], execution_time: float) -> str:
        """
        Format successful result.
        
        Args:
            result: Result dictionary
            execution_time: Execution time in seconds
            
        Returns:
            JSON string
        """
        import json
        
        formatted = {
            "status": "success",
            "tool": self.name,
            "execution_time": round(execution_time, 3),
            "result": result
        }
        
        return json.dumps(formatted, indent=2, default=str)
    
    def _format_error_result(self, error: str, error_type: str, execution_time: float) -> str:
        """
        Format error result.
        
        Args:
            error: Error message
            error_type: Type of error
            execution_time: Execution time in seconds
            
        Returns:
            JSON string
        """
        import json
        
        formatted = {
            "status": "error",
            "tool": self.name,
            "error_type": error_type,
            "error_message": error,
            "execution_time": round(execution_time, 3)
        }
        
        return json.dumps(formatted, indent=2)
    
    def _record_success(self, execution_time: float):
        """Record successful execution metrics."""
        metrics = self.metrics  # Use property
        metrics["execution_count"] += 1
        metrics["success_count"] += 1
        metrics["total_execution_time"] += execution_time
        metrics["average_execution_time"] = (
            metrics["total_execution_time"] / metrics["execution_count"]
        )
    
    def _record_error(self):
        """Record error metrics."""
        metrics = self.metrics  # Use property
        metrics["execution_count"] += 1
        metrics["error_count"] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get tool performance metrics.
        
        Returns:
            Dictionary with metrics
        """
        metrics = self.metrics  # Use property
        return {
            **metrics,
            "success_rate": (
                metrics["success_count"] / metrics["execution_count"]
                if metrics["execution_count"] > 0 else 0.0
            )
        }
    
    def reset_metrics(self):
        """Reset all metrics."""
        self._metrics = {
            "execution_count": 0,
            "success_count": 0,
            "error_count": 0,
            "total_execution_time": 0.0,
            "average_execution_time": 0.0
        }
