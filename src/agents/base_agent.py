"""
Base agent class providing common functionality for all agents.

This module defines the base class that all specialized agents inherit from,
providing common utilities, error handling, and memory management.
"""

from crewai import Agent
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod

from src.utils.logger import LoggerMixin
from src.orchestration.error_handler import ErrorHandler, ErrorStrategy
from src.orchestration.memory_manager import MemoryManager


class BaseAgent(ABC, LoggerMixin):
    """
    Base class for all agents in the system.
    
    Provides common functionality including:
    - Error handling
    - Memory management
    - Logging
    - Task creation
    - Performance tracking
    """
    
    def __init__(
        self,
        memory_manager: Optional[MemoryManager] = None,
        error_handler: Optional[ErrorHandler] = None
    ):
        """
        Initialize base agent.
        
        Args:
            memory_manager: Shared memory manager instance
            error_handler: Error handler instance
        """
        self.memory_manager = memory_manager
        self.error_handler = error_handler or ErrorHandler(
            strategy=ErrorStrategy.RETRY_WITH_FALLBACK,
            max_retries=3
        )
        
        # Performance metrics
        self.metrics = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_execution_time": 0.0,
            "errors_recovered": 0
        }
        
        # Agent instance
        self._agent: Optional[Agent] = None
    
    @abstractmethod
    def create(self, **kwargs) -> Agent:
        """
        Create and configure the CrewAI agent.
        
        Returns:
            Configured Agent instance
        """
        pass
    
    @abstractmethod
    def create_tasks(self, agent: Agent, context: Dict[str, Any]) -> List[Any]:
        """
        Create tasks for this agent.
        
        Args:
            agent: The agent instance
            context: Context dictionary with task parameters
            
        Returns:
            List of Task objects
        """
        pass
    
    def get_agent(self, **kwargs) -> Agent:
        """
        Get or create agent instance.
        
        Args:
            **kwargs: Additional arguments for agent creation
            
        Returns:
            Agent instance
        """
        if self._agent is None:
            self._agent = self.create(**kwargs)
        return self._agent
    
    def log_to_memory(self, message: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Log message to memory system.
        
        Args:
            message: Message to log
            metadata: Additional metadata
        """
        if self.memory_manager:
            self.memory_manager.conversation.add_message(
                agent=self.__class__.__name__,
                message=message,
                metadata=metadata
            )
    
    def get_context_from_memory(self, key: str, default: Any = None) -> Any:
        """
        Get context from memory.
        
        Args:
            key: Context key
            default: Default value if not found
            
        Returns:
            Context value
        """
        if self.memory_manager:
            return self.memory_manager.get_context(key, default)
        return default
    
    def set_context_in_memory(self, key: str, value: Any):
        """
        Set context in memory.
        
        Args:
            key: Context key
            value: Context value
        """
        if self.memory_manager:
            self.memory_manager.add_context(key, value)
    
    def record_metric(self, metric_name: str, value: float):
        """
        Record performance metric.
        
        Args:
            metric_name: Name of metric
            value: Metric value
        """
        if self.memory_manager:
            self.memory_manager.record_metric(
                f"{self.__class__.__name__}.{metric_name}",
                value
            )
    
    def increment_task_completed(self):
        """Increment completed task counter."""
        self.metrics["tasks_completed"] += 1
        self.record_metric("tasks_completed", 1)
    
    def increment_task_failed(self):
        """Increment failed task counter."""
        self.metrics["tasks_failed"] += 1
        self.record_metric("tasks_failed", 1)
    
    def record_execution_time(self, duration: float):
        """
        Record task execution time.
        
        Args:
            duration: Execution time in seconds
        """
        self.metrics["total_execution_time"] += duration
        self.record_metric("execution_time", duration)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get agent performance metrics.
        
        Returns:
            Dictionary of metrics
        """
        return self.metrics.copy()
    
    def handle_error(self, error: Exception, context: str = "") -> Dict[str, Any]:
        """
        Handle error with error handler.
        
        Args:
            error: Exception that occurred
            context: Error context
            
        Returns:
            Error information dictionary
        """
        self.logger.error(f"Error in {self.__class__.__name__}: {str(error)}")
        self.increment_task_failed()
        
        return {
            "status": "error",
            "agent": self.__class__.__name__,
            "error": str(error),
            "context": context
        }
    
    def validate_context(self, context: Dict[str, Any], required_keys: List[str]) -> bool:
        """
        Validate that required keys exist in context.
        
        Args:
            context: Context dictionary
            required_keys: List of required keys
            
        Returns:
            True if all keys present
            
        Raises:
            ValueError: If required keys missing
        """
        missing = [key for key in required_keys if key not in context]
        if missing:
            raise ValueError(f"Missing required context keys: {missing}")
        return True
    
    def __repr__(self) -> str:
        """String representation of agent."""
        return f"{self.__class__.__name__}(tasks_completed={self.metrics['tasks_completed']})"
