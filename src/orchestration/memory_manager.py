"""
Sophisticated memory management system for cross-agent context sharing.

This module implements memory systems for maintaining context across agent interactions.
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
from collections import deque
import logging


class MemoryType:
    """Memory type constants."""
    CONVERSATION = "conversation"
    ENTITY = "entity"
    SUMMARY = "summary"
    EPISODIC = "episodic"


class Memory:
    """Base memory class."""
    
    def __init__(self, max_size: int = 100):
        """
        Initialize memory.
        
        Args:
            max_size: Maximum number of entries to store
        """
        self.max_size = max_size
        self.entries: deque = deque(maxlen=max_size)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def add(self, entry: Dict[str, Any]):
        """Add entry to memory."""
        entry["timestamp"] = datetime.now().isoformat()
        self.entries.append(entry)
    
    def get_recent(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get n most recent entries."""
        return list(self.entries)[-n:]
    
    def get_all(self) -> List[Dict[str, Any]]:
        """Get all entries."""
        return list(self.entries)
    
    def clear(self):
        """Clear all entries."""
        self.entries.clear()
    
    def size(self) -> int:
        """Get number of entries."""
        return len(self.entries)


class ConversationMemory(Memory):
    """
    Stores conversation history between agents.
    """
    
    def add_message(
        self,
        agent: str,
        message: str,
        role: str = "assistant",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add conversation message.
        
        Args:
            agent: Agent name
            message: Message content
            role: Role (user, assistant, system)
            metadata: Additional metadata
        """
        entry = {
            "agent": agent,
            "role": role,
            "message": message,
            "metadata": metadata or {}
        }
        self.add(entry)
    
    def get_conversation_history(self, agent: Optional[str] = None, n: int = 10) -> List[Dict[str, Any]]:
        """
        Get conversation history.
        
        Args:
            agent: Filter by agent name (None for all)
            n: Number of recent messages
            
        Returns:
            List of conversation entries
        """
        entries = self.get_recent(n)
        
        if agent:
            entries = [e for e in entries if e.get("agent") == agent]
        
        return entries
    
    def format_for_prompt(self, n: int = 10) -> str:
        """
        Format conversation history for prompt.
        
        Args:
            n: Number of recent messages
            
        Returns:
            Formatted conversation string
        """
        entries = self.get_recent(n)
        formatted = []
        
        for entry in entries:
            agent = entry.get("agent", "Unknown")
            message = entry.get("message", "")
            formatted.append(f"{agent}: {message}")
        
        return "\n".join(formatted)


class EntityMemory(Memory):
    """
    Stores information about entities (data sources, columns, metrics).
    """
    
    def __init__(self, max_size: int = 1000):
        super().__init__(max_size)
        self.entities: Dict[str, Dict[str, Any]] = {}
    
    def add_entity(self, entity_id: str, entity_type: str, properties: Dict[str, Any]):
        """
        Add or update entity.
        
        Args:
            entity_id: Unique entity identifier
            entity_type: Type of entity (data_source, column, metric, etc.)
            properties: Entity properties
        """
        self.entities[entity_id] = {
            "type": entity_type,
            "properties": properties,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
    
    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get entity by ID."""
        return self.entities.get(entity_id)
    
    def get_entities_by_type(self, entity_type: str) -> List[Dict[str, Any]]:
        """Get all entities of a specific type."""
        return [
            {"id": eid, **entity}
            for eid, entity in self.entities.items()
            if entity.get("type") == entity_type
        ]
    
    def update_entity(self, entity_id: str, properties: Dict[str, Any]):
        """Update entity properties."""
        if entity_id in self.entities:
            self.entities[entity_id]["properties"].update(properties)
            self.entities[entity_id]["updated_at"] = datetime.now().isoformat()
    
    def search_entities(self, query: str) -> List[Dict[str, Any]]:
        """Search entities by query string."""
        results = []
        query_lower = query.lower()
        
        for entity_id, entity in self.entities.items():
            if query_lower in entity_id.lower():
                results.append({"id": entity_id, **entity})
            elif query_lower in str(entity.get("properties", {})).lower():
                results.append({"id": entity_id, **entity})
        
        return results


class SummaryMemory(Memory):
    """
    Stores summarized information from analysis runs.
    """
    
    def add_summary(
        self,
        analysis_id: str,
        summary: str,
        key_findings: List[str],
        metrics: Dict[str, Any]
    ):
        """
        Add analysis summary.
        
        Args:
            analysis_id: Unique analysis identifier
            summary: Summary text
            key_findings: List of key findings
            metrics: Performance metrics
        """
        entry = {
            "analysis_id": analysis_id,
            "summary": summary,
            "key_findings": key_findings,
            "metrics": metrics
        }
        self.add(entry)
    
    def get_summary(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Get summary by analysis ID."""
        for entry in self.entries:
            if entry.get("analysis_id") == analysis_id:
                return entry
        return None


class MemoryManager:
    """
    Central memory management system coordinating different memory types.
    """
    
    def __init__(
        self,
        persist_to_disk: bool = True,
        storage_path: str = "data/cache/memory",
        max_conversation_history: int = 100,
        max_entities: int = 1000
    ):
        """
        Initialize memory manager.
        
        Args:
            persist_to_disk: Whether to persist memory to disk
            storage_path: Path for persistent storage
            max_conversation_history: Maximum conversation entries
            max_entities: Maximum number of entities
        """
        self.persist_to_disk = persist_to_disk
        self.storage_path = Path(storage_path)
        
        if self.persist_to_disk:
            self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize different memory types
        self.conversation = ConversationMemory(max_size=max_conversation_history)
        self.entity = EntityMemory(max_size=max_entities)
        self.summary = SummaryMemory(max_size=100)
        
        # Shared context for current session
        self.session_context: Dict[str, Any] = {}
        
        # Performance metrics
        self.metrics: Dict[str, List[float]] = {}
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Load persisted memory if available
        if self.persist_to_disk:
            self.load()
    
    def add_context(self, key: str, value: Any):
        """
        Add to session context.
        
        Args:
            key: Context key
            value: Context value
        """
        self.session_context[key] = value
        self.logger.debug(f"Added context: {key}")
    
    def get_context(self, key: str, default: Any = None) -> Any:
        """
        Get from session context.
        
        Args:
            key: Context key
            default: Default value if not found
            
        Returns:
            Context value or default
        """
        return self.session_context.get(key, default)
    
    def get_all_context(self) -> Dict[str, Any]:
        """Get all session context."""
        return self.session_context.copy()
    
    def clear_context(self):
        """Clear session context."""
        self.session_context.clear()
    
    def record_metric(self, metric_name: str, value: float):
        """
        Record performance metric.
        
        Args:
            metric_name: Name of metric
            value: Metric value
        """
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(value)
    
    def get_metric_stats(self, metric_name: str) -> Dict[str, float]:
        """
        Get statistics for a metric.
        
        Args:
            metric_name: Name of metric
            
        Returns:
            Dictionary with min, max, avg, count
        """
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return {}
        
        values = self.metrics[metric_name]
        return {
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "count": len(values)
        }
    
    def save(self):
        """Persist memory to disk."""
        if not self.persist_to_disk:
            return
        
        try:
            # Save conversation memory
            with open(self.storage_path / "conversation.json", 'w') as f:
                json.dump(list(self.conversation.entries), f, indent=2, default=str)
            
            # Save entity memory
            with open(self.storage_path / "entities.json", 'w') as f:
                json.dump(self.entity.entities, f, indent=2, default=str)
            
            # Save summary memory
            with open(self.storage_path / "summaries.json", 'w') as f:
                json.dump(list(self.summary.entries), f, indent=2, default=str)
            
            # Save session context
            with open(self.storage_path / "context.json", 'w') as f:
                json.dump(self.session_context, f, indent=2, default=str)
            
            # Save metrics
            with open(self.storage_path / "metrics.json", 'w') as f:
                json.dump(self.metrics, f, indent=2)
            
            self.logger.info(f"Memory persisted to {self.storage_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to persist memory: {str(e)}")
    
    def load(self):
        """Load memory from disk."""
        if not self.persist_to_disk:
            return
        
        try:
            # Load conversation memory
            conv_path = self.storage_path / "conversation.json"
            if conv_path.exists():
                with open(conv_path, 'r') as f:
                    data = json.load(f)
                    for entry in data:
                        self.conversation.entries.append(entry)
            
            # Load entity memory
            entity_path = self.storage_path / "entities.json"
            if entity_path.exists():
                with open(entity_path, 'r') as f:
                    self.entity.entities = json.load(f)
            
            # Load summary memory
            summary_path = self.storage_path / "summaries.json"
            if summary_path.exists():
                with open(summary_path, 'r') as f:
                    data = json.load(f)
                    for entry in data:
                        self.summary.entries.append(entry)
            
            # Load context
            context_path = self.storage_path / "context.json"
            if context_path.exists():
                with open(context_path, 'r') as f:
                    self.session_context = json.load(f)
            
            # Load metrics
            metrics_path = self.storage_path / "metrics.json"
            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    self.metrics = json.load(f)
            
            self.logger.info(f"Memory loaded from {self.storage_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load memory: {str(e)}")
    
    def clear_all(self):
        """Clear all memory."""
        self.conversation.clear()
        self.entity.entities.clear()
        self.summary.clear()
        self.session_context.clear()
        self.metrics.clear()
        self.logger.info("All memory cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            "conversation_entries": self.conversation.size(),
            "entity_count": len(self.entity.entities),
            "summary_count": self.summary.size(),
            "context_keys": len(self.session_context),
            "metrics_tracked": len(self.metrics)
        }
