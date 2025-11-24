"""
Configuration management for the Data Analysis Agentic System.

This module handles loading and managing configuration from YAML files
and environment variables.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv


class Config:
    """Configuration manager for the application."""
    
    def __init__(self, config_path: Optional[str] = None, env: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to configuration file. If None, uses default.yaml
            env: Environment name (development, staging, production)
        """
        # Load environment variables
        load_dotenv()
        
        # Determine config path
        if config_path is None:
            base_dir = Path(__file__).parent.parent.parent
            config_dir = base_dir / "configs"
            
            # Try environment-specific config first
            if env:
                env_config = config_dir / f"{env}.yaml"
                if env_config.exists():
                    config_path = str(env_config)
            
            # Fall back to default
            if config_path is None:
                config_path = str(config_dir / "default.yaml")
        
        self.config_path = config_path
        self._config = self._load_config()
        self._merge_env_vars()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            print(f"Warning: Config file not found at {self.config_path}, using defaults")
            return self._get_default_config()
        except yaml.YAMLError as e:
            print(f"Error parsing config file: {e}, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "system": {
                "name": "Data Analysis Agentic System",
                "version": "1.0.0",
                "environment": "development"
            },
            "orchestration": {
                "process_type": "sequential",
                "max_iterations": 3,
                "timeout_seconds": 300,
                "enable_feedback_loops": True,
                "enable_memory": True,
                "verbose": True
            },
            "logging": {
                "level": "INFO",
                "file_logging": True,
                "console_logging": True,
                "log_file": "results/logs/system.log"
            },
            "output": {
                "directory": "results",
                "create_timestamp_folder": True,
                "save_intermediate_results": True
            }
        }
    
    def _merge_env_vars(self):
        """Merge environment variables into configuration."""
        # OpenAI Configuration
        if os.getenv("OPENAI_API_KEY"):
            if "openai" not in self._config:
                self._config["openai"] = {}
            self._config["openai"]["api_key"] = os.getenv("OPENAI_API_KEY")
            self._config["openai"]["model"] = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
            self._config["openai"]["temperature"] = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
        
        # Application Configuration
        if os.getenv("APP_ENV"):
            self._config["system"]["environment"] = os.getenv("APP_ENV")
        
        if os.getenv("LOG_LEVEL"):
            self._config["logging"]["level"] = os.getenv("LOG_LEVEL")
        
        # Output Configuration
        if os.getenv("OUTPUT_DIR"):
            self._config["output"]["directory"] = os.getenv("OUTPUT_DIR")
        
        # Feature Flags
        if os.getenv("ENABLE_WEB_SEARCH"):
            if "tools" not in self._config:
                self._config["tools"] = {}
            if "web_search" not in self._config["tools"]:
                self._config["tools"]["web_search"] = {}
            self._config["tools"]["web_search"]["enabled"] = os.getenv("ENABLE_WEB_SEARCH").lower() == "true"
        
        if os.getenv("ENABLE_PARALLEL_EXECUTION"):
            enable_parallel = os.getenv("ENABLE_PARALLEL_EXECUTION").lower() == "true"
            self._config["orchestration"]["enable_parallel"] = enable_parallel
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-notation key.
        
        Args:
            key: Dot-separated key path (e.g., 'orchestration.process_type')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """
        Set configuration value by dot-notation key.
        
        Args:
            key: Dot-separated key path
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def get_all(self) -> Dict[str, Any]:
        """Get entire configuration dictionary."""
        return self._config.copy()
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any):
        """Allow dictionary-style setting."""
        self.set(key, value)


# Global configuration instance
_config_instance: Optional[Config] = None


def get_config(config_path: Optional[str] = None, env: Optional[str] = None) -> Config:
    """
    Get or create global configuration instance.
    
    Args:
        config_path: Path to configuration file
        env: Environment name
        
    Returns:
        Config instance
    """
    global _config_instance
    
    if _config_instance is None:
        _config_instance = Config(config_path, env)
    
    return _config_instance


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
    """
    config = Config(config_path)
    return config.get_all()
