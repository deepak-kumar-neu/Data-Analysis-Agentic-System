"""
Helper utility functions for the Data Analysis Agentic System.
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd


def get_timestamp(format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Get current timestamp as formatted string.
    
    Args:
        format_str: strftime format string
        
    Returns:
        Formatted timestamp string
    """
    return datetime.now().strftime(format_str)


def generate_timestamp(format_str: str = "%Y%m%d_%H%M%S") -> str:
    """
    Generate timestamp string.
    
    Args:
        format_str: strftime format string
        
    Returns:
        Formatted timestamp string
    """
    return datetime.now().strftime(format_str)


def generate_hash(data: Union[str, bytes], algorithm: str = "md5") -> str:
    """
    Generate hash of data.
    
    Args:
        data: Data to hash
        algorithm: Hash algorithm (md5, sha1, sha256)
        
    Returns:
        Hex digest of hash
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    
    hash_func = getattr(hashlib, algorithm)()
    hash_func.update(data)
    return hash_func.hexdigest()


def save_json(data: Dict[str, Any], file_path: str, indent: int = 2) -> str:
    """
    Save data as JSON file.
    
    Args:
        data: Data to save
        file_path: Path to save file
        indent: JSON indentation
        
    Returns:
        Path to saved file
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=indent, default=str)
    
    return str(path)


def load_json(file_path: str) -> Dict[str, Any]:
    """
    Load data from JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Loaded data
    """
    with open(file_path, 'r') as f:
        return json.load(f)


def format_bytes(size: int) -> str:
    """
    Format bytes to human-readable string.
    
    Args:
        size: Size in bytes
        
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"


def format_size(size_bytes: int) -> str:
    """
    Format byte size to human-readable string.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def format_duration(seconds: float) -> str:
    """
    Format duration to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string (e.g., "2m 30s")
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def truncate_string(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate string to maximum length.
    
    Args:
        text: String to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated string
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    Flatten nested dictionary.
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key prefix
        sep: Separator between keys
        
    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_dataframe_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get comprehensive DataFrame information.
    
    Args:
        df: DataFrame
        
    Returns:
        Dictionary with DataFrame statistics
    """
    return {
        "shape": df.shape,
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "memory_usage": format_bytes(df.memory_usage(deep=True).sum()),
        "missing_values": df.isnull().sum().to_dict(),
        "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict(),
    }


def create_output_directory(base_dir: str = "results", use_timestamp: bool = True) -> Path:
    """
    Create output directory with optional timestamp.
    
    Args:
        base_dir: Base directory path
        use_timestamp: Whether to add timestamp to directory name
        
    Returns:
        Path to created directory
    """
    path = Path(base_dir)
    
    if use_timestamp:
        timestamp = generate_timestamp()
        path = path / timestamp
    
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_directory(directory_path: str) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        directory_path: Path to directory
        
    Returns:
        Path object
    """
    path = Path(directory_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def merge_dicts(*dicts: Dict[str, Any], deep: bool = True) -> Dict[str, Any]:
    """
    Merge multiple dictionaries.
    
    Args:
        *dicts: Dictionaries to merge
        deep: Whether to perform deep merge
        
    Returns:
        Merged dictionary
    """
    result = {}
    
    for d in dicts:
        if not deep:
            result.update(d)
        else:
            for key, value in d.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = merge_dicts(result[key], value, deep=True)
                else:
                    result[key] = value
    
    return result


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, handling division by zero.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division by zero
        
    Returns:
        Result of division or default
    """
    try:
        return numerator / denominator if denominator != 0 else default
    except (TypeError, ZeroDivisionError):
        return default


def extract_numeric(text: str) -> Optional[float]:
    """
    Extract numeric value from text.
    
    Args:
        text: Text containing number
        
    Returns:
        Extracted number or None
    """
    import re
    match = re.search(r'-?\d+\.?\d*', text)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return None
    return None
