"""
Input validation utilities for the Data Analysis Agentic System.
"""

import os
from pathlib import Path
from typing import Any, List, Optional, Union, Dict, Tuple
import pandas as pd


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class DataValidator:
    """Comprehensive data validator for DataFrame validation and quality checks"""
    
    def __init__(self):
        """Initialize DataValidator"""
        pass
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, min_rows: int = 1) -> bool:
        """
        Validate that DataFrame meets basic requirements.
        
        Args:
            df: DataFrame to validate
            min_rows: Minimum number of rows required
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If DataFrame is invalid
        """
        if df is None:
            raise ValidationError("DataFrame is None")
        
        if df.empty:
            raise ValidationError("DataFrame is empty")
        
        if len(df) < min_rows:
            raise ValidationError(f"DataFrame has {len(df)} rows, minimum {min_rows} required")
        
        return True
    
    @staticmethod
    def validate_columns(df: pd.DataFrame, required_columns: List[str]) -> Tuple[bool, List[str]]:
        """
        Validate that DataFrame has required columns.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            
        Returns:
            Tuple of (is_valid, missing_columns)
        """
        if df is None:
            return False, required_columns
        
        missing = [col for col in required_columns if col not in df.columns]
        return len(missing) == 0, missing
    
    @staticmethod
    def validate_numeric_columns(df: pd.DataFrame, columns: List[str]) -> Tuple[bool, List[str]]:
        """
        Validate that specified columns are numeric.
        
        Args:
            df: DataFrame to validate
            columns: List of column names that should be numeric
            
        Returns:
            Tuple of (is_valid, non_numeric_columns)
        """
        if df is None:
            return False, columns
        
        non_numeric = []
        for col in columns:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    non_numeric.append(col)
            else:
                non_numeric.append(col)
        
        return len(non_numeric) == 0, non_numeric
    
    @staticmethod
    def validate_data_types(df: pd.DataFrame, type_requirements: Dict[str, str]) -> Tuple[bool, Dict[str, str]]:
        """
        Validate column data types.
        
        Args:
            df: DataFrame to validate
            type_requirements: Dict mapping column names to expected types
                              (e.g., {'age': 'numeric', 'name': 'object'})
        
        Returns:
            Tuple of (is_valid, type_mismatches)
        """
        if df is None:
            return False, type_requirements
        
        mismatches = {}
        for col, expected_type in type_requirements.items():
            if col not in df.columns:
                mismatches[col] = f"Column not found (expected {expected_type})"
                continue
            
            actual_type = str(df[col].dtype)
            
            if expected_type == 'numeric':
                if not pd.api.types.is_numeric_dtype(df[col]):
                    mismatches[col] = f"Expected numeric, got {actual_type}"
            elif expected_type == 'datetime':
                if not pd.api.types.is_datetime64_any_dtype(df[col]):
                    mismatches[col] = f"Expected datetime, got {actual_type}"
            elif expected_type == 'categorical' or expected_type == 'object':
                if not pd.api.types.is_object_dtype(df[col]) and not pd.api.types.is_categorical_dtype(df[col]):
                    mismatches[col] = f"Expected categorical/object, got {actual_type}"
        
        return len(mismatches) == 0, mismatches
    
    def check_missing_values(self, df: pd.DataFrame, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Check for missing values and report columns exceeding threshold.
        
        Args:
            df: DataFrame to check
            threshold: Maximum acceptable proportion of missing values (0.0 to 1.0)
            
        Returns:
            Dict with missing value statistics
        """
        if df is None or df.empty:
            return {"error": "Invalid or empty DataFrame"}
        
        missing_stats = {}
        problematic_columns = []
        
        for col in df.columns:
            missing_count = df[col].isna().sum()
            missing_pct = missing_count / len(df)
            
            missing_stats[col] = {
                "count": int(missing_count),
                "percentage": round(missing_pct * 100, 2)
            }
            
            if missing_pct > threshold:
                problematic_columns.append(col)
        
        return {
            "total_rows": len(df),
            "columns_checked": len(df.columns),
            "missing_stats": missing_stats,
            "problematic_columns": problematic_columns,
            "threshold_percentage": threshold * 100
        }
    
    def check_duplicates(self, df: pd.DataFrame, subset: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Check for duplicate rows.
        
        Args:
            df: DataFrame to check
            subset: Optional list of columns to check for duplicates
            
        Returns:
            Dict with duplicate statistics
        """
        if df is None or df.empty:
            return {"error": "Invalid or empty DataFrame"}
        
        duplicate_count = df.duplicated(subset=subset).sum()
        
        return {
            "total_rows": len(df),
            "duplicate_rows": int(duplicate_count),
            "duplicate_percentage": round((duplicate_count / len(df)) * 100, 2),
            "checked_columns": subset if subset else "all"
        }
    
    def validate_value_ranges(
        self, 
        df: pd.DataFrame, 
        range_specs: Dict[str, Tuple[Optional[float], Optional[float]]]
    ) -> Dict[str, Any]:
        """
        Validate that numeric columns fall within specified ranges.
        
        Args:
            df: DataFrame to validate
            range_specs: Dict mapping column names to (min, max) tuples
                        Use None for unbounded ranges
        
        Returns:
            Dict with validation results
        """
        if df is None or df.empty:
            return {"error": "Invalid or empty DataFrame"}
        
        violations = {}
        
        for col, (min_val, max_val) in range_specs.items():
            if col not in df.columns:
                violations[col] = {"error": "Column not found"}
                continue
            
            if not pd.api.types.is_numeric_dtype(df[col]):
                violations[col] = {"error": "Column is not numeric"}
                continue
            
            col_violations = []
            
            if min_val is not None:
                below_min = (df[col] < min_val).sum()
                if below_min > 0:
                    col_violations.append(f"{below_min} values below {min_val}")
            
            if max_val is not None:
                above_max = (df[col] > max_val).sum()
                if above_max > 0:
                    col_violations.append(f"{above_max} values above {max_val}")
            
            if col_violations:
                violations[col] = {
                    "issues": col_violations,
                    "min_allowed": min_val,
                    "max_allowed": max_val,
                    "actual_min": float(df[col].min()),
                    "actual_max": float(df[col].max())
                }
        
        return {
            "columns_checked": len(range_specs),
            "violations": violations,
            "is_valid": len(violations) == 0
        }


def validate_file_exists(file_path: str) -> bool:
    """
    Validate that a file exists.
    
    Args:
        file_path: Path to file
        
    Returns:
        True if file exists
        
    Raises:
        ValidationError: If file does not exist
    """
    if not os.path.exists(file_path):
        raise ValidationError(f"File not found: {file_path}")
    
    if not os.path.isfile(file_path):
        raise ValidationError(f"Path is not a file: {file_path}")
    
    return True


def validate_file_format(file_path: str, allowed_formats: List[str]) -> bool:
    """
    Validate file format.
    
    Args:
        file_path: Path to file
        allowed_formats: List of allowed file extensions (e.g., ['csv', 'xlsx'])
        
    Returns:
        True if format is valid
        
    Raises:
        ValidationError: If format is not allowed
    """
    extension = Path(file_path).suffix.lower().lstrip('.')
    
    if extension not in allowed_formats:
        raise ValidationError(
            f"Unsupported file format: {extension}. "
            f"Allowed formats: {', '.join(allowed_formats)}"
        )
    
    return True


def validate_directory(dir_path: str, create_if_missing: bool = True) -> bool:
    """
    Validate directory exists or create it.
    
    Args:
        dir_path: Path to directory
        create_if_missing: Create directory if it doesn't exist
        
    Returns:
        True if directory exists or was created
        
    Raises:
        ValidationError: If directory doesn't exist and create_if_missing is False
    """
    path = Path(dir_path)
    
    if not path.exists():
        if create_if_missing:
            path.mkdir(parents=True, exist_ok=True)
            return True
        else:
            raise ValidationError(f"Directory not found: {dir_path}")
    
    if not path.is_dir():
        raise ValidationError(f"Path is not a directory: {dir_path}")
    
    return True


def validate_data_source(source: Union[str, pd.DataFrame]) -> bool:
    """
    Validate data source.
    
    Args:
        source: File path or DataFrame
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If source is invalid
    """
    if isinstance(source, pd.DataFrame):
        if source.empty:
            raise ValidationError("DataFrame is empty")
        return True
    
    if isinstance(source, str):
        validate_file_exists(source)
        validate_file_format(source, ['csv', 'xlsx', 'xls', 'json', 'parquet'])
        return True
    
    raise ValidationError(f"Invalid data source type: {type(source)}")


def validate_column_exists(df: pd.DataFrame, column: str) -> bool:
    """
    Validate that a column exists in DataFrame.
    
    Args:
        df: DataFrame
        column: Column name
        
    Returns:
        True if column exists
        
    Raises:
        ValidationError: If column doesn't exist
    """
    if column not in df.columns:
        raise ValidationError(
            f"Column '{column}' not found. Available columns: {list(df.columns)}"
        )
    
    return True


def validate_columns_exist(df: pd.DataFrame, columns: List[str]) -> bool:
    """
    Validate that multiple columns exist in DataFrame.
    
    Args:
        df: DataFrame
        columns: List of column names
        
    Returns:
        True if all columns exist
        
    Raises:
        ValidationError: If any column doesn't exist
    """
    missing = [col for col in columns if col not in df.columns]
    
    if missing:
        raise ValidationError(
            f"Columns not found: {missing}. Available columns: {list(df.columns)}"
        )
    
    return True


def validate_numeric_column(df: pd.DataFrame, column: str) -> bool:
    """
    Validate that a column contains numeric data.
    
    Args:
        df: DataFrame
        column: Column name
        
    Returns:
        True if column is numeric
        
    Raises:
        ValidationError: If column is not numeric
    """
    validate_column_exists(df, column)
    
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValidationError(f"Column '{column}' is not numeric (type: {df[column].dtype})")
    
    return True


def validate_string_in_list(value: str, allowed_values: List[str], name: str = "value") -> bool:
    """
    Validate that a string is in a list of allowed values.
    
    Args:
        value: Value to validate
        allowed_values: List of allowed values
        name: Name of the parameter (for error message)
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If value not in allowed list
    """
    if value not in allowed_values:
        raise ValidationError(
            f"Invalid {name}: '{value}'. Allowed values: {', '.join(allowed_values)}"
        )
    
    return True


def validate_positive_number(value: Union[int, float], name: str = "value") -> bool:
    """
    Validate that a number is positive.
    
    Args:
        value: Number to validate
        name: Name of the parameter (for error message)
        
    Returns:
        True if positive
        
    Raises:
        ValidationError: If not positive
    """
    if not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be a number, got {type(value)}")
    
    if value <= 0:
        raise ValidationError(f"{name} must be positive, got {value}")
    
    return True


def validate_range(value: Union[int, float], min_val: float, max_val: float, name: str = "value") -> bool:
    """
    Validate that a number is within a range.
    
    Args:
        value: Number to validate
        min_val: Minimum value (inclusive)
        max_val: Maximum value (inclusive)
        name: Name of the parameter (for error message)
        
    Returns:
        True if in range
        
    Raises:
        ValidationError: If not in range
    """
    if not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be a number, got {type(value)}")
    
    if value < min_val or value > max_val:
        raise ValidationError(f"{name} must be between {min_val} and {max_val}, got {value}")
    
    return True


def validate_api_key(api_key: Optional[str], key_name: str = "API key") -> bool:
    """
    Validate that an API key is provided.
    
    Args:
        api_key: API key to validate
        key_name: Name of the API key (for error message)
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If API key is missing or invalid
    """
    if not api_key:
        raise ValidationError(f"{key_name} is required but not provided")
    
    if not isinstance(api_key, str) or len(api_key) < 10:
        raise ValidationError(f"{key_name} appears to be invalid")
    
    return True


# Convenience aliases for backward compatibility
validate_file_path = validate_file_exists
validate_columns = DataValidator.validate_columns
validate_column_types = DataValidator.validate_data_types
