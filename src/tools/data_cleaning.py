"""
Data Cleaning Tool
Handles missing values, outliers, duplicates, and data quality issues.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Type
from datetime import datetime
import logging
from pydantic import BaseModel, Field

from src.tools.base_tool import BaseCustomTool
from src.utils.validators import DataValidator
from src.utils.helpers import get_timestamp


class DataCleaningInput(BaseModel):
    """Input schema for Data Cleaning Tool."""
    
    data_path: str = Field(description="Path to the data file to clean")
    missing_strategy: str = Field(
        default="mean",
        description="Strategy for missing values: 'drop', 'mean', 'median', 'mode', 'ffill', 'bfill'"
    )
    outlier_strategy: str = Field(
        default="clip",
        description="Strategy for outliers: 'clip', 'remove', 'transform', 'none'"
    )
    remove_duplicates: bool = Field(
        default=True,
        description="Whether to remove duplicate rows"
    )


class DataCleaningTool(BaseCustomTool):
    """
    Advanced data cleaning tool with multiple strategies for handling
    missing values, outliers, duplicates, and data quality issues.
    """
    
    name: str = "Data Cleaning Tool"
    description: str = """Comprehensive data cleaning with missing value imputation, 
    outlier detection/handling, duplicate removal, type conversion, and data quality validation."""
    args_schema: Type[BaseModel] = DataCleaningInput
    
    def execute(
        self,
        data_path: str,
        missing_strategy: str = "mean",
        outlier_strategy: str = "clip",
        remove_duplicates: bool = True
    ) -> Dict[str, Any]:
        """
        Execute data cleaning operations.
        
        Args:
            data_path: Path to data file
            missing_strategy: Strategy for missing values
            outlier_strategy: Strategy for outliers  
            remove_duplicates: Whether to remove duplicates
            
        Returns:
            Dict with cleaned data info and cleaning report
        """
        # Load data
        df = pd.read_csv(data_path)
        
        # Initialize cleaning report
        report = {
            'original_shape': df.shape,
            'original_missing': int(df.isnull().sum().sum()),
            'operations_performed': [],
            'warnings': [],
            'quality_score': 100.0,
            'timestamp': get_timestamp()
        }
        
        try:
            # 1. Handle missing values
            df, missing_report = self._handle_missing_values(df, missing_strategy)
            report['operations_performed'].append(missing_report)
            
            # 2. Handle outliers
            if outlier_strategy != 'none':
                df, outlier_report = self._handle_outliers(df, outlier_strategy, method='iqr')
                report['operations_performed'].append(outlier_report)
            
            # 3. Remove duplicates
            if remove_duplicates:
                df, duplicate_report = self._remove_duplicates(df)
                report['operations_performed'].append(duplicate_report)
            
            # 4. Convert data types
            df, type_report = self._convert_types(df)
            report['operations_performed'].append(type_report)
            
            # 5. Calculate final statistics
            report['final_shape'] = df.shape
            report['final_missing'] = int(df.isnull().sum().sum())
            report['rows_removed'] = report['original_shape'][0] - report['final_shape'][0]
            report['missing_resolved'] = report['original_missing'] - report['final_missing']
            
            # 6. Calculate quality score
            report['quality_score'] = self._calculate_quality_score(df, report)
            
            # 7. Generate summary
            report['summary'] = self._generate_cleaning_summary(report)
            
            # Save cleaned data
            cleaned_path = data_path.replace('.csv', '_cleaned.csv')
            df.to_csv(cleaned_path, index=False)
            
            return {
                'success': True,
                'cleaned_data_path': cleaned_path,
                'report': report,
                'metadata': {
                    'tool': self.name,
                    'original_rows': report['original_shape'][0],
                    'final_rows': report['final_shape'][0],
                    'data_size_mb': round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error during data cleaning: {str(e)}")
            raise
    
    def _handle_missing_values(
        self,
        df: pd.DataFrame,
        strategy: str
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Handle missing values using specified strategy."""
        original_missing = df.isnull().sum().sum()
        
        report = {
            'operation': 'missing_value_handling',
            'strategy': strategy,
            'original_missing_count': int(original_missing),
            'columns_affected': []
        }
        
        if original_missing == 0:
            report['action'] = 'none_required'
            return df, report
        
        # Get columns with missing values
        missing_cols = df.columns[df.isnull().any()].tolist()
        
        for col in missing_cols:
            missing_count = df[col].isnull().sum()
            col_report = {
                'column': col,
                'missing_count': int(missing_count),
                'missing_percent': round(missing_count / len(df) * 100, 2)
            }
            
            if strategy == 'drop':
                # Drop rows with missing values
                df = df.dropna(subset=[col])
                col_report['action'] = 'rows_dropped'
                
            elif strategy == 'mean' and pd.api.types.is_numeric_dtype(df[col]):
                # Fill with mean for numeric columns
                mean_val = df[col].mean()
                df[col].fillna(mean_val, inplace=True)
                col_report['action'] = f'filled_with_mean_{mean_val:.2f}'
                
            elif strategy == 'median' and pd.api.types.is_numeric_dtype(df[col]):
                # Fill with median for numeric columns
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                col_report['action'] = f'filled_with_median_{median_val:.2f}'
                
            elif strategy == 'mode':
                # Fill with mode (most frequent value)
                mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                df[col].fillna(mode_val, inplace=True)
                col_report['action'] = f'filled_with_mode_{mode_val}'
                
            elif strategy == 'ffill':
                # Forward fill
                df[col].fillna(method='ffill', inplace=True)
                col_report['action'] = 'forward_filled'
                
            elif strategy == 'bfill':
                # Backward fill
                df[col].fillna(method='bfill', inplace=True)
                col_report['action'] = 'backward_filled'
                
            elif strategy == 'interpolate' and pd.api.types.is_numeric_dtype(df[col]):
                # Interpolate for numeric columns
                df[col].interpolate(inplace=True)
                col_report['action'] = 'interpolated'
                
            else:
                # Default: fill with 'Unknown' for non-numeric
                df[col].fillna('Unknown', inplace=True)
                col_report['action'] = 'filled_with_unknown'
            
            report['columns_affected'].append(col_report)
        
        report['final_missing_count'] = int(df.isnull().sum().sum())
        report['resolved_count'] = int(original_missing - report['final_missing_count'])
        
        return df, report
    
    def _handle_outliers(
        self,
        df: pd.DataFrame,
        strategy: str,
        method: str
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Detect and handle outliers using specified method and strategy."""
        report = {
            'operation': 'outlier_handling',
            'method': method,
            'strategy': strategy,
            'columns_affected': []
        }
        
        # Only process numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_cols:
            if method == 'iqr':
                outliers_mask = self._detect_outliers_iqr(df[col])
            elif method == 'zscore':
                outliers_mask = self._detect_outliers_zscore(df[col])
            elif method == 'isolation_forest':
                outliers_mask = self._detect_outliers_isolation_forest(df[[col]])
            else:
                continue
            
            outlier_count = outliers_mask.sum()
            
            if outlier_count > 0:
                col_report = {
                    'column': col,
                    'outlier_count': int(outlier_count),
                    'outlier_percent': round(outlier_count / len(df) * 100, 2)
                }
                
                if strategy == 'remove':
                    df = df[~outliers_mask]
                    col_report['action'] = 'outliers_removed'
                    
                elif strategy == 'clip':
                    # Clip to IQR bounds
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    df.loc[outliers_mask, col] = df.loc[outliers_mask, col].clip(Q1, Q3)
                    col_report['action'] = f'clipped_to_IQR_[{Q1:.2f}, {Q3:.2f}]'
                    
                elif strategy == 'transform':
                    # Log transform (for positive values)
                    if df[col].min() > 0:
                        df[col] = np.log1p(df[col])
                        col_report['action'] = 'log_transformed'
                    else:
                        col_report['action'] = 'transform_skipped_negative_values'
                
                report['columns_affected'].append(col_report)
        
        report['total_outliers_detected'] = sum(
            c['outlier_count'] for c in report['columns_affected']
        )
        
        return df, report
    
    def _detect_outliers_iqr(self, series: pd.Series) -> pd.Series:
        """Detect outliers using IQR method."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (series < lower_bound) | (series > upper_bound)
    
    def _detect_outliers_zscore(self, series: pd.Series, threshold: float = 3.0) -> pd.Series:
        """Detect outliers using Z-score method."""
        z_scores = np.abs((series - series.mean()) / series.std())
        return z_scores > threshold
    
    def _detect_outliers_isolation_forest(self, df: pd.DataFrame) -> pd.Series:
        """Detect outliers using Isolation Forest."""
        try:
            from sklearn.ensemble import IsolationForest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            predictions = iso_forest.fit_predict(df)
            return pd.Series(predictions == -1, index=df.index)
        except ImportError:
            self.logger.warning("sklearn not available, using IQR method instead")
            return self._detect_outliers_iqr(df.iloc[:, 0])
    
    def _remove_duplicates(
        self,
        df: pd.DataFrame,
        subset: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Remove duplicate rows."""
        original_count = len(df)
        df = df.drop_duplicates(subset=subset, keep='first')
        duplicates_removed = original_count - len(df)
        
        report = {
            'operation': 'duplicate_removal',
            'original_count': original_count,
            'duplicates_removed': duplicates_removed,
            'duplicate_percent': round(duplicates_removed / original_count * 100, 2) if original_count > 0 else 0,
            'subset_columns': subset if subset else 'all_columns'
        }
        
        return df, report
    
    def _convert_types(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Auto-convert data types for better memory efficiency."""
        report = {
            'operation': 'type_conversion',
            'conversions': []
        }
        
        for col in df.columns:
            original_dtype = str(df[col].dtype)
            
            # Try to convert object columns to appropriate types
            if df[col].dtype == 'object':
                # Try datetime
                try:
                    df[col] = pd.to_datetime(df[col], errors='ignore')
                    if df[col].dtype != 'object':
                        report['conversions'].append({
                            'column': col,
                            'from': original_dtype,
                            'to': str(df[col].dtype)
                        })
                        continue
                except:
                    pass
                
                # Try numeric
                try:
                    numeric_col = pd.to_numeric(df[col], errors='coerce')
                    if numeric_col.notna().sum() > len(df) * 0.9:  # 90% success rate
                        df[col] = numeric_col
                        report['conversions'].append({
                            'column': col,
                            'from': original_dtype,
                            'to': str(df[col].dtype)
                        })
                        continue
                except:
                    pass
                
                # Try category for low cardinality
                if df[col].nunique() / len(df) < 0.5:  # Less than 50% unique
                    df[col] = df[col].astype('category')
                    report['conversions'].append({
                        'column': col,
                        'from': original_dtype,
                        'to': 'category'
                    })
            
            # Downcast numeric types
            elif df[col].dtype in ['int64', 'float64']:
                if df[col].dtype == 'int64':
                    df[col] = pd.to_numeric(df[col], downcast='integer')
                else:
                    df[col] = pd.to_numeric(df[col], downcast='float')
                
                if str(df[col].dtype) != original_dtype:
                    report['conversions'].append({
                        'column': col,
                        'from': original_dtype,
                        'to': str(df[col].dtype)
                    })
        
        return df, report
    
    def _apply_custom_rules(
        self,
        df: pd.DataFrame,
        rules: List[Dict[str, Any]]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply custom cleaning rules."""
        report = {
            'operation': 'custom_rules',
            'rules_applied': []
        }
        
        for rule in rules:
            rule_type = rule.get('type')
            column = rule.get('column')
            
            if rule_type == 'replace':
                # Replace values
                old_val = rule.get('old_value')
                new_val = rule.get('new_value')
                df[column] = df[column].replace(old_val, new_val)
                report['rules_applied'].append({
                    'type': 'replace',
                    'column': column,
                    'details': f'{old_val} -> {new_val}'
                })
                
            elif rule_type == 'filter':
                # Filter rows
                condition = rule.get('condition')
                original_len = len(df)
                df = df.query(condition)
                report['rules_applied'].append({
                    'type': 'filter',
                    'condition': condition,
                    'rows_removed': original_len - len(df)
                })
                
            elif rule_type == 'transform':
                # Apply transformation function
                func = rule.get('function')
                df[column] = df[column].apply(func)
                report['rules_applied'].append({
                    'type': 'transform',
                    'column': column,
                    'function': func.__name__
                })
        
        return df, report
    
    def _calculate_quality_score(self, df: pd.DataFrame, report: Dict[str, Any]) -> float:
        """Calculate overall data quality score (0-100)."""
        score = 100.0
        
        # Deduct for remaining missing values
        missing_percent = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        score -= missing_percent * 0.5
        
        # Deduct for high data loss
        if report['original_shape'][0] > 0:
            loss_percent = (report['rows_removed'] / report['original_shape'][0]) * 100
            if loss_percent > 10:
                score -= (loss_percent - 10) * 0.3
        
        # Bonus for data type optimization
        type_conversions = sum(
            1 for op in report['operations_performed']
            if op.get('operation') == 'type_conversion'
        )
        score += min(type_conversions * 2, 10)
        
        return max(0.0, min(100.0, score))
    
    def _generate_cleaning_summary(self, report: Dict[str, Any]) -> str:
        """Generate human-readable cleaning summary."""
        summary_parts = [
            f"Data cleaning completed at {report['timestamp']}.",
            f"Original shape: {report['original_shape']}, Final shape: {report['final_shape']}.",
            f"Rows removed: {report['rows_removed']}, Missing values resolved: {report['missing_resolved']}."
        ]
        
        operations = [op['operation'] for op in report['operations_performed']]
        summary_parts.append(f"Operations performed: {', '.join(operations)}.")
        summary_parts.append(f"Quality score: {report['quality_score']:.1f}/100.")
        
        return " ".join(summary_parts)
    
    def validate_inputs(self, **kwargs) -> bool:
        """Validate tool inputs."""
        data = kwargs.get('data')
        if data is None or not isinstance(data, pd.DataFrame):
            return False
        
        valid_missing_strategies = ['drop', 'mean', 'median', 'mode', 'ffill', 'bfill', 'interpolate']
        missing_strategy = kwargs.get('missing_strategy', 'mean')
        if missing_strategy not in valid_missing_strategies:
            return False
        
        valid_outlier_strategies = ['clip', 'remove', 'transform', 'none']
        outlier_strategy = kwargs.get('outlier_strategy', 'clip')
        if outlier_strategy not in valid_outlier_strategies:
            return False
        
        return True
