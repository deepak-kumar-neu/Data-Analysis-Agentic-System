"""
Data Retrieval Tool - Fetches data from various sources.

This tool handles data retrieval from files, APIs, databases, and URLs
with comprehensive validation and error handling.
"""

import pandas as pd
import requests
from pathlib import Path
from typing import Dict, Any, Type
from pydantic import BaseModel, Field

from src.tools.base_tool import BaseCustomTool
from src.utils.validators import validate_file_exists, validate_file_format, ValidationError


class DataRetrievalInput(BaseModel):
    """Input schema for Data Retrieval Tool."""
    
    source_path: str = Field(
        description="Path to data file, URL, or database connection string"
    )
    source_type: str = Field(
        default="file",
        description="Type of source: 'file', 'api', 'url', 'database'"
    )
    file_format: str = Field(
        default="csv",
        description="File format: 'csv', 'xlsx', 'json', 'parquet', 'sql'"
    )
    options: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional options for data loading (e.g., delimiter, sheet_name)"
    )


class DataRetrievalTool(BaseCustomTool):
    """
    Tool for retrieving data from various sources.
    
    Supports:
    - Local files (CSV, Excel, JSON, Parquet)
    - URLs (for downloading data)
    - APIs (REST endpoints)
    - Databases (SQL queries)
    """
    
    name: str = "Data Retrieval Tool"
    description: str = """Retrieve data from various sources including local files (CSV, Excel, 
    JSON, Parquet), URLs, APIs, and databases. Validates the data and returns basic information 
    about the retrieved dataset."""
    args_schema: Type[BaseModel] = DataRetrievalInput
    
    def execute(
        self,
        source_path: str,
        source_type: str = "file",
        file_format: str = "csv",
        options: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Retrieve data from the specified source.
        
        Args:
            source_path: Path to data source
            source_type: Type of source (file, api, url, database)
            file_format: Format of the file
            options: Additional options
            
        Returns:
            Dictionary with retrieval results and data info
        """
        if options is None:
            options = {}
        
        self.logger.info(f"Retrieving data from {source_type}: {source_path}")
        
        # Route to appropriate retrieval method
        if source_type == "file":
            df = self._retrieve_from_file(source_path, file_format, options)
        elif source_type == "url":
            df = self._retrieve_from_url(source_path, file_format, options)
        elif source_type == "api":
            df = self._retrieve_from_api(source_path, options)
        elif source_type == "database":
            df = self._retrieve_from_database(source_path, options)
        else:
            raise ValidationError(f"Unsupported source type: {source_type}")
        
        # Save to temporary location for other tools
        output_path = self._save_data(df, source_path, file_format)
        
        # Generate data summary
        summary = self._generate_summary(df)
        
        return {
            "source_path": source_path,
            "source_type": source_type,
            "output_path": output_path,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "column_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
            "summary": summary
        }
    
    def _retrieve_from_file(self, file_path: str, file_format: str, options: Dict) -> pd.DataFrame:
        """Retrieve data from local file."""
        # Validate file exists
        validate_file_exists(file_path)
        
        # Validate format
        allowed_formats = ['csv', 'xlsx', 'xls', 'json', 'parquet']
        validate_file_format(file_path, allowed_formats)
        
        self.logger.info(f"Loading {file_format} file: {file_path}")
        
        try:
            if file_format == 'csv':
                df = pd.read_csv(file_path, **options)
            elif file_format in ['xlsx', 'xls']:
                df = pd.read_excel(file_path, **options)
            elif file_format == 'json':
                df = pd.read_json(file_path, **options)
            elif file_format == 'parquet':
                df = pd.read_parquet(file_path, **options)
            else:
                raise ValidationError(f"Unsupported file format: {file_format}")
            
            self.logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
            return df
            
        except Exception as e:
            raise ValidationError(f"Failed to load file: {str(e)}")
    
    def _retrieve_from_url(self, url: str, file_format: str, options: Dict) -> pd.DataFrame:
        """Retrieve data from URL."""
        self.logger.info(f"Downloading data from URL: {url}")
        
        try:
            # Download data
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Save temporarily
            temp_file = Path("data/cache") / f"downloaded_{file_format}"
            temp_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(temp_file, 'wb') as f:
                f.write(response.content)
            
            # Load as file
            return self._retrieve_from_file(str(temp_file), file_format, options)
            
        except requests.RequestException as e:
            raise ValidationError(f"Failed to download from URL: {str(e)}")
    
    def _retrieve_from_api(self, endpoint: str, options: Dict) -> pd.DataFrame:
        """Retrieve data from API endpoint."""
        self.logger.info(f"Fetching data from API: {endpoint}")
        
        try:
            # Get API parameters
            method = options.get('method', 'GET')
            headers = options.get('headers', {})
            params = options.get('params', {})
            
            # Make API request
            if method.upper() == 'GET':
                response = requests.get(endpoint, headers=headers, params=params, timeout=30)
            elif method.upper() == 'POST':
                response = requests.post(endpoint, headers=headers, json=params, timeout=30)
            else:
                raise ValidationError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            
            # Parse JSON response
            data = response.json()
            
            # Convert to DataFrame
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                # Look for common data keys
                for key in ['data', 'results', 'records', 'items']:
                    if key in data and isinstance(data[key], list):
                        df = pd.DataFrame(data[key])
                        break
                else:
                    df = pd.DataFrame([data])
            else:
                raise ValidationError("Unexpected API response format")
            
            return df
            
        except requests.RequestException as e:
            raise ValidationError(f"API request failed: {str(e)}")
        except ValueError as e:
            raise ValidationError(f"Failed to parse API response: {str(e)}")
    
    def _retrieve_from_database(self, connection_string: str, options: Dict) -> pd.DataFrame:
        """Retrieve data from database."""
        self.logger.info("Fetching data from database")
        
        try:
            import sqlalchemy
            
            # Get query
            query = options.get('query')
            if not query:
                raise ValidationError("Database query is required in options")
            
            # Create engine
            engine = sqlalchemy.create_engine(connection_string)
            
            # Execute query
            df = pd.read_sql_query(query, engine)
            
            self.logger.info(f"Retrieved {len(df)} rows from database")
            return df
            
        except ImportError:
            raise ValidationError("SQLAlchemy is required for database access")
        except Exception as e:
            raise ValidationError(f"Database query failed: {str(e)}")
    
    def _save_data(self, df: pd.DataFrame, source_path: str, file_format: str) -> str:
        """Save retrieved data to cache."""
        cache_dir = Path("data/cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate output filename
        import hashlib
        import time
        
        source_hash = hashlib.md5(source_path.encode()).hexdigest()[:8]
        timestamp = int(time.time())
        output_path = cache_dir / f"retrieved_{source_hash}_{timestamp}.csv"
        
        # Save as CSV for easy access by other tools
        df.to_csv(output_path, index=False)
        
        self.logger.info(f"Data saved to: {output_path}")
        return str(output_path)
    
    def _generate_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate data summary."""
        summary = {
            "shape": list(df.shape),
            "missing_values": df.isnull().sum().to_dict(),
            "missing_percentage": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
            "duplicate_rows": int(df.duplicated().sum()),
        }
        
        # Add numeric column statistics
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            summary["numeric_summary"] = df[numeric_cols].describe().to_dict()
        
        # Add categorical column info
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            summary["categorical_summary"] = {
                col: {
                    "unique_values": int(df[col].nunique()),
                    "top_values": df[col].value_counts().head(5).to_dict()
                }
                for col in categorical_cols
            }
        
        return summary
