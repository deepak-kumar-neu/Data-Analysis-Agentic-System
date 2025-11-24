"""
Utilities for Streamlit UI.
Helper functions for data handling, formatting, and visualization.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List, Optional, Tuple
import io
import base64
from datetime import datetime
import json


def load_data_file(file_path: str, file_type: str = None) -> pd.DataFrame:
    """
    Load data file into DataFrame.
    
    Args:
        file_path: Path to file
        file_type: File type (csv, excel, json, parquet)
        
    Returns:
        DataFrame
    """
    if file_type is None:
        file_type = file_path.split('.')[-1].lower()
        
    loaders = {
        'csv': pd.read_csv,
        'xlsx': pd.read_excel,
        'xls': pd.read_excel,
        'json': pd.read_json,
        'parquet': pd.read_parquet,
        'txt': lambda x: pd.read_csv(x, sep='\t')
    }
    
    loader = loaders.get(file_type, pd.read_csv)
    return loader(file_path)


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted string
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get comprehensive data summary.
    
    Args:
        df: DataFrame
        
    Returns:
        Summary dictionary
    """
    return {
        'rows': len(df),
        'columns': len(df.columns),
        'memory_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'missing_pct': (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100),
        'duplicate_rows': df.duplicated().sum(),
        'numeric_cols': len(df.select_dtypes(include=[np.number]).columns),
        'categorical_cols': len(df.select_dtypes(include=['object']).columns),
        'datetime_cols': len(df.select_dtypes(include=['datetime']).columns)
    }


def create_download_link(data: Any, filename: str, mime_type: str = "text/plain") -> str:
    """
    Create download link for data.
    
    Args:
        data: Data to download
        filename: Output filename
        mime_type: MIME type
        
    Returns:
        HTML download link
    """
    if isinstance(data, pd.DataFrame):
        data = data.to_csv(index=False)
        mime_type = "text/csv"
    elif isinstance(data, dict):
        data = json.dumps(data, indent=2)
        mime_type = "application/json"
        
    b64 = base64.b64encode(data.encode()).decode()
    return f'<a href="data:{mime_type};base64,{b64}" download="{filename}">Download {filename}</a>'


def format_execution_time(seconds: float) -> str:
    """
    Format execution time in readable format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted string
    """
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def create_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Create correlation heatmap.
    
    Args:
        df: DataFrame
        
    Returns:
        Plotly figure
    """
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        return None
        
    corr_matrix = numeric_df.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title="Correlation Heatmap",
        xaxis_title="Features",
        yaxis_title="Features",
        width=800,
        height=700
    )
    
    return fig


def create_distribution_plots(df: pd.DataFrame, max_cols: int = 6) -> List[go.Figure]:
    """
    Create distribution plots for numeric columns.
    
    Args:
        df: DataFrame
        max_cols: Maximum columns to plot
        
    Returns:
        List of Plotly figures
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns[:max_cols]
    figures = []
    
    for col in numeric_cols:
        fig = px.histogram(
            df,
            x=col,
            nbins=30,
            title=f"Distribution of {col}",
            labels={col: col, 'count': 'Frequency'},
            marginal='box'
        )
        fig.update_layout(showlegend=False)
        figures.append(fig)
        
    return figures


def create_missing_data_plot(df: pd.DataFrame) -> go.Figure:
    """
    Create missing data visualization.
    
    Args:
        df: DataFrame
        
    Returns:
        Plotly figure
    """
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0].sort_values(ascending=True)
    
    if missing_data.empty:
        return None
        
    fig = go.Figure(go.Bar(
        x=missing_data.values,
        y=missing_data.index,
        orientation='h',
        marker=dict(color='crimson')
    ))
    
    fig.update_layout(
        title="Missing Data by Column",
        xaxis_title="Missing Count",
        yaxis_title="Column",
        height=max(400, len(missing_data) * 30)
    )
    
    return fig


def create_time_series_plot(df: pd.DataFrame, date_col: str, value_col: str) -> go.Figure:
    """
    Create time series plot.
    
    Args:
        df: DataFrame
        date_col: Date column name
        value_col: Value column name
        
    Returns:
        Plotly figure
    """
    fig = px.line(
        df,
        x=date_col,
        y=value_col,
        title=f"{value_col} over Time",
        markers=True
    )
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title=value_col,
        hovermode='x unified'
    )
    
    return fig


def format_insight_html(insight: Dict[str, Any]) -> str:
    """
    Format insight as HTML.
    
    Args:
        insight: Insight dictionary
        
    Returns:
        HTML string
    """
    title = insight.get('title', 'Insight')
    description = insight.get('description', '')
    confidence = insight.get('confidence', 0)
    
    confidence_color = 'green' if confidence > 0.7 else 'orange' if confidence > 0.4 else 'red'
    
    html = f"""
    <div style='background: #f8f9fa; padding: 1.5rem; border-radius: 0.5rem; 
                margin: 1rem 0; border-left: 4px solid {confidence_color};'>
        <h4 style='margin-top: 0;'>{title}</h4>
        <p>{description}</p>
        <div style='margin-top: 1rem;'>
            <small>Confidence: {confidence*100:.1f}%</small>
            <div style='background: #e9ecef; height: 8px; border-radius: 4px; margin-top: 0.5rem;'>
                <div style='background: {confidence_color}; height: 100%; 
                           width: {confidence*100}%; border-radius: 4px;'></div>
            </div>
        </div>
    </div>
    """
    
    return html


def validate_data_file(file_path: str, max_size_mb: int = 200) -> Tuple[bool, str]:
    """
    Validate data file.
    
    Args:
        file_path: Path to file
        max_size_mb: Maximum file size in MB
        
    Returns:
        (is_valid, error_message)
    """
    import os
    
    # Check if file exists
    if not os.path.exists(file_path):
        return False, "File not found"
        
    # Check file size
    size_mb = os.path.getsize(file_path) / 1024**2
    if size_mb > max_size_mb:
        return False, f"File too large ({size_mb:.1f}MB). Maximum: {max_size_mb}MB"
        
    # Check file extension
    valid_extensions = ['.csv', '.xlsx', '.xls', '.json', '.parquet', '.txt']
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in valid_extensions:
        return False, f"Invalid file type: {ext}. Supported: {', '.join(valid_extensions)}"
        
    return True, ""


def create_metric_cards_html(metrics: Dict[str, Any]) -> str:
    """
    Create HTML for metric cards.
    
    Args:
        metrics: Metrics dictionary
        
    Returns:
        HTML string
    """
    cards_html = "<div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;'>"
    
    for name, value in metrics.items():
        cards_html += f"""
        <div style='background: white; padding: 1.5rem; border-radius: 0.5rem; 
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center;'>
            <h3 style='margin: 0; color: #667eea;'>{value}</h3>
            <p style='margin: 0.5rem 0 0 0; color: #666;'>{name}</p>
        </div>
        """
        
    cards_html += "</div>"
    return cards_html


def export_results_package(results: Dict[str, Any], output_dir: str = "./exports"):
    """
    Export complete results package.
    
    Args:
        results: Results dictionary
        output_dir: Output directory
    """
    import os
    from pathlib import Path
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    package_dir = output_path / f"results_{timestamp}"
    package_dir.mkdir(exist_ok=True)
    
    # Export JSON results
    with open(package_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
        
    # Export processed data if available
    if 'processed_data' in results and isinstance(results['processed_data'], pd.DataFrame):
        results['processed_data'].to_csv(package_dir / "data.csv", index=False)
        
    # Export report if available
    if 'report' in results:
        with open(package_dir / "report.html", 'w') as f:
            f.write(results['report'])
            
    return str(package_dir)


def create_status_badge(status: str) -> str:
    """
    Create HTML status badge.
    
    Args:
        status: Status text
        
    Returns:
        HTML string
    """
    colors = {
        'idle': '#6c757d',
        'running': '#ffc107',
        'success': '#28a745',
        'completed': '#28a745',
        'error': '#dc3545',
        'warning': '#fd7e14'
    }
    
    color = colors.get(status.lower(), '#6c757d')
    
    return f"""
    <span style='background: {color}; color: white; padding: 0.25rem 0.75rem; 
                  border-radius: 1rem; font-size: 0.875rem; font-weight: 600;'>
        {status.upper()}
    </span>
    """
