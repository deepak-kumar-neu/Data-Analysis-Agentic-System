"""
Report Generator Tool
Creates comprehensive analysis reports in multiple formats.
"""

import pandas as pd
import json
from typing import Dict, Any, List, Optional
import logging
from pathlib import Path
from datetime import datetime

from .base_tool import BaseCustomTool
from ..utils.helpers import get_timestamp, ensure_directory
from pydantic import BaseModel, Field
from typing import Type


class ReportGeneratorInput(BaseModel):
    """Input schema for Report Generator Tool."""
    
    report_data: Dict[str, Any] = Field(description="Report data to generate from")
    output_format: str = Field(
        default='all',
        description="Output format for report"
    )


class ReportGeneratorTool(BaseCustomTool):
    """
    Comprehensive report generation tool that creates professional analysis reports
    in multiple formats (JSON, Markdown, HTML, PDF).
    """
    
    name: str = "Report Generator Tool"
    description: str = """Generate comprehensive analysis reports in multiple formats 
    including JSON, Markdown, HTML, and PDF with visualizations."""
    args_schema: Type[BaseModel] = ReportGeneratorInput
    
    def execute(self, report_data: Dict[str, Any], output_format: str = 'all', **kwargs) -> Dict[str, Any]:
        """
        Execute report generation.
        
        Args:
            report_data: Dict containing all report sections and content
            output_format: Format for report ('json', 'markdown', 'html', 'all')
            **kwargs: Additional parameters (output_path, include_visualizations, etc.)
            
        Returns:
            Dict with report file paths and metadata
        """
        if not report_data:
            raise ValueError("Report data is required")
        
        output_path = kwargs.get('output_path', './results/reports')
        include_visualizations = kwargs.get('include_visualizations', True)
        
        results = {
            'timestamp': get_timestamp(),
            'generated_reports': [],
            'report_summary': {}
        }
        
        try:
            # Ensure output directory exists
            ensure_directory(output_path)
            
            # Generate reports in requested formats
            if output_format in ['json', 'all']:
                json_path = self._generate_json_report(report_data, output_path)
                results['generated_reports'].append({
                    'format': 'json',
                    'path': json_path
                })
            
            if output_format in ['markdown', 'all']:
                md_path = self._generate_markdown_report(
                    report_data, output_path, include_visualizations
                )
                results['generated_reports'].append({
                    'format': 'markdown',
                    'path': md_path
                })
            
            if output_format in ['html', 'all']:
                html_path = self._generate_html_report(
                    report_data, output_path, include_visualizations
                )
                results['generated_reports'].append({
                    'format': 'html',
                    'path': html_path
                })
            
            # Generate summary
            results['report_summary'] = self._generate_summary(report_data)
            results['primary_output'] = results['generated_reports'][0]['path'] if results['generated_reports'] else None
            
            return {
                'success': True,
                'results': results,
                'metadata': {
                    'tool': self.name,
                    'formats_generated': len(results['generated_reports']),
                    'report_sections': len(report_data.get('sections', []))
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            raise
    
    def _generate_json_report(self, report_data: Dict[str, Any], output_path: str) -> str:
        """Generate JSON format report."""
        timestamp = get_timestamp().replace(':', '-').replace(' ', '_')
        filename = f"analysis_report_{timestamp}.json"
        filepath = str(Path(output_path) / filename)
        
        # Ensure all data is JSON serializable
        clean_data = self._make_json_serializable(report_data)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(clean_data, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def _generate_markdown_report(
        self,
        report_data: Dict[str, Any],
        output_path: str,
        include_viz: bool
    ) -> str:
        """Generate Markdown format report."""
        timestamp = get_timestamp().replace(':', '-').replace(' ', '_')
        filename = f"analysis_report_{timestamp}.md"
        filepath = str(Path(output_path) / filename)
        
        lines = []
        
        # Header
        lines.append(f"# {report_data.get('title', 'Data Analysis Report')}")
        lines.append("")
        lines.append(f"**Generated:** {get_timestamp()}")
        lines.append("")
        
        # Executive Summary
        if 'executive_summary' in report_data:
            lines.append("## Executive Summary")
            lines.append("")
            lines.append(report_data['executive_summary'])
            lines.append("")
        
        # Metadata
        if 'metadata' in report_data:
            lines.append("## Metadata")
            lines.append("")
            for key, value in report_data['metadata'].items():
                lines.append(f"- **{key}:** {value}")
            lines.append("")
        
        # Data Overview
        if 'data_overview' in report_data:
            lines.append("## Data Overview")
            lines.append("")
            overview = report_data['data_overview']
            lines.append(f"- **Rows:** {overview.get('rows', 'N/A')}")
            lines.append(f"- **Columns:** {overview.get('columns', 'N/A')}")
            lines.append(f"- **Missing Values:** {overview.get('missing_values', 'N/A')}")
            lines.append("")
        
        # Insights
        if 'insights' in report_data:
            lines.append("## Key Insights")
            lines.append("")
            for idx, insight in enumerate(report_data['insights'], 1):
                if isinstance(insight, dict):
                    msg = insight.get('message', str(insight))
                    confidence = insight.get('confidence', 'N/A')
                    lines.append(f"{idx}. {msg} (Confidence: {confidence})")
                else:
                    lines.append(f"{idx}. {insight}")
            lines.append("")
        
        # Statistical Analysis
        if 'statistics' in report_data:
            lines.append("## Statistical Analysis")
            lines.append("")
            stats = report_data['statistics']
            if isinstance(stats, dict):
                for key, value in stats.items():
                    lines.append(f"### {key}")
                    lines.append("")
                    if isinstance(value, dict):
                        for k, v in value.items():
                            lines.append(f"- **{k}:** {v}")
                    else:
                        lines.append(str(value))
                    lines.append("")
        
        # Visualizations
        if include_viz and 'visualizations' in report_data:
            lines.append("## Visualizations")
            lines.append("")
            for viz in report_data['visualizations']:
                if isinstance(viz, dict):
                    title = viz.get('title', 'Visualization')
                    path = viz.get('path', '')
                    lines.append(f"### {title}")
                    if path:
                        lines.append(f"![{title}]({path})")
                    lines.append("")
        
        # Recommendations
        if 'recommendations' in report_data:
            lines.append("## Recommendations")
            lines.append("")
            for idx, rec in enumerate(report_data['recommendations'], 1):
                lines.append(f"{idx}. {rec}")
            lines.append("")
        
        # Conclusion
        if 'conclusion' in report_data:
            lines.append("## Conclusion")
            lines.append("")
            lines.append(report_data['conclusion'])
            lines.append("")
        
        # Footer
        lines.append("---")
        lines.append(f"*Report generated by AI Data Analysis System on {get_timestamp()}*")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        return filepath
    
    def _generate_html_report(
        self,
        report_data: Dict[str, Any],
        output_path: str,
        include_viz: bool
    ) -> str:
        """Generate HTML format report."""
        timestamp = get_timestamp().replace(':', '-').replace(' ', '_')
        filename = f"analysis_report_{timestamp}.html"
        filepath = str(Path(output_path) / filename)
        
        html = []
        html.append("<!DOCTYPE html>")
        html.append("<html lang='en'>")
        html.append("<head>")
        html.append("<meta charset='UTF-8'>")
        html.append("<meta name='viewport' content='width=device-width, initial-scale=1.0'>")
        html.append(f"<title>{report_data.get('title', 'Data Analysis Report')}</title>")
        html.append(self._get_html_styles())
        html.append("</head>")
        html.append("<body>")
        html.append("<div class='container'>")
        
        # Header
        html.append(f"<h1>{report_data.get('title', 'Data Analysis Report')}</h1>")
        html.append(f"<p class='timestamp'>Generated: {get_timestamp()}</p>")
        
        # Executive Summary
        if 'executive_summary' in report_data:
            html.append("<div class='section'>")
            html.append("<h2>Executive Summary</h2>")
            html.append(f"<p>{report_data['executive_summary']}</p>")
            html.append("</div>")
        
        # Data Overview
        if 'data_overview' in report_data:
            html.append("<div class='section'>")
            html.append("<h2>Data Overview</h2>")
            html.append("<table>")
            for key, value in report_data['data_overview'].items():
                html.append(f"<tr><td><strong>{key}</strong></td><td>{value}</td></tr>")
            html.append("</table>")
            html.append("</div>")
        
        # Insights
        if 'insights' in report_data:
            html.append("<div class='section'>")
            html.append("<h2>Key Insights</h2>")
            html.append("<ul class='insights'>")
            for insight in report_data['insights']:
                if isinstance(insight, dict):
                    msg = insight.get('message', str(insight))
                    confidence = insight.get('confidence', 'N/A')
                    html.append(f"<li>{msg} <span class='confidence'>(Confidence: {confidence})</span></li>")
                else:
                    html.append(f"<li>{insight}</li>")
            html.append("</ul>")
            html.append("</div>")
        
        # Visualizations
        if include_viz and 'visualizations' in report_data:
            html.append("<div class='section'>")
            html.append("<h2>Visualizations</h2>")
            for viz in report_data['visualizations']:
                if isinstance(viz, dict):
                    title = viz.get('title', 'Visualization')
                    path = viz.get('path', '')
                    if path:
                        html.append(f"<div class='viz'><h3>{title}</h3><img src='{path}' alt='{title}'></div>")
            html.append("</div>")
        
        # Recommendations
        if 'recommendations' in report_data:
            html.append("<div class='section'>")
            html.append("<h2>Recommendations</h2>")
            html.append("<ol>")
            for rec in report_data['recommendations']:
                html.append(f"<li>{rec}</li>")
            html.append("</ol>")
            html.append("</div>")
        
        # Footer
        html.append("<div class='footer'>")
        html.append(f"<p>Report generated by AI Data Analysis System on {get_timestamp()}</p>")
        html.append("</div>")
        
        html.append("</div>")
        html.append("</body>")
        html.append("</html>")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(html))
        
        return filepath
    
    def _get_html_styles(self) -> str:
        """Get CSS styles for HTML report."""
        return """
<style>
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        line-height: 1.6;
        color: #333;
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
        background-color: #f5f5f5;
    }
    .container {
        background-color: white;
        padding: 40px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1 {
        color: #2c3e50;
        border-bottom: 3px solid #3498db;
        padding-bottom: 10px;
    }
    h2 {
        color: #34495e;
        margin-top: 30px;
        border-left: 4px solid #3498db;
        padding-left: 10px;
    }
    .timestamp {
        color: #7f8c8d;
        font-style: italic;
    }
    .section {
        margin: 30px 0;
        padding: 20px;
        background-color: #f9f9f9;
        border-radius: 5px;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 15px 0;
    }
    td {
        padding: 10px;
        border-bottom: 1px solid #ddd;
    }
    .insights {
        list-style-type: none;
        padding-left: 0;
    }
    .insights li {
        padding: 10px;
        margin: 10px 0;
        background-color: #ecf0f1;
        border-left: 4px solid #3498db;
        border-radius: 3px;
    }
    .confidence {
        color: #7f8c8d;
        font-size: 0.9em;
    }
    .viz {
        margin: 20px 0;
        text-align: center;
    }
    .viz img {
        max-width: 100%;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .footer {
        margin-top: 50px;
        padding-top: 20px;
        border-top: 1px solid #ddd;
        text-align: center;
        color: #7f8c8d;
        font-size: 0.9em;
    }
</style>
        """
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (pd.DataFrame, pd.Series)):
            return obj.to_dict()
        elif isinstance(obj, (datetime, pd.Timestamp)):
            return str(obj)
        elif hasattr(obj, '__dict__'):
            return self._make_json_serializable(obj.__dict__)
        else:
            return obj
    
    def _generate_summary(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of the report."""
        return {
            'title': report_data.get('title', 'Data Analysis Report'),
            'timestamp': get_timestamp(),
            'sections': list(report_data.keys()),
            'total_insights': len(report_data.get('insights', [])),
            'has_visualizations': 'visualizations' in report_data,
            'has_recommendations': 'recommendations' in report_data
        }
    
    def validate_inputs(self, **kwargs) -> bool:
        """Validate tool inputs."""
        report_data = kwargs.get('report_data')
        if not report_data or not isinstance(report_data, dict):
            return False
        
        output_format = kwargs.get('output_format', 'all')
        if output_format not in ['json', 'markdown', 'html', 'all']:
            return False
        
        return True
