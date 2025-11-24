"""
Visualization Tool
Creates comprehensive data visualizations including charts, plots, and dashboards.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from .base_tool import BaseCustomTool
from ..utils.helpers import get_timestamp, ensure_directory
from pydantic import BaseModel, Field
from typing import Type


class VisualizationInput(BaseModel):
    """Input schema for Visualization Tool."""
    
    data_path: str = Field(description="Path to the data file to visualize")
    plot_type: str = Field(
        default='bar',
        description="Type of plot to create"
    )


class VisualizationTool(BaseCustomTool):
    """
    Advanced visualization tool for creating professional data visualizations
    including charts, plots, heatmaps, and multi-panel dashboards.
    """
    
    name: str = "Visualization Tool"
    description: str = """Create professional visualizations including bar charts, line plots, 
    scatter plots, heatmaps, distributions, and multi-panel dashboards."""
    args_schema: Type[BaseModel] = VisualizationInput
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
    
    def execute(self, data_path: str, plot_type: str = 'bar', **kwargs) -> Dict[str, Any]:
        """
        Execute visualization creation.
        
        Args:
            data_path: Path to data file
            plot_type: Type of plot ('bar', 'line', 'scatter', 'hist', 'box', 
                      'violin', 'heatmap', 'pair', 'dashboard')
            **kwargs: Additional parameters (x_column, y_column, output_path, etc.)
            
        Returns:
            Dict with visualization info and file path
        """
        # Load data
        data = pd.read_csv(data_path)
        
        output_path = kwargs.get('output_path', './results/visualizations')
        
        results = {
            'timestamp': get_timestamp(),
            'plot_type': plot_type,
            'visualizations_created': [],
            'insights': []
        }
        
        try:
            # Ensure output directory exists
            ensure_directory(output_path)
            
            # Create visualization based on type
            if plot_type == 'bar':
                fig_path, insights = self._create_bar_chart(data, kwargs, output_path)
            elif plot_type == 'line':
                fig_path, insights = self._create_line_plot(data, kwargs, output_path)
            elif plot_type == 'scatter':
                fig_path, insights = self._create_scatter_plot(data, kwargs, output_path)
            elif plot_type == 'hist':
                fig_path, insights = self._create_histogram(data, kwargs, output_path)
            elif plot_type == 'box':
                fig_path, insights = self._create_box_plot(data, kwargs, output_path)
            elif plot_type == 'violin':
                fig_path, insights = self._create_violin_plot(data, kwargs, output_path)
            elif plot_type == 'heatmap':
                fig_path, insights = self._create_heatmap(data, kwargs, output_path)
            elif plot_type == 'pair':
                fig_path, insights = self._create_pair_plot(data, kwargs, output_path)
            elif plot_type == 'dashboard':
                fig_path, insights = self._create_dashboard(data, kwargs, output_path)
            else:
                raise ValueError(f"Unknown plot type: {plot_type}")
            
            results['visualizations_created'].append({
                'type': plot_type,
                'path': fig_path,
                'timestamp': get_timestamp()
            })
            results['insights'].extend(insights)
            results['primary_output'] = fig_path
            
            return {
                'success': True,
                'results': results,
                'metadata': {
                    'tool': self.name,
                    'total_visualizations': len(results['visualizations_created'])
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error creating visualization: {str(e)}")
            raise
    
    def _create_bar_chart(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        output_path: str
    ) -> tuple:
        """Create a bar chart."""
        x_col = params.get('x_column')
        y_col = params.get('y_column')
        hue_col = params.get('hue_column')
        title = params.get('title', 'Bar Chart')
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if hue_col:
            data_pivot = df.pivot_table(values=y_col, index=x_col, columns=hue_col, aggfunc='mean')
            data_pivot.plot(kind='bar', ax=ax)
        else:
            df.groupby(x_col)[y_col].mean().plot(kind='bar', ax=ax)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(x_col if x_col else '', fontsize=12)
        ax.set_ylabel(y_col if y_col else '', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        filename = f"bar_chart_{get_timestamp()}.png"
        filepath = str(Path(output_path) / filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        insights = [f"Bar chart created comparing {y_col} across {x_col}"]
        
        return filepath, insights
    
    def _create_line_plot(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        output_path: str
    ) -> tuple:
        """Create a line plot."""
        x_col = params.get('x_column')
        y_col = params.get('y_column')
        hue_col = params.get('hue_column')
        title = params.get('title', 'Line Plot')
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if hue_col:
            for group in df[hue_col].unique():
                group_data = df[df[hue_col] == group]
                ax.plot(group_data[x_col], group_data[y_col], marker='o', label=group)
            ax.legend()
        else:
            ax.plot(df[x_col], df[y_col], marker='o', linewidth=2)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(x_col if x_col else '', fontsize=12)
        ax.set_ylabel(y_col if y_col else '', fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = f"line_plot_{get_timestamp()}.png"
        filepath = str(Path(output_path) / filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        insights = [f"Line plot showing trend of {y_col} over {x_col}"]
        
        return filepath, insights
    
    def _create_scatter_plot(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        output_path: str
    ) -> tuple:
        """Create a scatter plot."""
        x_col = params.get('x_column')
        y_col = params.get('y_column')
        hue_col = params.get('hue_column')
        title = params.get('title', 'Scatter Plot')
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if hue_col:
            sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, 
                          s=100, alpha=0.6, ax=ax)
        else:
            ax.scatter(df[x_col], df[y_col], s=100, alpha=0.6)
        
        # Add trend line
        z = np.polyfit(df[x_col].dropna(), df[y_col].dropna(), 1)
        p = np.poly1d(z)
        ax.plot(df[x_col], p(df[x_col]), "r--", alpha=0.8, linewidth=2, label='Trend')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(x_col if x_col else '', fontsize=12)
        ax.set_ylabel(y_col if y_col else '', fontsize=12)
        ax.legend()
        plt.tight_layout()
        
        filename = f"scatter_plot_{get_timestamp()}.png"
        filepath = str(Path(output_path) / filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calculate correlation
        corr = df[[x_col, y_col]].corr().iloc[0, 1]
        insights = [f"Scatter plot reveals correlation of {corr:.3f} between {x_col} and {y_col}"]
        
        return filepath, insights
    
    def _create_histogram(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        output_path: str
    ) -> tuple:
        """Create a histogram."""
        column = params.get('x_column') or params.get('y_column')
        hue_col = params.get('hue_column')
        title = params.get('title', 'Distribution')
        bins = params.get('bins', 30)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if hue_col:
            for group in df[hue_col].unique():
                group_data = df[df[hue_col] == group][column]
                ax.hist(group_data, bins=bins, alpha=0.6, label=group)
            ax.legend()
        else:
            ax.hist(df[column], bins=bins, edgecolor='black', alpha=0.7)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(column if column else '', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        plt.tight_layout()
        
        filename = f"histogram_{get_timestamp()}.png"
        filepath = str(Path(output_path) / filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        insights = [f"Distribution histogram shows spread of {column}"]
        
        return filepath, insights
    
    def _create_box_plot(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        output_path: str
    ) -> tuple:
        """Create a box plot."""
        x_col = params.get('x_column')
        y_col = params.get('y_column')
        title = params.get('title', 'Box Plot')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if x_col:
            sns.boxplot(data=df, x=x_col, y=y_col, ax=ax)
        else:
            sns.boxplot(data=df[[y_col]], ax=ax)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        filename = f"box_plot_{get_timestamp()}.png"
        filepath = str(Path(output_path) / filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        insights = [f"Box plot shows distribution and outliers in {y_col}"]
        
        return filepath, insights
    
    def _create_violin_plot(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        output_path: str
    ) -> tuple:
        """Create a violin plot."""
        x_col = params.get('x_column')
        y_col = params.get('y_column')
        title = params.get('title', 'Violin Plot')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.violinplot(data=df, x=x_col, y=y_col, ax=ax)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        filename = f"violin_plot_{get_timestamp()}.png"
        filepath = str(Path(output_path) / filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        insights = [f"Violin plot reveals distribution density of {y_col} across {x_col}"]
        
        return filepath, insights
    
    def _create_heatmap(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        output_path: str
    ) -> tuple:
        """Create a correlation heatmap."""
        title = params.get('title', 'Correlation Heatmap')
        
        # Select numeric columns only
        numeric_df = df.select_dtypes(include=[np.number])
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        corr = numeric_df.corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, ax=ax, cbar_kws={'shrink': 0.8})
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filename = f"heatmap_{get_timestamp()}.png"
        filepath = str(Path(output_path) / filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Find strongest correlations
        corr_pairs = []
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                if abs(corr.iloc[i, j]) > 0.7:
                    corr_pairs.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))
        
        insights = [f"Heatmap shows correlations among {len(numeric_df.columns)} variables"]
        if corr_pairs:
            insights.append(f"Found {len(corr_pairs)} strong correlations")
        
        return filepath, insights
    
    def _create_pair_plot(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        output_path: str
    ) -> tuple:
        """Create a pair plot."""
        hue_col = params.get('hue_column')
        columns = params.get('columns', df.select_dtypes(include=[np.number]).columns.tolist()[:5])
        
        if hue_col:
            pair_plot = sns.pairplot(df[columns + [hue_col]], hue=hue_col)
        else:
            pair_plot = sns.pairplot(df[columns])
        
        filename = f"pair_plot_{get_timestamp()}.png"
        filepath = str(Path(output_path) / filename)
        pair_plot.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        insights = [f"Pair plot showing relationships among {len(columns)} variables"]
        
        return filepath, insights
    
    def _create_dashboard(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        output_path: str
    ) -> tuple:
        """Create a comprehensive dashboard."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:3]
        
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. Correlation heatmap
        ax1 = fig.add_subplot(gs[0, :])
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax1)
        ax1.set_title('Correlation Matrix', fontweight='bold')
        
        # 2-4. Distribution histograms
        for idx, col in enumerate(numeric_cols):
            ax = fig.add_subplot(gs[1, idx % 2]) if idx < 2 else fig.add_subplot(gs[2, 0])
            df[col].hist(bins=30, ax=ax, edgecolor='black')
            ax.set_title(f'Distribution: {col}', fontweight='bold')
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
        
        # 5. Box plots
        ax5 = fig.add_subplot(gs[2, 1])
        df[numeric_cols].boxplot(ax=ax5)
        ax5.set_title('Box Plots', fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        
        plt.suptitle('Data Analysis Dashboard', fontsize=16, fontweight='bold', y=0.995)
        
        filename = f"dashboard_{get_timestamp()}.png"
        filepath = str(Path(output_path) / filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        insights = [f"Comprehensive dashboard created with {len(numeric_cols)} key metrics"]
        
        return filepath, insights
    
    def validate_inputs(self, **kwargs) -> bool:
        """Validate tool inputs."""
        data = kwargs.get('data')
        if data is None or not isinstance(data, pd.DataFrame):
            return False
        
        plot_type = kwargs.get('plot_type', 'bar')
        valid_types = ['bar', 'line', 'scatter', 'hist', 'box', 'violin', 'heatmap', 'pair', 'dashboard']
        if plot_type not in valid_types:
            return False
        
        return True
