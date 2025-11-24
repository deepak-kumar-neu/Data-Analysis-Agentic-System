"""
Statistical Analysis Tool
Performs comprehensive statistical analysis including descriptive stats,
correlation analysis, hypothesis testing, and trend detection.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging
from scipy import stats

from .base_tool import BaseCustomTool
from ..utils.helpers import get_timestamp
from pydantic import BaseModel, Field
from typing import Type


class StatisticalAnalysisInput(BaseModel):
    """Input schema for Statistical Analysis Tool."""
    
    data_path: str = Field(description="Path to the data file to analyze")
    analyses: List[str] = Field(
        default=['descriptive', 'correlation'],
        description="List of analyses to perform"
    )


class StatisticalAnalysisTool(BaseCustomTool):
    """
    Advanced statistical analysis tool providing descriptive statistics,
    correlation analysis, hypothesis testing, distribution analysis, and trend detection.
    """
    
    name: str = "Statistical Analysis Tool"
    description: str = """Comprehensive statistical analysis including descriptive stats, 
    correlation, hypothesis testing, distribution fitting, and trend analysis."""
    args_schema: Type[BaseModel] = StatisticalAnalysisInput
    
    def execute(self, data_path: str, analyses: List[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Execute statistical analysis.
        
        Args:
            data_path: Path to data file
            analyses: List of analyses to perform
                      ['descriptive', 'correlation', 'distribution', 'hypothesis', 'trends']
            **kwargs: Additional parameters (target_column, group_by, etc.)
            
        Returns:
            Dict with analysis results and insights
        """
        # Load data
        data = pd.read_csv(data_path)
        
        if analyses is None:
            analyses = ['descriptive', 'correlation']
        
        target_column = kwargs.get('target_column')
        group_by = kwargs.get('group_by')
        confidence_level = kwargs.get('confidence_level', 0.95)
        
        results = {
            'timestamp': get_timestamp(),
            'data_shape': data.shape,
            'analyses_performed': [],
            'insights': [],
            'warnings': []
        }
        
        try:
            # 1. Descriptive Statistics
            if 'descriptive' in analyses:
                desc_results = self._descriptive_analysis(
                    data, target_column, group_by
                )
                results['descriptive_statistics'] = desc_results
                results['analyses_performed'].append('descriptive')
                results['insights'].extend(desc_results.get('insights', []))
            
            # 2. Correlation Analysis
            if 'correlation' in analyses:
                corr_method = kwargs.get('correlation_method', 'pearson')
                corr_results = self._correlation_analysis(data, corr_method)
                results['correlation_analysis'] = corr_results
                results['analyses_performed'].append('correlation')
                results['insights'].extend(corr_results.get('insights', []))
            
            # 3. Distribution Analysis
            if 'distribution' in analyses:
                dist_results = self._distribution_analysis(data, target_column)
                results['distribution_analysis'] = dist_results
                results['analyses_performed'].append('distribution')
                results['insights'].extend(dist_results.get('insights', []))
            
            # 4. Hypothesis Testing
            if 'hypothesis' in analyses and target_column and group_by:
                hyp_results = self._hypothesis_testing(
                    data, target_column, group_by, confidence_level
                )
                results['hypothesis_testing'] = hyp_results
                results['analyses_performed'].append('hypothesis')
                results['insights'].extend(hyp_results.get('insights', []))
            
            # 5. Trend Analysis
            if 'trends' in analyses:
                trend_results = self._trend_analysis(data, target_column)
                results['trend_analysis'] = trend_results
                results['analyses_performed'].append('trends')
                results['insights'].extend(trend_results.get('insights', []))
            
            # 6. Generate summary
            results['summary'] = self._generate_summary(results)
            
            return {
                'success': True,
                'results': results,
                'metadata': {
                    'tool': self.name,
                    'total_analyses': len(results['analyses_performed']),
                    'total_insights': len(results['insights'])
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error during statistical analysis: {str(e)}")
            raise
    
    def _descriptive_analysis(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        group_by: Optional[str] = None
    ) -> Dict[str, Any]:
        """Perform comprehensive descriptive statistical analysis."""
        results = {
            'operation': 'descriptive_statistics',
            'insights': []
        }
        
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if target_column:
            if target_column not in numeric_cols:
                results['error'] = f"Target column '{target_column}' is not numeric"
                return results
            numeric_cols = [target_column]
        
        # Basic statistics
        desc_stats = df[numeric_cols].describe().to_dict()
        results['basic_statistics'] = desc_stats
        
        # Additional statistics
        for col in numeric_cols:
            col_stats = {
                'column': col,
                'count': int(df[col].count()),
                'missing': int(df[col].isnull().sum()),
                'mean': float(df[col].mean()),
                'median': float(df[col].median()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'range': float(df[col].max() - df[col].min()),
                'variance': float(df[col].var()),
                'skewness': float(df[col].skew()),
                'kurtosis': float(df[col].kurtosis()),
                'cv': float(df[col].std() / df[col].mean()) if df[col].mean() != 0 else 0
            }
            
            # Quartiles and IQR
            col_stats['q1'] = float(df[col].quantile(0.25))
            col_stats['q3'] = float(df[col].quantile(0.75))
            col_stats['iqr'] = col_stats['q3'] - col_stats['q1']
            
            # Mode
            mode_vals = df[col].mode()
            col_stats['mode'] = float(mode_vals.iloc[0]) if not mode_vals.empty else None
            
            results[f'statistics_{col}'] = col_stats
            
            # Generate insights
            if abs(col_stats['skewness']) > 1:
                direction = "right" if col_stats['skewness'] > 0 else "left"
                results['insights'].append(
                    f"{col}: Highly skewed distribution ({direction}-skewed, skewness={col_stats['skewness']:.2f})"
                )
            
            if col_stats['cv'] > 1:
                results['insights'].append(
                    f"{col}: High variability (CV={col_stats['cv']:.2f})"
                )
            
            if col_stats['kurtosis'] > 3:
                results['insights'].append(
                    f"{col}: Heavy-tailed distribution (kurtosis={col_stats['kurtosis']:.2f})"
                )
        
        # Grouped analysis
        if group_by and group_by in df.columns:
            grouped_stats = df.groupby(group_by)[numeric_cols].agg([
                'count', 'mean', 'median', 'std', 'min', 'max'
            ]).to_dict()
            results['grouped_statistics'] = grouped_stats
            
            # Group insights
            for col in numeric_cols:
                group_means = df.groupby(group_by)[col].mean()
                max_group = group_means.idxmax()
                min_group = group_means.idxmin()
                results['insights'].append(
                    f"{col}: Highest mean in group '{max_group}', lowest in '{min_group}'"
                )
        
        return results
    
    def _correlation_analysis(
        self,
        df: pd.DataFrame,
        method: str = 'pearson'
    ) -> Dict[str, Any]:
        """Perform correlation analysis."""
        results = {
            'operation': 'correlation_analysis',
            'method': method,
            'insights': []
        }
        
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            results['error'] = "Need at least 2 numeric columns for correlation analysis"
            return results
        
        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr(method=method)
        results['correlation_matrix'] = corr_matrix.to_dict()
        
        # Find strong correlations
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                corr_val = corr_matrix.iloc[i, j]
                
                if abs(corr_val) > 0.7:
                    strong_correlations.append({
                        'variable1': col1,
                        'variable2': col2,
                        'correlation': float(corr_val),
                        'strength': 'strong',
                        'direction': 'positive' if corr_val > 0 else 'negative'
                    })
                    
                    results['insights'].append(
                        f"Strong {('positive' if corr_val > 0 else 'negative')} "
                        f"correlation between {col1} and {col2} (r={corr_val:.3f})"
                    )
                elif abs(corr_val) > 0.5:
                    strong_correlations.append({
                        'variable1': col1,
                        'variable2': col2,
                        'correlation': float(corr_val),
                        'strength': 'moderate',
                        'direction': 'positive' if corr_val > 0 else 'negative'
                    })
        
        results['strong_correlations'] = strong_correlations
        
        # Eigenvalue analysis for multicollinearity
        eigenvalues = np.linalg.eigvals(corr_matrix)
        condition_number = max(abs(eigenvalues)) / min(abs(eigenvalues))
        
        if condition_number > 30:
            results['insights'].append(
                f"Warning: Potential multicollinearity detected (condition number={condition_number:.1f})"
            )
        
        results['condition_number'] = float(condition_number)
        
        return results
    
    def _distribution_analysis(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze data distributions and fit theoretical distributions."""
        results = {
            'operation': 'distribution_analysis',
            'insights': []
        }
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if target_column:
            if target_column not in numeric_cols:
                results['error'] = f"Target column '{target_column}' is not numeric"
                return results
            numeric_cols = [target_column]
        
        for col in numeric_cols:
            col_data = df[col].dropna()
            
            if len(col_data) < 3:
                continue
            
            dist_info = {
                'column': col,
                'sample_size': len(col_data)
            }
            
            # Normality test (Shapiro-Wilk for small samples, Anderson-Darling for large)
            if len(col_data) < 5000:
                stat, p_value = stats.shapiro(col_data)
                dist_info['normality_test'] = 'shapiro-wilk'
            else:
                result = stats.anderson(col_data, dist='norm')
                stat = result.statistic
                p_value = 0.05 if stat > result.critical_values[2] else 0.1
                dist_info['normality_test'] = 'anderson-darling'
            
            dist_info['normality_statistic'] = float(stat)
            dist_info['normality_p_value'] = float(p_value)
            dist_info['is_normal'] = p_value > 0.05
            
            if not dist_info['is_normal']:
                results['insights'].append(
                    f"{col}: Data is not normally distributed (p={p_value:.4f})"
                )
            
            # Fit common distributions
            distributions_to_try = ['norm', 'lognorm', 'expon', 'gamma']
            best_fit = {'name': None, 'ks_statistic': float('inf')}
            
            for dist_name in distributions_to_try:
                try:
                    dist = getattr(stats, dist_name)
                    params = dist.fit(col_data)
                    ks_stat, ks_p = stats.kstest(col_data, dist_name, args=params)
                    
                    if ks_stat < best_fit['ks_statistic']:
                        best_fit = {
                            'name': dist_name,
                            'parameters': params,
                            'ks_statistic': float(ks_stat),
                            'ks_p_value': float(ks_p)
                        }
                except:
                    continue
            
            dist_info['best_fit_distribution'] = best_fit
            
            if best_fit['name']:
                results['insights'].append(
                    f"{col}: Best fit distribution is {best_fit['name']} "
                    f"(KS statistic={best_fit['ks_statistic']:.4f})"
                )
            
            results[f'distribution_{col}'] = dist_info
        
        return results
    
    def _hypothesis_testing(
        self,
        df: pd.DataFrame,
        target_column: str,
        group_by: str,
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """Perform hypothesis testing between groups."""
        results = {
            'operation': 'hypothesis_testing',
            'target': target_column,
            'groups': group_by,
            'confidence_level': confidence_level,
            'insights': []
        }
        
        # Get groups
        groups = [group[target_column].dropna() for name, group in df.groupby(group_by)]
        group_names = df[group_by].unique().tolist()
        
        if len(groups) < 2:
            results['error'] = "Need at least 2 groups for hypothesis testing"
            return results
        
        # Test for equal variances (Levene's test)
        levene_stat, levene_p = stats.levene(*groups)
        results['levene_test'] = {
            'statistic': float(levene_stat),
            'p_value': float(levene_p),
            'equal_variances': levene_p > 0.05
        }
        
        # Choose appropriate test
        if len(groups) == 2:
            # Two-sample t-test or Mann-Whitney U
            # Check normality
            normal = all(stats.shapiro(g)[1] > 0.05 for g in groups if len(g) >= 3)
            
            if normal and results['levene_test']['equal_variances']:
                # Independent t-test
                stat, p_value = stats.ttest_ind(groups[0], groups[1])
                test_name = 'independent_t_test'
            elif normal:
                # Welch's t-test (unequal variances)
                stat, p_value = stats.ttest_ind(groups[0], groups[1], equal_var=False)
                test_name = 'welch_t_test'
            else:
                # Mann-Whitney U (non-parametric)
                stat, p_value = stats.mannwhitneyu(groups[0], groups[1])
                test_name = 'mann_whitney_u'
            
            results['test'] = {
                'name': test_name,
                'statistic': float(stat),
                'p_value': float(p_value),
                'significant': p_value < (1 - confidence_level),
                'group1_mean': float(groups[0].mean()),
                'group2_mean': float(groups[1].mean()),
                'effect_size': float(abs(groups[0].mean() - groups[1].mean()) / 
                                   np.sqrt((groups[0].var() + groups[1].var()) / 2))
            }
            
            if results['test']['significant']:
                results['insights'].append(
                    f"Significant difference between groups (p={p_value:.4f})"
                )
        else:
            # Multiple groups - ANOVA or Kruskal-Wallis
            normal = all(stats.shapiro(g)[1] > 0.05 for g in groups if len(g) >= 3)
            
            if normal and results['levene_test']['equal_variances']:
                # One-way ANOVA
                stat, p_value = stats.f_oneway(*groups)
                test_name = 'one_way_anova'
            else:
                # Kruskal-Wallis (non-parametric)
                stat, p_value = stats.kruskal(*groups)
                test_name = 'kruskal_wallis'
            
            results['test'] = {
                'name': test_name,
                'statistic': float(stat),
                'p_value': float(p_value),
                'significant': p_value < (1 - confidence_level),
                'group_means': {str(name): float(group.mean()) 
                              for name, group in zip(group_names, groups)}
            }
            
            if results['test']['significant']:
                results['insights'].append(
                    f"Significant difference among groups (p={p_value:.4f})"
                )
                # Post-hoc: find which groups differ
                for i in range(len(groups)):
                    for j in range(i+1, len(groups)):
                        _, pairwise_p = stats.mannwhitneyu(groups[i], groups[j])
                        if pairwise_p < (1 - confidence_level):
                            results['insights'].append(
                                f"Groups '{group_names[i]}' and '{group_names[j]}' "
                                f"differ significantly (p={pairwise_p:.4f})"
                            )
        
        return results
    
    def _trend_analysis(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """Detect trends and patterns in time series or sequential data."""
        results = {
            'operation': 'trend_analysis',
            'insights': []
        }
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if target_column:
            if target_column not in numeric_cols:
                results['error'] = f"Target column '{target_column}' is not numeric"
                return results
            numeric_cols = [target_column]
        
        for col in numeric_cols:
            col_data = df[col].dropna()
            
            if len(col_data) < 3:
                continue
            
            trend_info = {'column': col}
            
            # Linear regression for trend
            x = np.arange(len(col_data))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, col_data)
            
            trend_info['linear_trend'] = {
                'slope': float(slope),
                'intercept': float(intercept),
                'r_squared': float(r_value ** 2),
                'p_value': float(p_value),
                'significant': p_value < 0.05
            }
            
            if trend_info['linear_trend']['significant']:
                direction = "increasing" if slope > 0 else "decreasing"
                results['insights'].append(
                    f"{col}: Significant {direction} trend "
                    f"(slope={slope:.4f}, R²={r_value**2:.4f})"
                )
            
            # Moving average
            window = min(7, len(col_data) // 3)
            if window >= 3:
                ma = col_data.rolling(window=window).mean()
                trend_info['moving_average_window'] = window
                trend_info['moving_average_values'] = ma.tolist()
            
            # Detect change points (simple method)
            mean_first_half = col_data[:len(col_data)//2].mean()
            mean_second_half = col_data[len(col_data)//2:].mean()
            change_magnitude = abs(mean_second_half - mean_first_half)
            change_percent = (change_magnitude / mean_first_half * 100) if mean_first_half != 0 else 0
            
            if change_percent > 20:
                direction = "increase" if mean_second_half > mean_first_half else "decrease"
                results['insights'].append(
                    f"{col}: Significant {direction} in second half "
                    f"({change_percent:.1f}% change)"
                )
            
            results[f'trend_{col}'] = trend_info
        
        return results
    
    def _generate_summary(self, results: Dict[str, Any]) -> str:
        """Generate human-readable summary of all analyses."""
        summary_parts = [
            f"Statistical analysis completed at {results['timestamp']}.",
            f"Dataset shape: {results['data_shape']}.",
            f"Analyses performed: {', '.join(results['analyses_performed'])}.",
            f"Total insights generated: {len(results['insights'])}."
        ]
        
        if results['insights']:
            summary_parts.append(f"Key findings: {results['insights'][0]}")
        
        return " ".join(summary_parts)
    
    def validate_inputs(self, **kwargs) -> bool:
        """Validate tool inputs."""
        data = kwargs.get('data')
        if data is None or not isinstance(data, pd.DataFrame):
            return False
        
        valid_analyses = ['descriptive', 'correlation', 'distribution', 'hypothesis', 'trends']
        analyses = kwargs.get('analyses', ['descriptive'])
        if not all(a in valid_analyses for a in analyses):
            return False
        
        correlation_method = kwargs.get('correlation_method', 'pearson')
        if correlation_method not in ['pearson', 'spearman', 'kendall']:
            return False
        
        return True
