"""
Analysis Agent - Performs statistical analysis and discovers insights.

This agent applies various analytical techniques to discover patterns,
relationships, and generate actionable insights.
"""

from crewai import Agent, Task
from typing import Dict, Any, List

from src.agents.base_agent import BaseAgent


class AnalysisAgent(BaseAgent):
    """
    Analysis Agent responsible for statistical analysis and insight discovery.
    
    Key Responsibilities:
    - Perform descriptive and inferential statistics
    - Discover correlations and relationships
    - Detect trends and patterns
    - Generate testable hypotheses
    - Produce actionable insights
    """
    
    def create(self, **kwargs) -> Agent:
        """
        Create the Analysis Agent.
        
        Returns:
            Configured Agent instance
        """
        agent = Agent(
            role="Data Analysis Expert",
            goal="""Discover meaningful patterns, relationships, and insights in data through 
            rigorous statistical analysis, hypothesis testing, and advanced analytical techniques, 
            translating findings into actionable business intelligence.""",
            backstory="""You are a highly skilled data scientist with a Ph.D. in Statistics and 
            years of experience in applied analytics across multiple industries. You have deep 
            expertise in both descriptive and inferential statistics, and you know how to choose 
            the right analytical technique for each situation. You're proficient in correlation 
            analysis, regression modeling, hypothesis testing, time series analysis, and clustering. 
            You understand the assumptions behind statistical tests and always verify them before 
            applying methods. You're skilled at identifying spurious correlations and distinguishing 
            correlation from causation. You know how to handle different data distributions and 
            when transformations are needed. Beyond technical skills, you excel at translating 
            complex statistical findings into clear, actionable insights that non-technical 
            stakeholders can understand and act upon. You always consider statistical significance, 
            practical significance, and business relevance when presenting findings.""",
            verbose=kwargs.get('verbose', True),
            allow_delegation=False,
            memory=kwargs.get('memory', True),
            max_iter=kwargs.get('max_iter', 3)
        )
        
        self.logger.info("Analysis Agent created successfully")
        return agent
    
    def create_tasks(self, agent: Agent, context: Dict[str, Any]) -> List[Task]:
        """
        Create tasks for the Analysis Agent.
        
        Args:
            agent: The agent instance
            context: Context dictionary
                
        Returns:
            List of Task objects
        """
        analysis_objective = context.get('analysis_objective', '')
        target_column = context.get('target_column')
        business_context = context.get('business_context', '')
        
        tasks = []
        
        # Task: Comprehensive Statistical Analysis
        analysis_task = Task(
            description=f"""
            Perform comprehensive statistical analysis to discover insights.
            
            Analysis Objective: {analysis_objective}
            Business Context: {business_context}
            Target Column: {target_column if target_column else 'Not specified - perform exploratory analysis'}
            
            Your responsibilities:
            1. Descriptive Statistics:
               - Calculate comprehensive summary statistics
               - Analyze distributions for all numeric variables
               - Examine frequency distributions for categorical variables
               - Identify central tendencies and dispersions
               - Document skewness and kurtosis
            
            2. Correlation Analysis:
               - Compute correlation matrices (Pearson, Spearman)
               - Identify strong correlations (|r| > 0.7)
               - Identify moderate correlations (0.3 < |r| < 0.7)
               - Test correlation significance
               - Watch for multicollinearity
               - Distinguish correlation from causation
            
            3. Relationship Discovery:
               - Analyze relationships between variables
               - Identify potential predictive features (if target specified)
               - Examine interactions between variables
               - Look for non-linear relationships
               - Test for independence (chi-square, etc.)
            
            4. Trend Analysis (if time-based data):
               - Identify temporal trends
               - Detect seasonality
               - Analyze growth rates
               - Identify change points
               - Forecast short-term trends
            
            5. Segmentation Analysis:
               - Identify natural groupings in data
               - Compare segments on key metrics
               - Analyze within-group vs between-group variation
               - Find distinguishing characteristics
            
            6. Outlier and Anomaly Analysis:
               - Statistical identification of outliers
               - Context-based anomaly detection
               - Determine if outliers are errors or insights
               - Analyze impact of outliers on results
            
            7. Hypothesis Testing:
               - Formulate testable hypotheses
               - Select appropriate statistical tests
               - Verify test assumptions
               - Calculate p-values and confidence intervals
               - Interpret results with business context
            
            8. Predictive Insights (if target specified):
               - Identify key drivers of target variable
               - Assess feature importance
               - Build simple predictive model
               - Evaluate model performance
               - Provide predictions with confidence levels
            
            Focus on insights that directly address the analysis objective.
            Document all statistical tests and their assumptions.
            """,
            expected_output="""A comprehensive statistical analysis report containing:
            1. Executive Summary:
               - Top 5 most important findings
               - Key insights ranked by significance
               - Confidence levels for main conclusions
            
            2. Descriptive Statistics:
               - Summary statistics table for all variables
               - Distribution analysis
               - Central tendency and dispersion metrics
               - Frequency tables for categorical variables
            
            3. Correlation Analysis Results:
               - Correlation matrix with significance indicators
               - Strong correlations identified (with p-values)
               - Moderate correlations of interest
               - Interpretation of key correlations
               - Potential causal relationships flagged
            
            4. Relationship Analysis:
               - Significant relationships discovered
               - Direction and strength of relationships
               - Evidence supporting findings
               - Confidence levels
            
            5. Trend Analysis (if applicable):
               - Identified trends with statistical significance
               - Trend magnitude and direction
               - Seasonality patterns
               - Change points detected
               - Short-term forecasts
            
            6. Segmentation Findings:
               - Identified segments/clusters
               - Segment characteristics
               - Key differentiators
               - Segment sizes and distributions
               - Business relevance of segments
            
            7. Outlier/Anomaly Analysis:
               - Outliers identified by variable
               - Classification (error vs legitimate extreme)
               - Contextual explanation
               - Recommended treatment
            
            8. Hypothesis Testing Results:
               - Hypotheses tested
               - Test methods used
               - Test statistics and p-values
               - Confidence intervals
               - Conclusions (reject/fail to reject)
               - Business implications
            
            9. Predictive Insights (if applicable):
               - Key drivers of target variable
               - Feature importance rankings
               - Model performance metrics
               - Predictions with confidence intervals
               - Model limitations
            
            10. Statistical Validity:
                - Assumptions verified for each test
                - Limitations and caveats
                - Confidence in findings
                - Areas of uncertainty
            
            11. Actionable Insights:
                - Insights ranked by business impact
                - Supporting statistical evidence
                - Recommended actions
                - Expected outcomes
                - Priority levels
            
            12. Technical Details:
                - Methods used
                - Parameters and settings
                - Software/packages utilized
                - Reproducibility information""",
            agent=agent
        )
        tasks.append(analysis_task)
        
        self.logger.info(f"Created {len(tasks)} tasks for Analysis Agent")
        self.log_to_memory(f"Created {len(tasks)} analysis tasks", {
            "objective": analysis_objective,
            "target": target_column
        })
        
        return tasks
