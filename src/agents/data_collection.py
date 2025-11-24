"""
Data Collection Agent - Retrieves and validates data from various sources.

This agent is responsible for gathering high-quality data and performing
initial validation and quality assessment.
"""

from crewai import Agent, Task
from typing import Dict, Any, List

from src.agents.base_agent import BaseAgent


class DataCollectionAgent(BaseAgent):
    """
    Data Collection Agent responsible for gathering and validating data.
    
    Key Responsibilities:
    - Retrieve data from multiple sources (files, APIs, databases)
    - Validate data structure and format
    - Perform initial quality assessment
    - Handle data source errors gracefully
    - Provide data profiling information
    """
    
    def create(self, **kwargs) -> Agent:
        """
        Create the Data Collection Agent.
        
        Returns:
            Configured Agent instance
        """
        agent = Agent(
            role="Data Collection Specialist",
            goal="""Retrieve high-quality data from various sources, validate its structure 
            and integrity, and provide comprehensive data profiling to support downstream analysis.""",
            backstory="""You are a meticulous data engineer with expertise in data acquisition 
            from diverse sources. You have years of experience working with APIs, databases, 
            files, and web scraping. You understand that quality analysis starts with quality 
            data, so you're thorough in your validation and profiling. You're skilled at 
            identifying data quality issues early and communicating them clearly. You know 
            how to handle various data formats (CSV, JSON, Excel, Parquet, SQL databases) and 
            can work with both structured and semi-structured data. You always document your 
            data sources and any transformations applied during collection.""",
            verbose=kwargs.get('verbose', True),
            allow_delegation=False,
            memory=kwargs.get('memory', True),
            max_iter=kwargs.get('max_iter', 2)
        )
        
        self.logger.info("Data Collection Agent created successfully")
        return agent
    
    def create_tasks(self, agent: Agent, context: Dict[str, Any]) -> List[Task]:
        """
        Create tasks for the Data Collection Agent.
        
        Args:
            agent: The agent instance
            context: Context dictionary containing:
                - data_source: Data source information
                - analysis_objective: Analysis objective
                
        Returns:
            List of Task objects
        """
        data_source = context.get('data_source', {})
        source_type = data_source.get('source_type', 'file')
        source_path = data_source.get('source_path', 'unknown')
        analysis_objective = context.get('analysis_objective', '')
        
        tasks = []
        
        # Task 1: Data Retrieval and Validation
        retrieval_task = Task(
            description=f"""
            Retrieve and validate data for the analysis.
            
            Data Source Type: {source_type}
            Source Path: {source_path}
            Analysis Objective: {analysis_objective}
            
            Your responsibilities:
            1. Retrieve data from the specified source
            2. Validate data structure and format
            3. Check for basic data quality issues:
               - Missing or null values
               - Duplicate records
               - Data type inconsistencies
               - Structural problems
            4. Assess data completeness
            5. Document any issues encountered
            6. Provide initial recommendations for data cleaning
            
            For file sources:
            - Verify file exists and is readable
            - Check file format is supported
            - Validate column names and data types
            
            For API sources:
            - Verify endpoint accessibility
            - Validate response format
            - Check for rate limiting
            
            For database sources:
            - Verify connection
            - Validate schema
            - Check for access permissions
            
            Handle errors gracefully and provide clear error messages.
            """,
            expected_output="""A data retrieval report containing:
            1. Source Information:
               - Source type and location
               - Retrieval timestamp
               - Access method used
            
            2. Data Structure:
               - Number of rows and columns
               - Column names and data types
               - Memory usage estimate
            
            3. Initial Quality Assessment:
               - Missing value count per column
               - Duplicate record count
               - Data type issues
               - Structural problems
            
            4. Data Sample:
               - First few rows
               - Summary statistics for numeric columns
               - Unique value counts for categorical columns
            
            5. Issues Encountered:
               - List of problems found
               - Severity level for each issue
               - Potential impact on analysis
            
            6. Recommendations:
               - Suggested cleaning steps
               - Data transformation needs
               - Columns that may need special handling
            
            7. Data Readiness:
               - Overall quality score (0-100)
               - Ready for analysis: Yes/No
               - Required preprocessing steps""",
            agent=agent
        )
        tasks.append(retrieval_task)
        
        # Task 2: Data Profiling
        profiling_task = Task(
            description=f"""
            Create a comprehensive data profile to support analysis planning.
            
            Analysis Objective: {analysis_objective}
            
            Your responsibilities:
            1. Generate detailed statistics for all columns:
               - Numeric columns: min, max, mean, median, std dev, quartiles
               - Categorical columns: unique values, frequency distribution
               - Date columns: range, gaps, patterns
            2. Identify data characteristics:
               - Distributions (normal, skewed, bimodal, etc.)
               - Outliers and anomalies
               - Correlation patterns (preliminary)
               - Missing data patterns
            3. Assess data suitability for analysis:
               - Sufficient sample size
               - Appropriate granularity
               - Relevant features for objective
               - Temporal coverage (if applicable)
            4. Document data lineage:
               - Original source
               - Collection method
               - Any filters applied
               - Sampling strategy (if used)
            5. Provide analysis-specific insights:
               - Key columns for the objective
               - Potential challenges
               - Data limitations
            
            Focus on insights that will guide the analysis strategy.
            """,
            expected_output="""A comprehensive data profile containing:
            1. Dataset Overview:
               - Total records and features
               - Date range (if applicable)
               - Sampling information
            
            2. Column Profiles:
               For each column:
               - Data type
               - Completeness (% non-null)
               - Unique values count
               - Distribution summary
               - Notable patterns or issues
            
            3. Statistical Summary:
               - Descriptive statistics table
               - Distribution characteristics
               - Outlier analysis
               - Correlation matrix (for numeric columns)
            
            4. Data Quality Metrics:
               - Completeness score
               - Consistency score
               - Validity score
               - Accuracy indicators
            
            5. Analysis Suitability:
               - Sufficient data: Yes/No with explanation
               - Key features identified
               - Potential challenges
               - Limitations and caveats
            
            6. Recommendations:
               - Priority columns for analysis
               - Suggested transformations
               - Feature engineering opportunities
               - Analysis techniques to consider
            
            7. Data Lineage:
               - Source documentation
               - Collection timestamp
               - Processing steps applied
               - Version/snapshot information""",
            agent=agent
        )
        tasks.append(profiling_task)
        
        self.logger.info(f"Created {len(tasks)} tasks for Data Collection Agent")
        self.log_to_memory(f"Created {len(tasks)} data collection tasks", {
            "source_type": source_type,
            "source_path": source_path
        })
        
        return tasks
