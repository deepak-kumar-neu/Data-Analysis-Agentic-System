"""
Data Processing Agent - Cleans and transforms data for analysis.

This agent handles data cleaning, transformation, and preparation to ensure
high-quality input for statistical analysis.
"""

from crewai import Agent, Task
from typing import Dict, Any, List

from src.agents.base_agent import BaseAgent


class DataProcessingAgent(BaseAgent):
    """
    Data Processing Agent responsible for cleaning and transforming data.
    
    Key Responsibilities:
    - Handle missing values intelligently
    - Detect and handle outliers
    - Remove duplicates
    - Normalize and standardize data
    - Create derived features
    - Ensure data quality for analysis
    """
    
    def create(self, **kwargs) -> Agent:
        """
        Create the Data Processing Agent.
        
        Returns:
            Configured Agent instance
        """
        agent = Agent(
            role="Data Processing Expert",
            goal="""Transform raw data into analysis-ready datasets through intelligent cleaning, 
            handling of missing values, outlier detection, and feature engineering, while maintaining 
            data integrity and documenting all transformations.""",
            backstory="""You are a seasoned data scientist with deep expertise in data preprocessing 
            and feature engineering. You understand that the quality of analysis depends heavily on 
            the quality of data preparation. You're skilled at making informed decisions about how 
            to handle missing data, whether to remove or transform outliers, and how to engineer 
            features that enhance analytical insights. You know multiple imputation techniques 
            (mean, median, forward-fill, KNN, etc.) and understand when each is appropriate. You're 
            familiar with various scaling methods (standardization, normalization, robust scaling) 
            and their impacts on different analytical techniques. You always document your decisions 
            and their rationale, ensuring reproducibility and transparency. You balance statistical 
            rigor with practical considerations, never blindly applying transformations without 
            understanding their implications.""",
            verbose=kwargs.get('verbose', True),
            allow_delegation=False,
            memory=kwargs.get('memory', True),
            max_iter=kwargs.get('max_iter', 2)
        )
        
        self.logger.info("Data Processing Agent created successfully")
        return agent
    
    def create_tasks(self, agent: Agent, context: Dict[str, Any]) -> List[Task]:
        """
        Create tasks for the Data Processing Agent.
        
        Args:
            agent: The agent instance
            context: Context dictionary
                
        Returns:
            List of Task objects
        """
        analysis_objective = context.get('analysis_objective', '')
        data_source = context.get('data_source', {})
        target_column = context.get('target_column')
        
        tasks = []
        
        # Task 1: Data Cleaning
        cleaning_task = Task(
            description=f"""
            Clean the dataset to ensure high-quality input for analysis.
            
            Analysis Objective: {analysis_objective}
            Target Column: {target_column if target_column else 'Not specified'}
            
            Your responsibilities:
            1. Handle Missing Values:
               - Analyze missing data patterns
               - Determine if data is MCAR, MAR, or MNAR
               - Choose appropriate imputation strategy per column:
                 * Numeric: mean, median, KNN, interpolation
                 * Categorical: mode, forward-fill, create 'unknown' category
                 * Time series: interpolation, seasonal methods
               - Document imputation decisions
            
            2. Handle Duplicates:
               - Identify exact and near-duplicates
               - Determine which duplicates to keep/remove
               - Document duplicate removal strategy
            
            3. Handle Outliers:
               - Detect outliers using multiple methods:
                 * IQR method
                 * Z-score method
                 * Isolation Forest
               - Determine treatment strategy:
                 * Keep (if legitimate extreme values)
                 * Remove (if errors)
                 * Cap/floor (winsorization)
                 * Transform (log, Box-Cox)
               - Justify decisions based on analysis objective
            
            4. Fix Data Type Issues:
               - Convert columns to appropriate types
               - Handle malformed entries
               - Standardize formatting (dates, strings, etc.)
            
            5. Handle Inconsistencies:
               - Standardize categorical values (case, spelling)
               - Fix encoding issues
               - Resolve conflicting data
            
            6. Quality Checks:
               - Verify no new issues introduced
               - Ensure data integrity maintained
               - Validate relationships preserved
            
            Prioritize actions that support the analysis objective.
            """,
            expected_output="""A data cleaning report containing:
            1. Cleaning Summary:
               - Original dataset size
               - Cleaned dataset size
               - Records removed/modified
               - Overall quality improvement
            
            2. Missing Value Treatment:
               - Columns with missing data
               - Missing data pattern (MCAR/MAR/MNAR)
               - Imputation method used per column
               - Justification for each method
               - Before/after missing value counts
            
            3. Duplicate Handling:
               - Number of duplicates found
               - Duplicate types (exact, near)
               - Removal strategy
               - Records removed
            
            4. Outlier Treatment:
               - Columns analyzed for outliers
               - Detection method used
               - Number of outliers found per column
               - Treatment applied (keep/remove/transform)
               - Justification for decisions
               - Impact on distributions
            
            5. Data Type Corrections:
               - Columns converted
               - Conversion details
               - Problematic entries handled
            
            6. Standardization:
               - Categorical values standardized
               - Format corrections made
               - Encoding fixes applied
            
            7. Quality Metrics:
               - Completeness: Before/After
               - Consistency: Before/After
               - Validity: Before/After
               - Overall Quality Score
            
            8. Data Transformations Log:
               - Chronological list of all transformations
               - Parameters used
               - Code/commands executed
            
            9. Cleaned Data Summary:
               - Final dataset characteristics
               - Ready for analysis: Yes/No
               - Any remaining issues""",
            agent=agent
        )
        tasks.append(cleaning_task)
        
        # Task 2: Feature Engineering and Transformation
        transformation_task = Task(
            description=f"""
            Engineer features and transform data to enhance analytical insights.
            
            Analysis Objective: {analysis_objective}
            Target Column: {target_column if target_column else 'Not specified'}
            
            Your responsibilities:
            1. Feature Engineering:
               - Create derived features that support analysis objective
               - Extract temporal features (if date columns exist):
                 * Day of week, month, quarter, year
                 * Time since event
                 * Seasonal indicators
               - Create interaction features (when appropriate)
               - Bin continuous variables (if beneficial)
               - Encode categorical variables:
                 * One-hot encoding for low cardinality
                 * Label encoding for ordinal
                 * Target encoding (if applicable)
            
            2. Feature Scaling/Normalization:
               - Determine if scaling is needed
               - Choose appropriate method:
                 * Standardization (z-score) for normally distributed
                 * Min-max scaling for bounded ranges
                 * Robust scaling for outlier presence
               - Apply to appropriate columns
               - Document scaling parameters for reproducibility
            
            3. Feature Transformation:
               - Apply transformations to improve distributions:
                 * Log transformation for right-skewed data
                 * Square root for moderate skewness
                 * Box-Cox for automatic selection
               - Create polynomial features (if beneficial)
               - Apply dimensionality reduction (if needed)
            
            4. Feature Selection:
               - Identify most relevant features for objective
               - Remove highly correlated features (if appropriate)
               - Remove low-variance features
               - Document feature importance
            
            5. Data Validation:
               - Verify transformations applied correctly
               - Check for data leakage
               - Ensure no information loss
               - Validate statistical properties
            
            Focus on transformations that will enhance the specific analysis objective.
            """,
            expected_output="""A feature engineering report containing:
            1. Feature Engineering Summary:
               - Original feature count
               - Final feature count
               - New features created
               - Features removed
            
            2. Derived Features:
               - List of new features created
               - Creation logic for each
               - Rationale and expected benefit
               - Sample values
            
            3. Temporal Features (if applicable):
               - Date/time features extracted
               - Temporal patterns identified
               - Seasonality indicators added
            
            4. Encoding Details:
               - Categorical features encoded
               - Encoding method per feature
               - Dimensionality impact
               - Encoding mappings documented
            
            5. Scaling/Normalization:
               - Features scaled
               - Scaling method used
               - Parameters (mean, std, min, max, etc.)
               - Distribution before/after
            
            6. Transformations Applied:
               - Features transformed
               - Transformation type
               - Parameters used
               - Impact on distribution
               - Improvement metrics
            
            7. Feature Selection:
               - Selection criteria used
               - Features retained/removed
               - Feature importance scores
               - Correlation analysis results
            
            8. Final Feature Set:
               - Complete list of features
               - Feature descriptions
               - Data types
               - Statistical summaries
            
            9. Quality Assurance:
               - Validation checks performed
               - Data leakage check: Pass/Fail
               - Information preservation verified
               - Ready for analysis: Yes/No
            
            10. Reproducibility:
                - All transformation parameters
                - Sequence of operations
                - Code/configuration for reproduction""",
            agent=agent
        )
        tasks.append(transformation_task)
        
        self.logger.info(f"Created {len(tasks)} tasks for Data Processing Agent")
        self.log_to_memory(f"Created {len(tasks)} data processing tasks", {
            "objective": analysis_objective
        })
        
        return tasks
