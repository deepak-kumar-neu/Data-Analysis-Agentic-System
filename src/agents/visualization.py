"""
Visualization Agent - Creates compelling data visualizations.

This agent generates clear, informative visualizations that communicate
insights effectively to stakeholders.
"""

from crewai import Agent, Task
from typing import Dict, Any, List

from src.agents.base_agent import BaseAgent


class VisualizationAgent(BaseAgent):
    """
    Visualization Agent responsible for creating data visualizations.
    
    Key Responsibilities:
    - Select appropriate visualization types
    - Create clear, informative charts
    - Design compelling data narratives
    - Ensure accessibility and clarity
    - Generate publication-quality outputs
    """
    
    def create(self, **kwargs) -> Agent:
        """
        Create the Visualization Agent.
        
        Returns:
            Configured Agent instance
        """
        agent = Agent(
            role="Data Visualization Specialist",
            goal="""Create clear, compelling, and insightful visualizations that effectively 
            communicate data patterns, relationships, and key findings to both technical and 
            non-technical audiences.""",
            backstory="""You are an expert data visualization designer with a background in both 
            data science and visual communication. You understand that a great visualization can 
            reveal insights that tables of numbers cannot, and you know how to choose the right 
            chart type for each situation. You're proficient in various visualization libraries 
            and follow best practices from Edward Tufte, Stephen Few, and modern data visualization 
            principles. You know when to use bar charts vs. line charts, when scatter plots reveal 
            more than correlations, and when to employ advanced visualizations like heatmaps, 
            network graphs, or parallel coordinates. You understand color theory and accessibility, 
            ensuring your visualizations work for colorblind audiences. You're skilled at creating 
            both exploratory visualizations for analysis and explanatory visualizations for 
            presentation. You always label axes clearly, include appropriate titles, add legends 
            when needed, and ensure your visualizations can stand alone without extensive explanation. 
            You know how to balance aesthetic appeal with analytical clarity.""",
            verbose=kwargs.get('verbose', True),
            allow_delegation=False,
            memory=kwargs.get('memory', True),
            max_iter=kwargs.get('max_iter', 2)
        )
        
        self.logger.info("Visualization Agent created successfully")
        return agent
    
    def create_tasks(self, agent: Agent, context: Dict[str, Any]) -> List[Task]:
        """
        Create tasks for the Visualization Agent.
        
        Args:
            agent: The agent instance
            context: Context dictionary
                
        Returns:
            List of Task objects
        """
        analysis_objective = context.get('analysis_objective', '')
        
        tasks = []
        
        # Task: Create Visualizations
        visualization_task = Task(
            description=f"""
            Create a comprehensive set of visualizations to support analysis findings.
            
            Analysis Objective: {analysis_objective}
            
            Your responsibilities:
            1. Visualization Planning:
               - Review analysis results and key findings
               - Identify insights that benefit from visualization
               - Select appropriate chart types for each insight
               - Plan visualization sequence for storytelling
            
            2. Core Visualizations to Create:
               
               A. Distribution Visualizations:
                  - Histograms for numeric variable distributions
                  - Box plots for comparison across groups
                  - Violin plots for detailed distribution comparison
                  - Add statistical annotations (mean, median, quartiles)
               
               B. Relationship Visualizations:
                  - Scatter plots for bivariate relationships
                  - Correlation heatmaps for multivariate analysis
                  - Pair plots for feature relationships
                  - Add trendlines and R² values where appropriate
               
               C. Comparison Visualizations:
                  - Bar charts for categorical comparisons
                  - Grouped/stacked bars for multi-dimensional comparison
                  - Error bars for confidence intervals
                  - Highlight significant differences
               
               D. Trend Visualizations (if time-based):
                  - Line charts for temporal trends
                  - Area charts for cumulative trends
                  - Multi-line charts for comparison over time
                  - Annotate key events or change points
               
               E. Composition Visualizations:
                  - Pie charts for simple proportions (use sparingly)
                  - Stacked bar charts for composition over categories
                  - Treemaps for hierarchical data
               
               F. Advanced Visualizations:
                  - Heatmaps for matrix data
                  - Network graphs for relationships
                  - Geographic maps (if location data)
                  - Parallel coordinates for multivariate data
            
            3. Visualization Best Practices:
               - Use clear, descriptive titles
               - Label all axes with units
               - Include legends when necessary
               - Use colorblind-friendly palettes
               - Ensure readable font sizes
               - Remove chart junk (unnecessary elements)
               - Maintain consistent styling
               - Add data source and timestamp
            
            4. Insights Annotation:
               - Highlight key findings in visualizations
               - Add text annotations for important points
               - Use colors to draw attention
               - Include statistical values (p-values, confidence intervals)
            
            5. Dashboard/Report Layout:
               - Arrange visualizations logically
               - Create narrative flow
               - Group related visualizations
               - Include summary statistics
            
            6. Quality Checks:
               - Verify data accuracy in visualizations
               - Ensure no misleading representations
               - Check for appropriate scales (not truncated axes)
               - Validate color choices
               - Test readability at different sizes
            
            Create visualizations that tell a clear, compelling data story.
            """,
            expected_output="""A visualization package containing:
            1. Visualization Catalog:
               - List of all visualizations created
               - Purpose of each visualization
               - Key insights highlighted
               - File formats (PNG, SVG, HTML interactive)
            
            2. Distribution Visualizations:
               - Histograms with statistical overlays
               - Box plots with outlier identification
               - Description of distribution characteristics
            
            3. Relationship Visualizations:
               - Scatter plots with trendlines
               - Correlation heatmaps with significance
               - Key relationships identified
            
            4. Comparison Visualizations:
               - Bar/column charts with error bars
               - Significant differences highlighted
               - Magnitude of differences quantified
            
            5. Trend Visualizations (if applicable):
               - Time series plots with smoothing
               - Trend directions and magnitudes
               - Seasonal patterns visualized
               - Forecast visualizations
            
            6. Composition Visualizations:
               - Proportion breakdowns
               - Hierarchical representations
               - Changes in composition over time/groups
            
            7. Advanced Visualizations:
               - Complex multivariate visualizations
               - Interactive elements (if HTML output)
               - Custom visualizations for unique insights
            
            8. Dashboard/Summary View:
               - Integrated view of key visualizations
               - Logical layout and flow
               - Summary statistics included
               - Call-outs for key findings
            
            9. Visualization Narrative:
               - Story told by visualization sequence
               - How visualizations support conclusions
               - Progression from exploration to insight
            
            10. Technical Details:
                - Visualization specifications
                - Color palettes used
                - Tools/libraries utilized
                - File locations and formats
                - Accessibility features implemented
            
            11. Usage Guide:
                - How to interpret each visualization
                - Key points to notice
                - Caveats and limitations
                - Recommended audience
            
            12. Quality Assurance:
                - Accuracy verification: Pass/Fail
                - Accessibility check: Pass/Fail
                - Best practices compliance: Yes/No
                - Ready for presentation: Yes/No""",
            agent=agent
        )
        tasks.append(visualization_task)
        
        self.logger.info(f"Created {len(tasks)} tasks for Visualization Agent")
        self.log_to_memory(f"Created {len(tasks)} visualization tasks", {
            "objective": analysis_objective
        })
        
        return tasks
