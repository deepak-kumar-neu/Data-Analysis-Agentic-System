"""
Controller Agent - Orchestrates the entire data analysis workflow.

This agent acts as the central coordinator, delegating tasks to specialized agents,
managing the overall workflow, and synthesizing final results.
"""

from crewai import Agent, Task
from typing import Dict, Any, List, Optional

from src.agents.base_agent import BaseAgent


class ControllerAgent(BaseAgent):
    """
    Controller Agent responsible for orchestrating the data analysis workflow.
    
    Key Responsibilities:
    - Break down analysis objectives into specific tasks
    - Delegate tasks to specialized agents
    - Coordinate workflow execution
    - Handle errors and fallback strategies
    - Synthesize results from all agents
    - Generate final comprehensive reports
    """
    
    def create(self, **kwargs) -> Agent:
        """
        Create the Controller Agent with appropriate configuration.
        
        Returns:
            Configured Agent instance
        """
        agent = Agent(
            role="Data Analysis Coordinator",
            goal="""Orchestrate the end-to-end data analysis process, coordinate specialized 
            agents, ensure quality results, and synthesize comprehensive insights that address 
            the business objectives.""",
            backstory="""You are an experienced data analytics project manager with a proven 
            track record of leading complex analytical initiatives. You excel at breaking down 
            ambiguous business questions into clear, actionable tasks and coordinating diverse 
            teams of specialists. Your strength lies in your ability to see the big picture while 
            managing details, ensuring that all analysis activities align with core business 
            objectives. You have a talent for synthesizing information from multiple sources into 
            coherent, focused reports that drive decision-making. You're known for your ability 
            to anticipate challenges, implement fallback strategies, and ensure projects deliver 
            value even when facing unexpected obstacles.""",
            verbose=kwargs.get('verbose', True),
            allow_delegation=True,  # Controller can delegate to other agents
            memory=kwargs.get('memory', True),
            max_iter=kwargs.get('max_iter', 3)
        )
        
        self.logger.info("Controller Agent created successfully")
        return agent
    
    def create_tasks(self, agent: Agent, context: Dict[str, Any]) -> List[Task]:
        """
        Create tasks for the Controller Agent.
        
        Args:
            agent: The agent instance
            context: Context dictionary containing:
                - analysis_objective: Main objective
                - business_context: Business context
                - data_source: Data source information
                
        Returns:
            List of Task objects
        """
        # Extract context
        analysis_objective = context.get('analysis_objective', 'Identify key patterns and insights')
        business_context = context.get('business_context', 'General business analysis')
        data_source = context.get('data_source', {})
        
        tasks = []
        
        # Task 1: Planning Task
        planning_task = Task(
            description=f"""
            Create a comprehensive analysis plan for the objective: {analysis_objective}
            
            Business Context: {business_context}
            Data Source: {data_source.get('source_type', 'unknown')} - {data_source.get('source_path', 'unknown')}
            
            Your responsibilities:
            1. Analyze the business objective and break it down into specific analytical questions
            2. Identify what data needs to be collected and validated
            3. Determine required data processing and cleaning steps
            4. Plan the sequence of analytical techniques to apply
            5. Identify potential challenges and define mitigation strategies
            6. Define success criteria and quality metrics
            7. Plan for visualization and reporting needs
            8. Create a task delegation strategy for specialized agents
            
            Consider:
            - Data quality requirements
            - Statistical rigor needed
            - Time and resource constraints
            - Stakeholder expectations
            - Potential data issues
            
            Output a detailed plan that will guide the entire analysis process.
            """,
            expected_output="""A comprehensive analysis plan document containing:
            1. Executive Summary: High-level overview of the approach
            2. Analytical Questions: Specific questions to be answered
            3. Data Requirements: Required data and quality standards
            4. Processing Steps: Data cleaning and transformation plan
            5. Analysis Strategy: Statistical methods and techniques to use
            6. Risk Assessment: Potential challenges and mitigation strategies
            7. Task Dependencies: Sequence and dependencies of tasks
            8. Quality Criteria: How success will be measured
            9. Deliverables: Expected outputs and visualizations
            10. Timeline: Estimated effort for each phase""",
            agent=agent
        )
        tasks.append(planning_task)
        
        # Task 2: Coordination Task (depends on other agents' outputs)
        coordination_task = Task(
            description=f"""
            Coordinate the execution of the analysis workflow and monitor progress.
            
            Objective: {analysis_objective}
            
            Your responsibilities:
            1. Monitor the progress of all specialized agents
            2. Ensure data flows correctly between agents
            3. Validate intermediate results for quality
            4. Handle any errors or issues that arise
            5. Make decisions about workflow adjustments if needed
            6. Ensure all agents have the context they need
            7. Track performance metrics
            8. Implement fallback strategies if primary approaches fail
            
            Focus on:
            - Quality assurance at each step
            - Communication between agents
            - Resource optimization
            - Timeline management
            - Risk mitigation
            
            Ensure smooth execution and high-quality outputs.
            """,
            expected_output="""A coordination report containing:
            1. Execution Status: Progress of each agent and task
            2. Quality Checks: Validation results at each stage
            3. Issues Encountered: Problems and their resolutions
            4. Performance Metrics: Execution times and resource usage
            5. Adjustments Made: Any workflow modifications
            6. Agent Communications: Key interactions and decisions
            7. Risk Mitigation: Fallback strategies employed
            8. Overall Health: Assessment of workflow execution""",
            agent=agent
        )
        tasks.append(coordination_task)
        
        # Task 3: Final Report Synthesis
        synthesis_task = Task(
            description=f"""
            Synthesize all findings into a comprehensive final report.
            
            Analysis Objective: {analysis_objective}
            Business Context: {business_context}
            
            Your responsibilities:
            1. Compile results from all specialized agents:
               - Data collection findings
               - Data quality assessment
               - Statistical analysis results
               - Generated insights
               - Visualizations created
            2. Organize insights by business relevance and impact
            3. Ensure all conclusions are data-driven and supported
            4. Provide clear, actionable recommendations
            5. Create an executive summary for stakeholders
            6. Document methodology and limitations
            7. Highlight key metrics and performance indicators
            8. Include quality assurance validation results
            
            The report should:
            - Tell a coherent, compelling story
            - Address the original business objective
            - Be accessible to both technical and non-technical audiences
            - Provide clear next steps
            - Acknowledge limitations and assumptions
            
            Create a report that drives informed decision-making.
            """,
            expected_output="""A comprehensive final analysis report containing:
            
            1. EXECUTIVE SUMMARY
               - Key findings (3-5 bullet points)
               - Primary recommendations
               - Business impact
               - Confidence level
            
            2. INTRODUCTION
               - Analysis objective and scope
               - Business context and motivation
               - Success criteria
            
            3. METHODOLOGY
               - Data sources and collection process
               - Data quality assessment
               - Analytical techniques applied
               - Tools and technologies used
            
            4. DATA OVERVIEW
               - Dataset characteristics
               - Data quality metrics
               - Cleaning and transformation steps
               - Limitations and constraints
            
            5. ANALYSIS RESULTS
               - Statistical findings with significance levels
               - Pattern and trend discoveries
               - Correlation and relationship analysis
               - Anomaly and outlier detection
               - Segment and cohort analysis
            
            6. INSIGHTS & INTERPRETATION
               - Key insights ranked by importance
               - Business implications
               - Supporting evidence
               - Confidence levels
            
            7. VISUALIZATIONS
               - Charts and graphs with interpretations
               - Data storytelling elements
            
            8. RECOMMENDATIONS
               - Actionable next steps
               - Prioritized by impact and feasibility
               - Resource requirements
               - Expected outcomes
            
            9. QUALITY ASSURANCE
               - Validation results
               - Consistency checks
               - Reliability assessment
            
            10. LIMITATIONS & ASSUMPTIONS
                - Data limitations
                - Methodological constraints
                - Assumptions made
                - Areas for further investigation
            
            11. APPENDICES
                - Detailed statistical outputs
                - Additional visualizations
                - Technical documentation
                - Glossary of terms""",
            agent=agent
        )
        tasks.append(synthesis_task)
        
        self.logger.info(f"Created {len(tasks)} tasks for Controller Agent")
        self.log_to_memory(f"Created {len(tasks)} controller tasks", {
            "objective": analysis_objective,
            "context": business_context
        })
        
        return tasks
    
    def create_fallback_plan(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a fallback plan if primary analysis fails.
        
        Args:
            context: Current analysis context
            
        Returns:
            Fallback plan dictionary
        """
        fallback = {
            "status": "fallback",
            "message": "Primary analysis encountered issues, using fallback strategy",
            "actions": [
                "Use simplified analysis methods",
                "Reduce scope to core questions",
                "Focus on descriptive statistics",
                "Provide preliminary insights",
                "Flag areas needing further investigation"
            ],
            "reduced_scope": True
        }
        
        self.logger.warning("Activating fallback plan")
        self.log_to_memory("Fallback plan activated", {"context": context})
        
        return fallback
    
    def validate_workflow_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate results from the workflow.
        
        Args:
            results: Results from workflow execution
            
        Returns:
            Validation results
        """
        validation = {
            "valid": True,
            "issues": [],
            "warnings": [],
            "quality_score": 1.0
        }
        
        # Check for required components
        required = ['data_summary', 'analysis_results', 'insights']
        for req in required:
            if req not in results:
                validation['issues'].append(f"Missing required component: {req}")
                validation['valid'] = False
                validation['quality_score'] -= 0.3
        
        # Check data quality
        if 'data_summary' in results:
            data_summary = results['data_summary']
            if data_summary.get('rows', 0) < 10:
                validation['warnings'].append("Small dataset (< 10 rows)")
                validation['quality_score'] -= 0.1
        
        # Check for insights
        if 'insights' in results:
            insights = results['insights']
            if len(insights) < 3:
                validation['warnings'].append("Few insights generated (< 3)")
                validation['quality_score'] -= 0.1
        
        validation['quality_score'] = max(0.0, validation['quality_score'])
        
        self.logger.info(f"Workflow validation: {validation['valid']}, Quality: {validation['quality_score']:.2f}")
        
        return validation
    
    def plan_workflow(self, objective: str, data_source: Dict[str, Any]) -> Dict[str, Any]:
        """
        Plan the analysis workflow based on objective and data source.
        
        Args:
            objective: Analysis objective
            data_source: Data source information
            
        Returns:
            Workflow plan dictionary
        """
        self.logger.info(f"Planning workflow for objective: {objective}")
        
        # Create a comprehensive workflow plan
        plan = {
            "objective": objective,
            "data_source": data_source,
            "phases": [
                {
                    "phase": "data_collection",
                    "agent": "data_collection",
                    "description": "Retrieve and validate data from source",
                    "priority": 1
                },
                {
                    "phase": "data_processing",
                    "agent": "data_processing",
                    "description": "Clean and transform data",
                    "priority": 2
                },
                {
                    "phase": "analysis",
                    "agent": "analysis",
                    "description": "Perform statistical analysis",
                    "priority": 3
                },
                {
                    "phase": "visualization",
                    "agent": "visualization",
                    "description": "Create visualizations and insights",
                    "priority": 4
                }
            ],
            "success_criteria": {
                "data_quality": "Data must be clean with <5% missing values",
                "analysis_depth": "At least 3 statistical analyses performed",
                "insights": "Minimum 5 actionable insights generated",
                "visualizations": "At least 3 professional visualizations"
            },
            "estimated_duration": "5-10 minutes",
            "fallback_strategy": "Use simplified analysis if primary methods fail"
        }
        
        self.log_to_memory("Workflow plan created", plan)
        self.logger.info(f"Workflow plan created with {len(plan['phases'])} phases")
        
        return plan
