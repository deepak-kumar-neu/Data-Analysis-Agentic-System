"""
Quality Assurance Agent - Validates results and ensures accuracy.

This agent performs final validation checks, ensures consistency,
and provides confidence scoring for results.
"""

from crewai import Agent, Task
from typing import Dict, Any, List

from src.agents.base_agent import BaseAgent


class QualityAssuranceAgent(BaseAgent):
    """
    Quality Assurance Agent responsible for validating analysis results.
    
    Key Responsibilities:
    - Validate statistical results
    - Check data consistency
    - Verify assumptions
    - Score confidence levels
    - Identify potential issues
    - Provide quality certification
    """
    
    def create(self, **kwargs) -> Agent:
        """
        Create the Quality Assurance Agent.
        
        Returns:
            Configured Agent instance
        """
        agent = Agent(
            role="Quality Assurance Specialist",
            goal="""Ensure the highest quality and reliability of analysis results through 
            rigorous validation, consistency checks, assumption verification, and providing 
            confidence scores for all findings.""",
            backstory="""You are a meticulous quality assurance expert with a background in 
            both statistics and auditing. You have years of experience reviewing analytical 
            work and catching errors that others miss. You understand that even small mistakes 
            in data analysis can lead to costly business decisions, so you're thorough and 
            systematic in your validation approach. You know all the common pitfalls in data 
            analysis: p-hacking, data leakage, survivorship bias, Simpson's paradox, and many 
            others. You're skilled at verifying statistical assumptions, checking for logical 
            consistency, and ensuring reproducibility. You validate that correlations make 
            sense, that sample sizes are adequate, that confidence intervals are properly 
            calculated, and that conclusions are supported by evidence. You provide clear, 
            actionable feedback about potential issues and assign appropriate confidence scores 
            to findings based on statistical rigor and data quality. You're not just a critic—
            you're a partner in ensuring that analysis delivers reliable, trustworthy insights 
            that stakeholders can confidently act upon.""",
            verbose=kwargs.get('verbose', True),
            allow_delegation=False,
            memory=kwargs.get('memory', True),
            max_iter=kwargs.get('max_iter', 2)
        )
        
        self.logger.info("Quality Assurance Agent created successfully")
        return agent
    
    def create_tasks(self, agent: Agent, context: Dict[str, Any]) -> List[Task]:
        """
        Create tasks for the Quality Assurance Agent.
        
        Args:
            agent: The agent instance
            context: Context dictionary
                
        Returns:
            List of Task objects
        """
        analysis_objective = context.get('analysis_objective', '')
        
        tasks = []
        
        # Task: Quality Validation
        validation_task = Task(
            description=f"""
            Perform comprehensive quality assurance on analysis results.
            
            Analysis Objective: {analysis_objective}
            
            Your responsibilities:
            1. Data Quality Validation:
               - Verify data collection completeness
               - Check data cleaning appropriateness
               - Validate transformation correctness
               - Ensure no data leakage
               - Confirm reproducibility
            
            2. Statistical Validity Checks:
               - Verify all statistical test assumptions
               - Check sample size adequacy for tests used
               - Validate significance levels and p-values
               - Review confidence interval calculations
               - Ensure appropriate test selection
               - Check for multiple testing corrections
            
            3. Result Consistency Checks:
               - Cross-validate findings across methods
               - Check for logical contradictions
               - Verify calculations independently
               - Ensure visualizations match reported statistics
               - Validate that conclusions follow from evidence
            
            4. Common Error Detection:
               - Check for p-hacking signs
               - Look for data leakage
               - Identify survivorship bias
               - Detect Simpson's paradox
               - Flag suspicious correlations
               - Identify overfitting indicators
               - Check for selection bias
            
            5. Assumption Verification:
               - Normality assumptions (if used)
               - Independence assumptions
               - Homoscedasticity (if relevant)
               - Linearity assumptions
               - No multicollinearity (for regression)
               - Document assumption violations
            
            6. Confidence Scoring:
               - Rate confidence in each major finding (0-100)
               - Consider:
                 * Statistical significance
                 * Effect size
                 * Sample size
                 * Data quality
                 * Methodological rigor
                 * Business plausibility
               - Provide justification for scores
            
            7. Reproducibility Check:
               - Verify all steps are documented
               - Check if analysis can be reproduced
               - Validate that code/methods are clear
               - Ensure parameters are recorded
            
            8. Limitations Documentation:
               - Identify analysis limitations
               - Document assumptions made
               - Note data constraints
               - Flag areas of uncertainty
               - Recommend additional validation
            
            9. Recommendation Review:
               - Assess if recommendations follow from analysis
               - Check for logical gaps
               - Evaluate feasibility
               - Consider potential unintended consequences
            
            10. Final Certification:
                - Determine overall quality level
                - Identify any blocking issues
                - Provide go/no-go recommendation
                - Suggest improvements if needed
            
            Be thorough but fair. The goal is reliable, trustworthy insights.
            """,
            expected_output="""A comprehensive quality assurance report containing:
            1. Executive Summary:
               - Overall quality rating (0-100)
               - Go/No-Go recommendation
               - Critical issues (if any)
               - Key strengths
               - Areas needing attention
            
            2. Data Quality Assessment:
               - Collection quality: Pass/Fail with score
               - Cleaning appropriateness: Pass/Fail with justification
               - Transformation validity: Pass/Fail with notes
               - Data leakage check: Pass/Fail
               - Reproducibility: Yes/No with details
            
            3. Statistical Validity Results:
               - Test assumptions verified: List with Pass/Fail
               - Sample size adequacy: Adequate/Inadequate with reasoning
               - Significance testing: Valid/Invalid with details
               - Test selection: Appropriate/Questionable with explanation
               - Overall statistical rigor: Score (0-100)
            
            4. Consistency Validation:
               - Cross-validation results: Consistent/Inconsistent
               - Logical coherence: Yes/No with any issues identified
               - Calculation verification: Verified/Issues found
               - Visualization accuracy: Accurate/Discrepancies
               - Conclusion validity: Supported/Not supported
            
            5. Error Detection Results:
               - P-hacking indicators: None/Possible/Likely
               - Data leakage: None detected/Found
               - Biases identified: List
               - Suspicious patterns: List with explanations
               - Overfitting signs: None/Present
            
            6. Assumption Verification:
               - For each assumption:
                 * Test name
                 * Assumption checked
                 * Verification result (Met/Violated)
                 * Impact if violated
                 * Mitigation applied
            
            7. Confidence Scores:
               - For each major finding:
                 * Finding description
                 * Confidence score (0-100)
                 * Justification
                 * Factors boosting confidence
                 * Factors reducing confidence
            
            8. Reproducibility Assessment:
               - Documentation completeness: Complete/Incomplete
               - Reproducibility: Fully/Partially/Not reproducible
               - Missing information: List
               - Recommendations for improvement
            
            9. Limitations Identified:
               - Data limitations: List
               - Methodological limitations: List
               - Scope limitations: List
               - Uncertainty areas: List
               - Recommended caveats for reporting
            
            10. Recommendations Validation:
                - Recommendations reviewed: Count
                - Evidence-based: Yes/No for each
                - Logical: Yes/No for each
                - Feasible: Yes/No for each
                - Risks identified: List
            
            11. Issues and Findings:
                - Critical issues (must fix): List
                - Major concerns (should fix): List
                - Minor issues (nice to fix): List
                - Positive findings: List
            
            12. Final Certification:
                - Overall quality score: 0-100
                - Quality level: Excellent/Good/Fair/Poor
                - Ready for presentation: Yes/No/With caveats
                - Certification statement
                - Recommendations for improvement
                - Required actions before use
            
            13. Detailed Findings Log:
                - Chronological list of all checks
                - Results for each check
                - Evidence reviewed
                - Decisions made
            
            14. Sign-off:
                - QA agent signature
                - Timestamp
                - Version reviewed
                - Next review date (if applicable)""",
            agent=agent
        )
        tasks.append(validation_task)
        
        self.logger.info(f"Created {len(tasks)} tasks for Quality Assurance Agent")
        self.log_to_memory(f"Created {len(tasks)} quality assurance tasks", {
            "objective": analysis_objective
        })
        
        return tasks
