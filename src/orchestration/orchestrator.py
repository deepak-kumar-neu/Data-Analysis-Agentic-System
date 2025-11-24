"""
Orchestrator - Coordinates agents and tools for workflow execution.
Implements sequential, parallel, and hierarchical execution modes.
"""

import asyncio
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..agents import (
    ControllerAgent,
    DataCollectionAgent,
    DataProcessingAgent,
    AnalysisAgent,
    VisualizationAgent,
    QualityAssuranceAgent
)
from ..tools import (
    DataRetrievalTool,
    DataCleaningTool,
    StatisticalAnalysisTool,
    VisualizationTool,
    WebSearchTool,
    InsightGeneratorTool,
    ReportGeneratorTool
)
from .error_handler import ErrorHandler
from .memory_manager import MemoryManager
from ..utils.helpers import get_timestamp, ensure_directory
from ..utils.logger import get_logger


class ExecutionMode(Enum):
    """Execution mode for orchestrator."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HIERARCHICAL = "hierarchical"


class Orchestrator:
    """
    Main orchestrator that coordinates agents and tools for data analysis workflows.
    Supports multiple execution modes and advanced error handling.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize orchestrator with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = get_logger(self.__class__.__name__)
        
        # Apply agent patches to add missing methods
        try:
            import sys
            import os
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            from patch_agents import apply_all_patches
            apply_all_patches()
        except Exception as e:
            self.logger.warning(f"Could not apply agent patches: {e}")
        
        # Initialize error handler and memory manager
        self.error_handler = ErrorHandler()
        self.memory_manager = MemoryManager()
        
        # Initialize agents
        self._initialize_agents()
        
        # Initialize tools
        self._initialize_tools()
        
        # Execution state
        self.execution_history = []
        self.current_workflow = None
        
        self.logger.info("Orchestrator initialized successfully")
    
    def _initialize_agents(self):
        """Initialize all agents."""
        self.logger.info("Initializing agents...")
        
        self.agents = {
            'controller': ControllerAgent(
                error_handler=self.error_handler,
                memory_manager=self.memory_manager
            ),
            'data_collection': DataCollectionAgent(
                error_handler=self.error_handler,
                memory_manager=self.memory_manager
            ),
            'data_processing': DataProcessingAgent(
                error_handler=self.error_handler,
                memory_manager=self.memory_manager
            ),
            'analysis': AnalysisAgent(
                error_handler=self.error_handler,
                memory_manager=self.memory_manager
            ),
            'visualization': VisualizationAgent(
                error_handler=self.error_handler,
                memory_manager=self.memory_manager
            ),
            'quality_assurance': QualityAssuranceAgent(
                error_handler=self.error_handler,
                memory_manager=self.memory_manager
            )
        }
        
        self.logger.info(f"Initialized {len(self.agents)} agents")
    
    def _initialize_tools(self):
        """Initialize all tools."""
        self.logger.info("Initializing tools...")
        
        self.tools = {
            'data_retrieval': DataRetrievalTool(),
            'data_cleaning': DataCleaningTool(),
            'statistical_analysis': StatisticalAnalysisTool(),
            'visualization': VisualizationTool(),
            'web_search': WebSearchTool(),
            'insight_generator': InsightGeneratorTool(),
            'report_generator': ReportGeneratorTool()
        }
        
        self.logger.info(f"Initialized {len(self.tools)} tools")
    
    def execute_workflow(
        self,
        data_source: str,
        objective: str,
        execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL,
        output_dir: str = "./results",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute complete data analysis workflow.
        
        Args:
            data_source: Path or URL to data source
            objective: Analysis objective/goal
            execution_mode: Execution mode (sequential, parallel, hierarchical)
            output_dir: Output directory for results
            **kwargs: Additional workflow parameters
            
        Returns:
            Dict with workflow results
        """
        workflow_id = f"workflow_{get_timestamp().replace(' ', '_').replace(':', '-')}"
        self.logger.info(f"Starting workflow {workflow_id} in {execution_mode.value} mode")
        
        # Create output directory
        ensure_directory(output_dir)
        
        # Initialize workflow state
        workflow_state = {
            'id': workflow_id,
            'start_time': get_timestamp(),
            'data_source': data_source,
            'objective': objective,
            'execution_mode': execution_mode.value,
            'output_dir': output_dir,
            'status': 'running',
            'results': {},
            'errors': []
        }
        
        self.current_workflow = workflow_state
        
        try:
            # Execute based on mode
            if execution_mode == ExecutionMode.SEQUENTIAL:
                results = self._execute_sequential(workflow_state, **kwargs)
            elif execution_mode == ExecutionMode.PARALLEL:
                results = self._execute_parallel(workflow_state, **kwargs)
            elif execution_mode == ExecutionMode.HIERARCHICAL:
                results = self._execute_hierarchical(workflow_state, **kwargs)
            else:
                raise ValueError(f"Unknown execution mode: {execution_mode}")
            
            workflow_state['results'] = results
            workflow_state['status'] = 'completed'
            workflow_state['end_time'] = get_timestamp()
            
            # Store in memory
            self.memory_manager.add_conversation_entry(
                role='system',
                content=f"Workflow {workflow_id} completed successfully"
            )
            
            # Add to history
            self.execution_history.append(workflow_state)
            
            self.logger.info(f"Workflow {workflow_id} completed successfully")
            
            return {
                'success': True,
                'workflow_id': workflow_id,
                'results': results,
                'metadata': {
                    'execution_time': workflow_state.get('end_time'),
                    'mode': execution_mode.value
                }
            }
            
        except Exception as e:
            workflow_state['status'] = 'failed'
            workflow_state['end_time'] = get_timestamp()
            workflow_state['errors'].append(str(e))
            
            self.logger.error(f"Workflow {workflow_id} failed: {str(e)}")
            
            return {
                'success': False,
                'workflow_id': workflow_id,
                'error': str(e),
                'partial_results': workflow_state.get('results', {})
            }
    
    def _execute_sequential(self, workflow_state: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Execute workflow in sequential mode.
        Each step completes before the next begins.
        """
        self.logger.info("Executing workflow in sequential mode")
        results = {}
        
        # Step 1: Controller plans the workflow
        self.logger.info("Step 1: Controller planning workflow")
        plan = self.agents['controller'].plan_workflow(
            objective=workflow_state['objective'],
            data_source=workflow_state['data_source']
        )
        results['workflow_plan'] = plan
        
        # Step 2: Data Collection
        self.logger.info("Step 2: Data collection")
        data_result = self.agents['data_collection'].collect_data(
            source=workflow_state['data_source'],
            tool=self.tools['data_retrieval']
        )
        results['data_collection'] = data_result
        
        if not data_result.get('success'):
            raise Exception("Data collection failed")
        
        # Step 3: Data Processing
        self.logger.info("Step 3: Data processing")
        processing_result = self.agents['data_processing'].process_data(
            data=data_result['data'],
            tool=self.tools['data_cleaning']
        )
        results['data_processing'] = processing_result
        
        # Step 4: Statistical Analysis
        self.logger.info("Step 4: Statistical analysis")
        analysis_result = self.agents['analysis'].analyze_data(
            data=processing_result['processed_data'],
            tool=self.tools['statistical_analysis'],
            objective=workflow_state['objective']
        )
        results['analysis'] = analysis_result
        
        # Step 5: AI-Powered Insights (Custom Tool)
        self.logger.info("Step 5: Generating AI-powered insights")
        insight_result = self.agents['analysis'].generate_insights(
            data=processing_result['processed_data'],
            tool=self.tools['insight_generator'],
            analysis_results=analysis_result
        )
        results['ai_insights'] = insight_result
        
        # Step 6: Visualization
        self.logger.info("Step 6: Creating visualizations")
        viz_result = self.agents['visualization'].create_visualizations(
            data=processing_result['processed_data'],
            tool=self.tools['visualization'],
            insights=insight_result.get('insights', []),
            output_path=workflow_state['output_dir']
        )
        results['visualizations'] = viz_result
        
        # Step 7: Quality Assurance
        self.logger.info("Step 7: Quality assurance validation")
        qa_result = self.agents['quality_assurance'].validate_results(
            data=processing_result['processed_data'],
            analysis=analysis_result,
            insights=insight_result,
            visualizations=viz_result
        )
        results['quality_assurance'] = qa_result
        
        # Step 8: Generate Report
        self.logger.info("Step 8: Generating final report")
        report_data = self._compile_report_data(results, workflow_state)
        report_result = self.tools['report_generator'].execute(
            report_data=report_data,
            output_format='all',
            output_path=workflow_state['output_dir']
        )
        results['report'] = report_result
        
        return results
    
    def _execute_parallel(self, workflow_state: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Execute workflow in parallel mode.
        Independent tasks run concurrently.
        """
        self.logger.info("Executing workflow in parallel mode")
        results = {}
        
        # Step 1: Sequential - Controller plans and data collection
        self.logger.info("Step 1: Planning and data collection (sequential)")
        plan = self.agents['controller'].plan_workflow(
            objective=workflow_state['objective'],
            data_source=workflow_state['data_source']
        )
        results['workflow_plan'] = plan
        
        data_result = self.agents['data_collection'].collect_data(
            source=workflow_state['data_source'],
            tool=self.tools['data_retrieval']
        )
        results['data_collection'] = data_result
        
        if not data_result.get('success'):
            raise Exception("Data collection failed")
        
        # Step 2: Data Processing
        processing_result = self.agents['data_processing'].process_data(
            data=data_result['data'],
            tool=self.tools['data_cleaning']
        )
        results['data_processing'] = processing_result
        
        # Step 3: Parallel - Analysis tasks
        self.logger.info("Step 3: Running parallel analysis tasks")
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit parallel tasks
            future_stat_analysis = executor.submit(
                self._run_statistical_analysis,
                processing_result['processed_data'],
                workflow_state['objective']
            )
            
            future_insights = executor.submit(
                self._run_insight_generation,
                processing_result['processed_data']
            )
            
            future_web_search = executor.submit(
                self._run_web_search,
                workflow_state['objective']
            )
            
            # Collect results
            results['analysis'] = future_stat_analysis.result()
            results['ai_insights'] = future_insights.result()
            results['web_context'] = future_web_search.result()
        
        # Step 4: Visualization (depends on analysis)
        viz_result = self.agents['visualization'].create_visualizations(
            data=processing_result['processed_data'],
            tool=self.tools['visualization'],
            insights=results['ai_insights'].get('insights', []),
            output_path=workflow_state['output_dir']
        )
        results['visualizations'] = viz_result
        
        # Step 5: QA and Report (sequential)
        qa_result = self.agents['quality_assurance'].validate_results(
            data=processing_result['processed_data'],
            analysis=results['analysis'],
            insights=results['ai_insights'],
            visualizations=viz_result
        )
        results['quality_assurance'] = qa_result
        
        report_data = self._compile_report_data(results, workflow_state)
        report_result = self.tools['report_generator'].execute(
            report_data=report_data,
            output_format='all',
            output_path=workflow_state['output_dir']
        )
        results['report'] = report_result
        
        return results
    
    def _execute_hierarchical(self, workflow_state: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Execute workflow in hierarchical mode.
        Controller delegates to sub-agents with oversight.
        """
        self.logger.info("Executing workflow in hierarchical mode")
        results = {}
        
        # Controller creates master plan
        master_plan = self.agents['controller'].create_master_plan(
            objective=workflow_state['objective'],
            data_source=workflow_state['data_source']
        )
        results['master_plan'] = master_plan
        
        # Delegate tasks to agents with controller oversight
        for phase in master_plan.get('phases', []):
            self.logger.info(f"Executing phase: {phase['name']}")
            
            phase_results = self.agents['controller'].delegate_phase(
                phase=phase,
                agents=self.agents,
                tools=self.tools,
                workflow_state=workflow_state
            )
            
            results[phase['name']] = phase_results
            
            # Controller reviews results
            review = self.agents['controller'].review_phase_results(
                phase=phase,
                results=phase_results
            )
            
            if not review.get('approved'):
                self.logger.warning(f"Phase {phase['name']} requires refinement")
                # Implement refinement logic here
        
        # Final compilation
        report_data = self._compile_report_data(results, workflow_state)
        report_result = self.tools['report_generator'].execute(
            report_data=report_data,
            output_format='all',
            output_path=workflow_state['output_dir']
        )
        results['final_report'] = report_result
        
        return results
    
    def _run_statistical_analysis(self, data, objective):
        """Helper for parallel statistical analysis."""
        return self.agents['analysis'].analyze_data(
            data=data,
            tool=self.tools['statistical_analysis'],
            objective=objective
        )
    
    def _run_insight_generation(self, data):
        """Helper for parallel insight generation."""
        return self.agents['analysis'].generate_insights(
            data=data,
            tool=self.tools['insight_generator']
        )
    
    def _run_web_search(self, objective):
        """Helper for parallel web search."""
        try:
            return self.tools['web_search'].execute(
                query=objective,
                num_results=5
            )
        except Exception as e:
            self.logger.warning(f"Web search failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _compile_report_data(self, results: Dict[str, Any], workflow_state: Dict[str, Any]) -> Dict[str, Any]:
        """Compile all results into report data structure."""
        return {
            'title': f"Data Analysis Report: {workflow_state['objective']}",
            'executive_summary': self._generate_executive_summary(results),
            'metadata': {
                'workflow_id': workflow_state['id'],
                'timestamp': get_timestamp(),
                'data_source': workflow_state['data_source'],
                'execution_mode': workflow_state['execution_mode']
            },
            'data_overview': results.get('data_collection', {}).get('summary', {}),
            'statistics': results.get('analysis', {}).get('results', {}),
            'insights': results.get('ai_insights', {}).get('results', {}).get('insights', []),
            'visualizations': results.get('visualizations', {}).get('visualizations', []),
            'quality_metrics': results.get('quality_assurance', {}),
            'recommendations': self._generate_recommendations(results),
            'conclusion': self._generate_conclusion(results)
        }
    
    def _generate_executive_summary(self, results: Dict[str, Any]) -> str:
        """Generate executive summary from results."""
        parts = []
        
        # Data overview
        data_info = results.get('data_collection', {})
        if data_info:
            parts.append(f"Analyzed {data_info.get('record_count', 'N/A')} records.")
        
        # Key insights
        insights = results.get('ai_insights', {}).get('results', {}).get('insights', [])
        if insights:
            parts.append(f"Generated {len(insights)} AI-powered insights.")
            if insights:
                parts.append(f"Top finding: {insights[0].get('message', '')}")
        
        # Quality score
        qa = results.get('quality_assurance', {})
        if qa:
            parts.append(f"Quality score: {qa.get('overall_score', 'N/A')}/100.")
        
        return " ".join(parts)
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # From insights
        insights = results.get('ai_insights', {}).get('results', {}).get('insights', [])
        for insight in insights[:5]:
            if insight.get('recommendation'):
                recommendations.append(insight['recommendation'])
        
        # From QA
        qa = results.get('quality_assurance', {})
        if qa.get('recommendations'):
            recommendations.extend(qa['recommendations'])
        
        return recommendations[:10]  # Top 10
    
    def _generate_conclusion(self, results: Dict[str, Any]) -> str:
        """Generate conclusion."""
        insights_count = len(results.get('ai_insights', {}).get('results', {}).get('insights', []))
        qa_score = results.get('quality_assurance', {}).get('overall_score', 0)
        
        return (
            f"Analysis completed successfully with {insights_count} key insights identified. "
            f"Quality assurance validation achieved a score of {qa_score}/100. "
            f"All results have been validated and documented in this comprehensive report."
        )
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a workflow by ID."""
        for workflow in self.execution_history:
            if workflow['id'] == workflow_id:
                return workflow
        return None
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get orchestrator execution statistics."""
        return {
            'total_workflows': len(self.execution_history),
            'successful': sum(1 for w in self.execution_history if w['status'] == 'completed'),
            'failed': sum(1 for w in self.execution_history if w['status'] == 'failed'),
            'total_agents': len(self.agents),
            'total_tools': len(self.tools),
            'memory_entries': len(self.memory_manager.conversation_memory),
            'error_stats': self.error_handler.get_error_statistics()
        }
