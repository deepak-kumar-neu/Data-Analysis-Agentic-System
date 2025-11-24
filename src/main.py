"""
Main entry point for the Data Analysis Agentic System.
Provides CLI interface and workflow execution.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional

from src.orchestration import Orchestrator, ExecutionMode
from src.config import load_config
from src.utils.logger import setup_logger, get_logger
from src.utils.helpers import get_timestamp, ensure_directory


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Data Analysis Agentic System - AI-Powered Data Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  python src/main.py --data-source data/sample.csv

  # Custom objective with parallel execution
  python src/main.py \\
    --data-source data/sales.csv \\
    --objective "Analyze quarterly sales trends" \\
    --mode parallel \\
    --output results/q4_analysis

  # Hierarchical execution with verbose logging
  python src/main.py \\
    --data-source data/customers.csv \\
    --objective "Customer segmentation analysis" \\
    --mode hierarchical \\
    --verbose \\
    --config configs/custom.yaml
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--data-source',
        type=str,
        required=True,
        help='Path or URL to data source (CSV, JSON, Excel, Parquet, etc.)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--objective',
        type=str,
        default='Comprehensive data analysis',
        help='Analysis objective or goal (default: "Comprehensive data analysis")'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['sequential', 'parallel', 'hierarchical'],
        default='sequential',
        help='Execution mode (default: sequential)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='./results',
        help='Output directory for results (default: ./results)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to custom configuration file (YAML)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        help='Path to log file (default: logs/analysis.log)'
    )
    
    parser.add_argument(
        '--no-report',
        action='store_true',
        help='Skip report generation'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        choices=['json', 'markdown', 'html', 'all'],
        default='all',
        help='Report output format (default: all)'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    log_level = 'DEBUG' if args.verbose else 'INFO'
    log_file = args.log_file or 'logs/analysis.log'
    setup_logger(level=log_level, log_file=log_file)
    
    logger = get_logger('main')
    logger.info("=" * 80)
    logger.info("Data Analysis Agentic System")
    logger.info("=" * 80)
    logger.info(f"Start time: {get_timestamp()}")
    logger.info(f"Data source: {args.data_source}")
    logger.info(f"Objective: {args.objective}")
    logger.info(f"Execution mode: {args.mode}")
    logger.info(f"Output directory: {args.output}")
    
    try:
        # Load configuration
        config = None
        if args.config:
            logger.info(f"Loading configuration from: {args.config}")
            config = load_config(args.config)
        else:
            logger.info("Using default configuration")
            config = load_config()
        
        # Initialize orchestrator
        logger.info("Initializing orchestrator...")
        orchestrator = Orchestrator(config=config)
        
        # Map execution mode
        execution_mode = {
            'sequential': ExecutionMode.SEQUENTIAL,
            'parallel': ExecutionMode.PARALLEL,
            'hierarchical': ExecutionMode.HIERARCHICAL
        }[args.mode]
        
        # Execute workflow
        logger.info("Starting workflow execution...")
        logger.info("-" * 80)
        
        result = orchestrator.execute_workflow(
            data_source=args.data_source,
            objective=args.objective,
            execution_mode=execution_mode,
            output_dir=args.output,
            generate_report=not args.no_report,
            report_format=args.format
        )
        
        logger.info("-" * 80)
        
        # Check result
        if result['success']:
            logger.info("✅ Workflow completed successfully!")
            logger.info(f"Workflow ID: {result['workflow_id']}")
            
            # Display summary
            if 'results' in result:
                display_summary(result['results'], logger)
            
            # Display output files
            if 'results' in result and 'report' in result['results']:
                display_outputs(result['results']['report'], args.output, logger)
            
            # Save workflow metadata
            save_workflow_metadata(result, args.output, logger)
            
            logger.info("-" * 80)
            logger.info(f"End time: {get_timestamp()}")
            logger.info("=" * 80)
            
            return 0
        else:
            logger.error("❌ Workflow failed!")
            logger.error(f"Error: {result.get('error', 'Unknown error')}")
            
            if 'partial_results' in result:
                logger.info("Partial results available:")
                display_summary(result['partial_results'], logger)
            
            return 1
            
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Workflow interrupted by user")
        return 130
        
    except Exception as e:
        logger.error(f"❌ Fatal error: {str(e)}", exc_info=True)
        return 1


def display_summary(results: dict, logger):
    """Display workflow results summary."""
    logger.info("\n📊 Workflow Summary:")
    
    # Data collection
    if 'data_collection' in results:
        dc = results['data_collection']
        logger.info(f"  • Data Collection: {dc.get('record_count', 'N/A')} records loaded")
    
    # Data processing
    if 'data_processing' in results:
        dp = results['data_processing']
        logger.info(f"  • Data Processing: Quality score {dp.get('quality_score', 'N/A')}/100")
    
    # Analysis
    if 'analysis' in results:
        analysis = results['analysis']
        logger.info(f"  • Statistical Analysis: {len(analysis.get('results', {}).get('insights', []))} insights")
    
    # AI Insights
    if 'ai_insights' in results:
        ai = results['ai_insights']
        insights = ai.get('results', {}).get('insights', [])
        logger.info(f"  • AI-Powered Insights: {len(insights)} generated")
        if insights:
            logger.info(f"    - Top insight: {insights[0].get('message', 'N/A')[:100]}...")
    
    # Visualizations
    if 'visualizations' in results:
        viz = results['visualizations']
        logger.info(f"  • Visualizations: {len(viz.get('visualizations', []))} created")
    
    # Quality Assurance
    if 'quality_assurance' in results:
        qa = results['quality_assurance']
        logger.info(f"  • Quality Assurance: Score {qa.get('overall_score', 'N/A')}/100")


def display_outputs(report_result: dict, output_dir: str, logger):
    """Display generated output files."""
    logger.info("\n📁 Generated Files:")
    
    if 'results' in report_result:
        reports = report_result['results'].get('generated_reports', [])
        for report in reports:
            logger.info(f"  • {report['format'].upper()}: {report['path']}")
    
    logger.info(f"\nAll results saved to: {Path(output_dir).absolute()}")


def save_workflow_metadata(result: dict, output_dir: str, logger):
    """Save workflow metadata to JSON file."""
    try:
        ensure_directory(output_dir)
        metadata_file = Path(output_dir) / 'workflow_metadata.json'
        
        metadata = {
            'workflow_id': result['workflow_id'],
            'timestamp': get_timestamp(),
            'success': result['success'],
            'execution_mode': result.get('metadata', {}).get('mode', 'unknown'),
            'statistics': {
                'total_insights': len(result.get('results', {}).get('ai_insights', {}).get('results', {}).get('insights', [])),
                'visualizations': len(result.get('results', {}).get('visualizations', {}).get('visualizations', [])),
            }
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"\n💾 Metadata saved: {metadata_file}")
        
    except Exception as e:
        logger.warning(f"Failed to save metadata: {str(e)}")


if __name__ == '__main__':
    sys.exit(main())
