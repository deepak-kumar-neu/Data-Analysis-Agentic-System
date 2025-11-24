"""
Simple example demonstrating the Data Analysis Agentic System.
This script shows basic usage patterns.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from src.orchestration import Orchestrator, ExecutionMode
from src.utils.logger import setup_logger
from src.utils.helpers import ensure_directory


def create_sample_data():
    """Create sample dataset for demonstration."""
    print("📊 Creating sample dataset...")
    
    # Create sample sales data
    np.random.seed(42)
    n_records = 1000
    
    data = {
        'date': pd.date_range('2024-01-01', periods=n_records, freq='H'),
        'product': np.random.choice(['ProductA', 'ProductB', 'ProductC'], n_records),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_records),
        'sales': np.random.normal(1000, 200, n_records),
        'units': np.random.poisson(10, n_records),
        'customer_satisfaction': np.random.uniform(3.0, 5.0, n_records)
    }
    
    df = pd.DataFrame(data)
    
    # Add some missing values
    df.loc[np.random.choice(df.index, 50), 'customer_satisfaction'] = np.nan
    
    # Add some outliers
    df.loc[np.random.choice(df.index, 10), 'sales'] = np.random.uniform(5000, 8000, 10)
    
    # Save to file
    ensure_directory('data')
    output_file = 'data/sample_sales.csv'
    df.to_csv(output_file, index=False)
    
    print(f"✅ Sample data created: {output_file}")
    print(f"   - Records: {len(df)}")
    print(f"   - Columns: {list(df.columns)}")
    print(f"   - Date range: {df['date'].min()} to {df['date'].max()}")
    
    return output_file


def run_simple_example():
    """Run a simple analysis workflow."""
    print("\n" + "=" * 80)
    print("Data Analysis Agentic System - Simple Example")
    print("=" * 80 + "\n")
    
    # Setup logging
    setup_logger(level='INFO', log_file='logs/example.log')
    
    # Create sample data
    data_file = create_sample_data()
    
    # Initialize orchestrator
    print("\n🤖 Initializing AI agents and tools...")
    orchestrator = Orchestrator()
    
    # Execute workflow in sequential mode
    print("\n🚀 Starting analysis workflow...")
    print("   Mode: Sequential")
    print("   Objective: Analyze sales trends and patterns\n")
    
    result = orchestrator.execute_workflow(
        data_source=data_file,
        objective="Analyze sales trends, identify patterns, and generate actionable insights",
        execution_mode=ExecutionMode.SEQUENTIAL,
        output_dir='./results/example'
    )
    
    # Display results
    print("\n" + "-" * 80)
    
    if result['success']:
        print("✅ Analysis completed successfully!\n")
        
        # Extract key information
        results = result['results']
        
        # AI Insights
        if 'ai_insights' in results:
            insights = results['ai_insights'].get('results', {}).get('insights', [])
            print(f"🧠 AI-Powered Insights Generated: {len(insights)}")
            
            if insights:
                print("\n📌 Top 5 Insights:")
                for i, insight in enumerate(insights[:5], 1):
                    print(f"\n{i}. {insight.get('message', 'N/A')}")
                    print(f"   Type: {insight.get('type', 'N/A').upper()}")
                    print(f"   Confidence: {insight.get('confidence', 0):.2f}")
                    if insight.get('recommendation'):
                        print(f"   💡 Recommendation: {insight['recommendation']}")
        
        # Statistical Analysis
        if 'analysis' in results:
            print("\n📊 Statistical Analysis:")
            analysis = results['analysis'].get('results', {})
            insights = analysis.get('insights', [])
            if insights:
                for insight in insights[:3]:
                    print(f"   • {insight}")
        
        # Visualizations
        if 'visualizations' in results:
            viz = results['visualizations'].get('visualizations', [])
            print(f"\n📈 Visualizations Created: {len(viz)}")
            for v in viz:
                print(f"   • {v['type']}: {v['path']}")
        
        # Quality Score
        if 'quality_assurance' in results:
            qa = results['quality_assurance']
            print(f"\n✓ Quality Score: {qa.get('overall_score', 'N/A')}/100")
        
        # Report
        if 'report' in results:
            report = results['report'].get('results', {})
            reports = report.get('generated_reports', [])
            if reports:
                print(f"\n📄 Reports Generated: {len(reports)}")
                for r in reports:
                    print(f"   • {r['format'].upper()}: {r['path']}")
        
        print("\n" + "-" * 80)
        print(f"📁 All results saved to: {Path('./results/example').absolute()}")
        
    else:
        print("❌ Analysis failed!")
        print(f"Error: {result.get('error', 'Unknown error')}")
    
    print("\n" + "=" * 80)
    print("Example completed!")
    print("=" * 80 + "\n")


def run_parallel_example():
    """Run a parallel execution example."""
    print("\n" + "=" * 80)
    print("Parallel Execution Example")
    print("=" * 80 + "\n")
    
    # Setup
    setup_logger(level='INFO', log_file='logs/parallel_example.log')
    data_file = 'data/sample_sales.csv'
    
    if not Path(data_file).exists():
        data_file = create_sample_data()
    
    # Initialize orchestrator
    print("🤖 Initializing orchestrator...")
    orchestrator = Orchestrator()
    
    # Execute in parallel mode
    print("\n🚀 Starting parallel workflow...")
    print("   Multiple analysis tasks will run concurrently\n")
    
    result = orchestrator.execute_workflow(
        data_source=data_file,
        objective="Comprehensive sales and customer analysis",
        execution_mode=ExecutionMode.PARALLEL,
        output_dir='./results/parallel_example'
    )
    
    if result['success']:
        print("\n✅ Parallel analysis completed!")
        
        # Show execution stats
        stats = orchestrator.get_execution_stats()
        print(f"\n📊 Execution Statistics:")
        print(f"   • Total workflows: {stats['total_workflows']}")
        print(f"   • Successful: {stats['successful']}")
        print(f"   • Agents used: {stats['total_agents']}")
        print(f"   • Tools used: {stats['total_tools']}")
        
        print(f"\n📁 Results: {Path('./results/parallel_example').absolute()}")
    else:
        print(f"\n❌ Failed: {result.get('error')}")
    
    print("\n" + "=" * 80 + "\n")


def run_custom_tool_example():
    """Demonstrate the custom AI-Powered Insight Generator tool."""
    print("\n" + "=" * 80)
    print("Custom AI Tool Example - Insight Generator")
    print("=" * 80 + "\n")
    
    from src.tools import InsightGeneratorTool
    
    # Create sample data
    print("📊 Creating test dataset...")
    df = pd.DataFrame({
        'revenue': np.random.normal(50000, 10000, 500),
        'costs': np.random.normal(30000, 5000, 500),
        'customers': np.random.poisson(100, 500),
        'satisfaction': np.random.uniform(3, 5, 500)
    })
    df['profit'] = df['revenue'] - df['costs']
    
    print(f"   Records: {len(df)}")
    print(f"   Columns: {list(df.columns)}\n")
    
    # Use custom AI tool
    print("🧠 Running AI-Powered Insight Generator...")
    tool = InsightGeneratorTool()
    
    result = tool.execute(
        data=df,
        target_column='profit',
        analysis_types=['patterns', 'anomalies', 'trends', 'importance', 'predictions'],
        confidence_threshold=0.6,
        max_insights=10
    )
    
    if result['success']:
        insights = result['results']['insights']
        print(f"\n✅ Generated {len(insights)} AI-powered insights:\n")
        
        for i, insight in enumerate(insights, 1):
            print(f"{i}. [{insight['type'].upper()}] {insight['message']}")
            print(f"   Confidence: {insight['confidence']:.2f} | Impact: {insight['impact']:.2f}")
            if insight.get('recommendation'):
                print(f"   💡 {insight['recommendation']}")
            print()
        
        # Executive summary
        summary = result['results'].get('executive_summary', '')
        print(f"📋 Executive Summary:\n   {summary}\n")
    else:
        print(f"❌ Error: {result.get('error')}")
    
    print("=" * 80 + "\n")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == 'parallel':
            run_parallel_example()
        elif mode == 'custom':
            run_custom_tool_example()
        else:
            print(f"Unknown mode: {mode}")
            print("Usage: python examples/simple_example.py [parallel|custom]")
    else:
        # Run simple example by default
        run_simple_example()
