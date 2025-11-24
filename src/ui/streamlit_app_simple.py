"""
Simplified Streamlit UI - Standalone Version
Works without requiring full orchestrator/CrewAI installation.
Perfect for UI testing and demonstration.
"""

import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import time
from datetime import datetime
from typing import Dict, Any, List
import io

# Page configuration
st.set_page_config(
    page_title="AI Data Analysis System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .agent-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'execution_log' not in st.session_state:
        st.session_state.execution_log = []
    if 'current_status' not in st.session_state:
        st.session_state.current_status = "Idle"
    if 'loaded_data' not in st.session_state:
        st.session_state.loaded_data = None
    if 'data_source' not in st.session_state:
        st.session_state.data_source = None


def render_header():
    """Render main header."""
    st.markdown('<h1 class="main-header">🤖 AI Data Analysis System</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Multi-Agent Agentic System for Intelligent Data Analysis</p>',
        unsafe_allow_html=True
    )
    st.markdown("---")


def render_sidebar():
    """Render sidebar configuration."""
    st.sidebar.title("⚙️ Configuration")
    
    st.sidebar.subheader("Execution Settings")
    execution_mode = st.sidebar.selectbox(
        "Execution Mode",
        options=["sequential", "parallel", "hierarchical"],
        help="Choose how agents should execute tasks"
    )
    
    objective = st.sidebar.text_area(
        "Analysis Objective",
        value="Comprehensive data analysis with insights and visualizations",
        help="Describe what you want to analyze"
    )
    
    with st.sidebar.expander("🔧 Advanced Settings"):
        enable_logging = st.checkbox("Enable Detailed Logging", value=True)
        max_retries = st.slider("Max Retries on Error", 0, 5, 3)
        
    return {
        'execution_mode': execution_mode,
        'objective': objective,
        'enable_logging': enable_logging,
        'max_retries': max_retries
    }


def render_file_upload():
    """Render file upload section."""
    st.subheader("📁 Data Upload")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload your data file",
            type=['csv', 'xlsx', 'json', 'parquet'],
            help="Supported formats: CSV, Excel, JSON, Parquet"
        )
        
        # Add buttons for sample data and clearing
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("📊 Load Sample Data", key="load_sample", use_container_width=True):
                st.session_state.data_source = "sample"
                st.session_state.analysis_results = None  # Clear previous results
                st.rerun()
        
        with btn_col2:
            if st.session_state.loaded_data is not None:
                if st.button("🗑️ Clear Data", key="clear_data", use_container_width=True):
                    st.session_state.loaded_data = None
                    st.session_state.data_source = None
                    st.session_state.analysis_results = None
                    st.rerun()
        
    with col2:
        st.info("""
        **Supported Sources:**
        - 📊 CSV files
        - 📈 Excel files
        - 🔗 JSON data
        - 🗄️ Parquet files
        
        **Or click "Load Sample Data"**
        """)
        
    return uploaded_file


def render_data_preview(data: pd.DataFrame):
    """Render data preview."""
    st.subheader("👀 Data Preview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", f"{len(data):,}")
    with col2:
        st.metric("Columns", f"{len(data.columns):,}")
    with col3:
        st.metric("Memory", f"{data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    with col4:
        missing_pct = (data.isnull().sum().sum() / (len(data) * len(data.columns)) * 100)
        st.metric("Missing Data", f"{missing_pct:.1f}%")
        
    tab1, tab2, tab3 = st.tabs(["📋 Data", "📊 Statistics", "🔍 Info"])
    
    with tab1:
        st.dataframe(data.head(100), use_container_width=True, height=400)
        
    with tab2:
        st.dataframe(data.describe(), use_container_width=True)
        
    with tab3:
        buffer = io.StringIO()
        data.info(buf=buffer)
        st.text(buffer.getvalue())


def simulate_analysis(data: pd.DataFrame, config: Dict[str, Any]):
    """Simulate analysis workflow."""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    steps = [
        ("Initializing workflow", 10),
        ("Loading data", 20),
        ("Processing data", 40),
        ("Running analysis", 60),
        ("Generating visualizations", 80),
        ("Creating report", 90),
        ("Finalizing results", 100)
    ]
    
    for step_name, progress in steps:
        status_text.text(f"⚡ {step_name}...")
        progress_bar.progress(progress / 100)
        time.sleep(0.5)
    
    # Calculate missing data percentage
    missing_pct = (data.isnull().sum().sum() / (len(data) * len(data.columns)) * 100)
        
    # Generate mock results
    results = {
        'execution_time': 3.5,
        'agents_count': 6,
        'tools_count': 7,
        'quality_score': 0.92,
        'processed_data': data,
        'insights': [
            {
                'title': 'Data Distribution Analysis',
                'description': f'Analyzed {len(data)} records across {len(data.columns)} features. Found normal distribution in numeric columns.',
                'confidence': 0.89
            },
            {
                'title': 'Correlation Patterns',
                'description': 'Identified strong correlations between related features. Key relationships discovered.',
                'confidence': 0.85
            },
            {
                'title': 'Missing Data Assessment',
                'description': f'Missing data: {missing_pct:.1f}%. Recommended imputation strategies identified.',
                'confidence': 0.91
            }
        ],
        'visualizations': generate_visualizations(data)
    }
    
    status_text.text("✅ Analysis complete!")
    progress_bar.progress(1.0)
    
    return results


def generate_visualizations(data: pd.DataFrame):
    """Generate visualizations from data."""
    viz = {}
    
    try:
        # Distribution plots
        numeric_cols = data.select_dtypes(include=['number']).columns[:3]
        viz['distributions'] = []
        
        if len(numeric_cols) > 0:
            for col in numeric_cols:
                try:
                    fig = px.histogram(data, x=col, title=f"Distribution of {col}", marginal="box")
                    viz['distributions'].append({
                        'title': f"Distribution: {col}",
                        'figure': fig
                    })
                except Exception as e:
                    st.warning(f"Could not create distribution plot for {col}: {e}")
        
        # Correlation heatmap
        if len(numeric_cols) > 1:
            try:
                corr_matrix = data[numeric_cols].corr()
                fig = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    title="Correlation Heatmap",
                    color_continuous_scale='RdBu',
                    aspect='auto'
                )
                viz['correlations'] = [{'title': 'Correlation Matrix', 'figure': fig}]
            except Exception as e:
                st.warning(f"Could not create correlation heatmap: {e}")
                
    except Exception as e:
        st.error(f"Error generating visualizations: {e}")
    
    return viz


def render_visualizations(results: Dict[str, Any]):
    """Render visualizations."""
    st.subheader("📊 Analysis Visualizations")
    
    if 'visualizations' not in results:
        st.info("No visualizations available")
        return
        
    viz = results['visualizations']
    
    # Distribution plots
    if 'distributions' in viz:
        st.markdown("### 📈 Distribution Analysis")
        for item in viz['distributions']:
            st.plotly_chart(item['figure'], use_container_width=True)
    
    # Correlations
    if 'correlations' in viz:
        st.markdown("### 🔗 Correlation Analysis")
        for item in viz['correlations']:
            st.plotly_chart(item['figure'], use_container_width=True)


def render_insights(results: Dict[str, Any]):
    """Render insights."""
    st.subheader("💡 AI-Generated Insights")
    
    if 'insights' not in results:
        st.info("No insights available")
        return
        
    insights = results['insights']
    
    for idx, insight in enumerate(insights, 1):
        with st.expander(f"Insight #{idx}: {insight['title']}", expanded=idx <= 3):
            st.markdown(insight['description'])
            
            if 'confidence' in insight:
                st.progress(insight['confidence'])
                st.caption(f"Confidence: {insight['confidence']*100:.1f}%")


def render_results_summary(results: Dict[str, Any]):
    """Render results summary."""
    st.subheader("📋 Results Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Processing Time", f"{results.get('execution_time', 0):.2f}s")
    with col2:
        st.metric("Agents Used", results.get('agents_count', 0))
    with col3:
        st.metric("Tools Executed", results.get('tools_count', 0))
    with col4:
        st.metric("Quality Score", f"{results.get('quality_score', 0)*100:.1f}%")


def render_export_options(results: Dict[str, Any]):
    """Render export options."""
    st.subheader("💾 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        json_data = json.dumps({k: v for k, v in results.items() if k != 'processed_data'}, indent=2, default=str)
        st.download_button(
            label="📥 Download JSON",
            data=json_data,
            file_name=f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
        
    with col2:
        if 'processed_data' in results:
            csv_data = results['processed_data'].to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name=f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )


def main():
    """Main application."""
    initialize_session_state()
    render_header()
    config = render_sidebar()
    
    uploaded_file = render_file_upload()
    
    # Handle both uploaded files and sample data using session state
    data = None
    
    # Check if we should load sample data
    if st.session_state.data_source == "sample":
        # Load sample data
        try:
            # Try multiple possible paths for the sample data
            possible_paths = [
                Path("sample_data/employees.csv"),
                Path(__file__).parent.parent.parent / "sample_data" / "employees.csv",
                Path.cwd() / "sample_data" / "employees.csv",
            ]
            
            sample_path = None
            for path in possible_paths:
                if path.exists():
                    sample_path = path
                    break
            
            if sample_path:
                data = pd.read_csv(sample_path)
                st.session_state.loaded_data = data
                st.success(f"✅ Loaded sample data: {len(data)} rows, {len(data.columns)} columns")
            else:
                st.error(f"❌ Sample data file not found. Searched paths: {[str(p) for p in possible_paths]}")
                st.info("Please upload a file instead.")
                st.session_state.data_source = None
        except Exception as e:
            st.error(f"❌ Error loading sample data: {e}")
            import traceback
            with st.expander("🔍 Error Details"):
                st.code(traceback.format_exc())
            st.session_state.data_source = None
    
    elif uploaded_file is not None:
        # Load uploaded data
        try:
            # Show file details for debugging
            st.info(f"📄 File: {uploaded_file.name} ({uploaded_file.size} bytes, type: {uploaded_file.type})")
            
            # Reset file pointer to beginning
            uploaded_file.seek(0)
            
            if uploaded_file.name.endswith('.csv'):
                # Try different encodings for CSV
                try:
                    data = pd.read_csv(uploaded_file)
                except UnicodeDecodeError:
                    uploaded_file.seek(0)
                    data = pd.read_csv(uploaded_file, encoding='latin-1')
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                data = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.json'):
                data = pd.read_json(uploaded_file)
            elif uploaded_file.name.endswith('.parquet'):
                data = pd.read_parquet(uploaded_file)
            else:
                st.error("❌ Unsupported file format")
                return
            
            # Check if data loaded successfully
            if data is None or data.empty:
                st.error("❌ File appears to be empty or unreadable")
                return
            
            # Store in session state
            st.session_state.loaded_data = data
            st.session_state.data_source = "upload"
            st.success(f"✅ Successfully loaded {len(data)} rows and {len(data.columns)} columns")
            
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            st.error(f"Error type: {type(e).__name__}")
            
            # Show detailed error for debugging
            with st.expander("🔍 Detailed Error Information"):
                import traceback
                st.code(traceback.format_exc())
                
            # Provide helpful suggestions
            st.info("""
            **Troubleshooting Tips:**
            - Make sure the file is not corrupted
            - For CSV files, check that it's properly formatted
            - For Excel files, ensure it's a valid .xlsx or .xls file
            - Try the "Load Sample Data" button to test the interface
            """)
            st.session_state.loaded_data = None
    
    # Use data from session state if available
    if st.session_state.loaded_data is not None:
        data = st.session_state.loaded_data
            
    # If data was loaded (either from upload or sample), show preview and analysis
    if data is not None:
        try:
            render_data_preview(data)
            
            # Analysis controls
            st.markdown("---")
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.markdown("### 🚀 Ready to Analyze")
                
            with col2:
                run_button = st.button("▶️ Run Analysis", type="primary", use_container_width=True)
                
            with col3:
                if st.session_state.analysis_results:
                    if st.button("🔄 Reset", use_container_width=True):
                        st.session_state.analysis_results = None
                        st.rerun()
                        
            # Run analysis
            if run_button:
                st.markdown("---")
                results = simulate_analysis(data, config)
                st.session_state.analysis_results = results
                st.success("✅ Analysis completed successfully!")
                st.balloons()
                
            # Display results
            if st.session_state.analysis_results:
                st.markdown("---")
                
                result_tabs = st.tabs(["📊 Visualizations", "💡 Insights", "📋 Summary", "💾 Export"])
                
                with result_tabs[0]:
                    render_visualizations(st.session_state.analysis_results)
                    
                with result_tabs[1]:
                    render_insights(st.session_state.analysis_results)
                    
                with result_tabs[2]:
                    render_results_summary(st.session_state.analysis_results)
                    
                with result_tabs[3]:
                    render_export_options(st.session_state.analysis_results)
                    
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            
            # Show detailed error for debugging
            with st.expander("🔍 Detailed Error Information"):
                import traceback
                st.code(traceback.format_exc())
            
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666; padding: 2rem;'>
            <p>🤖 Powered by Multi-Agent AI System | Built with Streamlit</p>
            <p>© 2025 Data Analysis Agentic System (Demo Mode)</p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
