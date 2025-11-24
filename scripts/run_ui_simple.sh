#!/bin/bash

# Quick Launch Script for Simplified Streamlit UI
# This version works without requiring CrewAI installation

set -e

echo "🎨 Launching Simplified Streamlit UI..."
echo ""

# Create temp directory
mkdir -p temp logs

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "📦 Streamlit not found. Installing..."
    pip install streamlit plotly pandas numpy
fi

# Set environment variables
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=localhost
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

echo "🌐 Launching UI at http://localhost:8501"
echo "📝 Using simplified UI (no CrewAI required)"
echo ""
echo "✨ Features:"
echo "   - File upload (CSV, Excel, JSON, Parquet)"
echo "   - Data preview and profiling"
echo "   - Simulated analysis workflow"
echo "   - Interactive visualizations"
echo "   - AI-powered insights"
echo "   - Export results"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Launch streamlit with simplified app
streamlit run src/ui/streamlit_app_simple.py \
    --server.port=$STREAMLIT_SERVER_PORT \
    --server.address=$STREAMLIT_SERVER_ADDRESS \
    --browser.gatherUsageStats=$STREAMLIT_BROWSER_GATHER_USAGE_STATS
