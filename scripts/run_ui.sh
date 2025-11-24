#!/bin/bash

# Streamlit UI Launch Script
# Starts the web interface for the Data Analysis Agentic System

set -e

echo "🚀 Starting Data Analysis Agentic System UI..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "⚠️  Virtual environment not found. Run ./scripts/quickstart.sh first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if Streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "📦 Installing Streamlit..."
    pip install streamlit streamlit-aggrid plotly
fi

# Create temp directory for uploads
mkdir -p temp

# Set environment variables
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=localhost
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Launch Streamlit
echo "🌐 Launching UI at http://localhost:8501"
echo "Press Ctrl+C to stop"
echo ""

streamlit run src/ui/streamlit_app.py \
    --server.port=$STREAMLIT_SERVER_PORT \
    --server.address=$STREAMLIT_SERVER_ADDRESS \
    --browser.gatherUsageStats=$STREAMLIT_BROWSER_GATHER_USAGE_STATS
