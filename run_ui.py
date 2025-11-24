"""
Launch script for Streamlit UI.
Simple entry point to start the web interface.
"""

import sys
from pathlib import Path
import subprocess

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def main():
    """Launch Streamlit application."""
    app_path = Path(__file__).parent / "src" / "ui" / "streamlit_app.py"
    
    print("🚀 Starting Data Analysis Agentic System UI...")
    print(f"📁 App location: {app_path}")
    print("🌐 Opening browser...")
    print("\nPress Ctrl+C to stop the server\n")
    
    # Launch Streamlit
    subprocess.run([
        "streamlit", "run",
        str(app_path),
        "--server.port=8501",
        "--server.address=localhost",
        "--browser.gatherUsageStats=false"
    ])


if __name__ == "__main__":
    main()
