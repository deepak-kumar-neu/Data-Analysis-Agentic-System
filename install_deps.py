#!/usr/bin/env python3
"""
Quick dependency installer and app launcher
"""
import subprocess
import sys

print("🚀 Installing dependencies for Full UI Application...")
print("=" * 60)

# Core dependencies
print("\n📦 Installing core dependencies...")
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "streamlit", "plotly", "pandas", "numpy", "scipy", 
    "scikit-learn", "pydantic", "python-dotenv", 
    "requests", "beautifulsoup4", "matplotlib", "seaborn"
])
print("✅ Core dependencies installed")

# CrewAI and LangChain
print("\n📦 Installing CrewAI and LangChain...")
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "crewai", "crewai-tools", "langchain", 
    "langchain-openai", "langchain-community"
])
print("✅ CrewAI and LangChain installed")

print("\n✨ All dependencies installed successfully!")
print("\n🌐 You can now run the application with:")
print("   streamlit run src/ui/streamlit_app.py")
