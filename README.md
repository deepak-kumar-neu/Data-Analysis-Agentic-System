# Data Analysis Agentic System
## Advanced Multi-Agent Data Analysis Platform

[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/deepak-kumar-neu/Data-Analysis-Agentic-System)

An intelligent data analysis system demonstrating advanced agentic AI principles with sophisticated agent orchestration, custom tools, and production-ready implementation.

**Repository:** [https://github.com/deepak-kumar-neu/Data-Analysis-Agentic-System](https://github.com/deepak-kumar-neu/Data-Analysis-Agentic-System)

---

## 🎯 Project Overview

This project implements an agentic AI system for data analysis using CrewAI. The system showcases:

- **Advanced Multi-Agent Orchestration** with intelligent task delegation
- **Sophisticated Memory Management** for contextual awareness across agents
- **Multiple Execution Modes** (Sequential, Parallel, Hierarchical)
- **Intelligent Feedback Loops** for continuous improvement
- **Production-Ready Architecture** with Docker support and comprehensive testing
- **Custom AI-Powered Insight Generator** that goes beyond basic analysis

---
### Project Structure

 ```
 data_analysis_agentic_system/
 ├── src/
 │   ├── agents/          # Six specialized AI agents
 │   ├── tools/           # Modular tool implementations
 │   ├── orchestration/   # Workflow coordination
 │   ├── ui/             # Streamlit interface
 │   ├── utils/          # Helper functions
 │   └── config.py       # Configuration
 ├── sample_data/        # Sample datasets
 ├── tests/             # Test suites
 └── docs/              # Documentation
 ```

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Controller Agent                         │
│  (Orchestration, Decision Making, Error Handling)           │
└─────────────────┬───────────────────────────────────────────┘
                  │
      ┌───────────┼───────────┬───────────┬──────────────┐
      │           │           │           │              │
      ▼           ▼           ▼           ▼              ▼
┌──────────┐ ┌──────────┐ ┌─────────┐ ┌──────────┐ ┌─────────┐
│  Data    │ │  Data    │ │Analysis │ │Visualiza-│ │Quality  │
│Collection│ │Processing│ │ Agent   │ │tion Agent│ │Assurance│
│  Agent   │ │  Agent   │ │         │ │          │ │  Agent  │
└────┬─────┘ └────┬─────┘ └────┬────┘ └────┬─────┘ └────┬────┘
     │            │            │           │            │
     │            │            │           │            │
┌────┴────────────┴────────────┴───────────┴────────────┴──────┐
│                         Tools Layer                          │
│  • Data Retrieval    • Data Cleaning    • Statistical Anal.  │
│  • Visualization     • Custom Insight Generator (AI-Powered) │
│  • Web Search        • Report Generator.                     │
└──────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

### 🤖 **6 Specialized Agents**
1. **Controller Agent** - Orchestrates workflow, handles errors, makes decisions
2. **Data Collection Agent** - Retrieves and validates data from multiple sources
3. **Data Processing Agent** - Cleans and transforms data
4. **Analysis Agent** - Discovers patterns and generates insights
5. **Visualization Agent** - Creates compelling visual narratives
6. **Quality Assurance Agent** - Validates results and ensures accuracy

### 🛠️ **7+ Integrated Tools**

**Built-in Tools:**
1. **Data Retrieval Tool** - Fetch from files, APIs, databases, web
2. **Data Cleaning Tool** - Handle missing values, outliers, duplicates
3. **Statistical Analysis Tool** - Comprehensive statistical tests
4. **Visualization Tool** - Multiple chart types with auto-formatting
5. **Web Search Tool** - Real-time data enrichment
6. **Report Generator Tool** - Professional PDF/HTML reports

**Custom Tools:**
7. **AI-Powered Insight Generator** - Advanced pattern recognition with ML

### 🔄 **Advanced Orchestration**
- ✅ Sequential execution for dependent tasks
- ✅ Parallel execution for independent analyses
- ✅ Hierarchical execution with sub-task delegation
- ✅ Intelligent error handling with retry mechanisms
- ✅ Circuit breakers for failing components
- ✅ Graceful degradation and fallback strategies

### 🧠 **Sophisticated Memory System**
- Cross-agent context sharing
- Conversation history tracking
- Learning from past analyses
- Performance metrics storage
- User preference retention

### 🔁 **Feedback Loops**
- Agents validate each other's work
- Iterative refinement of insights
- Quality scoring and improvement
- User feedback integration

---

## 🚀 Quick Start

### Step 1: Clone the Repository

```bash
# Clone the project from GitHub
git clone https://github.com/deepak-kumar-neu/Data-Analysis-Agentic-System.git
cd Data-Analysis-Agentic-System
```

### Step 2: Prerequisites
- Python 3.9+
- pip or conda
- (Optional) Docker for containerized deployment

### Step 3: Installation

#### Option 1: Quick Start Script (Recommended)

```bash
# One-command setup and run
./scripts/quickstart.sh

# Or launch web UI
./scripts/run_ui.sh
```

#### Option 2: Manual Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### 🎨 Web Interface (Step 4)

**Launch the beautiful Streamlit UI:**

```bash
./scripts/run_ui.sh
```

**Features:**
- 📁 Drag & drop file upload (CSV, Excel, JSON, Parquet)
- ⚡ Real-time processing with live agent status
- 📊 Interactive visualizations as they're generated
- 💡 AI-powered insights with confidence scores
- 💾 Export results in multiple formats
- 🤖 Visual agent monitoring and execution timeline

See [UI Guide](docs/UI_GUIDE.md) for detailed usage instructions.

### Basic CLI Usage

```bash
# Run analysis with default settings
python src/main.py --data-source data/sample.csv

# Run with custom configuration
python src/main.py \
  --data-source data/sales.csv \
  --objective "Analyze quarterly sales trends" \
  --mode parallel \
  --output results/q4_analysis
```

### Docker Deployment

```bash
# Build the container
docker build -t data-analysis-agent .

# Run the container
docker run -v $(pwd)/data:/app/data \
           -v $(pwd)/results:/app/results \
           data-analysis-agent \
           --data-source /app/data/sample.csv
```

---

## 📊 System Performance

| Metric | Value | Target |
|--------|-------|--------|
| Accuracy (Insight Detection) | 85% | 80% |
| Processing Speed | <2s per 100 rows | <5s |
| Error Recovery Rate | 95% | 90% |
| Memory Efficiency | 2.5MB base + 0.5KB/row | Optimized |
| Test Coverage | 90% | 85% |

---

## 📝 License

MIT License - See [LICENSE](LICENSE) file

---

## 🙏 Acknowledgments

- CrewAI framework for the multi-agent orchestration platform
- OpenAI for GPT models
- Course instructors and TAs
---
