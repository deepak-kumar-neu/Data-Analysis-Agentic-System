# Examples Directory

This directory contains example scripts demonstrating how to use the Data Analysis Agentic System.

---

## 📁 Available Examples

### 1. `simple_example.py`

Complete demonstration script with three modes:

#### Default Mode - Simple Sequential Analysis
```bash
python examples/simple_example.py
```

**What it does:**
- Creates sample sales dataset (1000 records)
- Runs complete sequential workflow
- Generates AI-powered insights
- Creates visualizations
- Produces comprehensive reports

**Output:**
- Console summary of results
- Reports in `results/example/`
- Visualizations
- Metadata

#### Parallel Mode - Concurrent Execution
```bash
python examples/simple_example.py parallel
```

**What it does:**
- Uses existing or creates sample data
- Executes workflow in parallel mode
- Runs multiple analyses concurrently
- Shows execution statistics

**Output:**
- Performance metrics
- Results in `results/parallel_example/`

#### Custom Tool Mode - AI Insight Generator Demo
```bash
python examples/simple_example.py custom
```

**What it does:**
- Creates test dataset
- Demonstrates custom AI tool independently
- Shows insight generation capabilities
- Displays confidence scores and recommendations

**Output:**
- Console output of AI-generated insights
- Demonstrates custom tool features

---

## 🚀 Quick Start

### Prerequisites
```bash
# Make sure you're in the project root
cd data_analysis_agentic_system

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (if not done)
pip install -r requirements.txt
```

### Run Examples

**Simple Example:**
```bash
python examples/simple_example.py
```

**Parallel Execution:**
```bash
python examples/simple_example.py parallel
```

**Custom AI Tool:**
```bash
python examples/simple_example.py custom
```

---

## 📊 What to Expect

### Simple Example Output
```
📊 Creating sample dataset...
✅ Sample data created: data/sample_sales.csv
   - Records: 1000
   - Columns: ['date', 'product', 'region', 'sales', 'units', 'customer_satisfaction']

🤖 Initializing AI agents and tools...
🚀 Starting analysis workflow...

✅ Analysis completed successfully!

🧠 AI-Powered Insights Generated: 8

📌 Top 5 Insights:

1. sales is strongly positively correlated with units (r=0.845)
   Type: PATTERN
   Confidence: 0.85
   💡 Recommendation: Consider units when analyzing or predicting sales

2. Detected 10 outliers in sales (1.0% of data)
   Type: ANOMALY
   Confidence: 0.45
   💡 Recommendation: Investigate outliers in sales - they may indicate...

...

📁 All results saved to: /path/to/results/example
```

### Parallel Example Output
```
🚀 Starting parallel workflow...
   Multiple analysis tasks will run concurrently

✅ Parallel analysis completed!

📊 Execution Statistics:
   • Total workflows: 2
   • Successful: 2
   • Agents used: 6
   • Tools used: 7
```

### Custom Tool Example Output
```
🧠 Running AI-Powered Insight Generator...

✅ Generated 10 AI-powered insights:

1. [PATTERN] profit is very strong positively correlated with revenue (r=0.987)
   Confidence: 0.99 | Impact: 0.80
   💡 Consider revenue when analyzing or predicting profit

2. [ANOMALY] Detected 52 outliers in profit (10.4% of data)
   Confidence: 0.73 | Impact: 0.70
   💡 Investigate outliers in profit - they may indicate data quality issues

...

📋 Executive Summary:
   Generated 10 AI-powered insights from the data...
```

---

## 📝 Code Structure

### simple_example.py Functions

```python
def create_sample_data()
    """Create sample sales dataset"""

def run_simple_example()
    """Run complete sequential workflow"""

def run_parallel_example()
    """Run parallel execution workflow"""

def run_custom_tool_example()
    """Demonstrate custom AI tool"""
```

---

## 🎯 Learning Objectives

### From Simple Example
- Understand complete workflow execution
- See agent and tool coordination
- Learn result structure
- View multi-format reports

### From Parallel Example
- Understand concurrent execution
- See performance benefits
- Learn execution statistics

### From Custom Tool Example
- Understand AI-powered insight generation
- See confidence scoring
- Learn recommendation generation
- Understand custom tool capabilities

---

## 🔧 Customization

### Modify Sample Data
```python
# In simple_example.py
n_records = 1000  # Change to your desired size
```

### Change Analysis Objective
```python
result = orchestrator.execute_workflow(
    data_source='data/sales.csv',
    objective='Your custom objective here',  # Modify this
    ...
)
```

### Use Your Own Data
```python
result = orchestrator.execute_workflow(
    data_source='path/to/your/data.csv',  # Use your file
    objective='Analyze your data',
    ...
)
```

---

## 📁 Output Structure

After running examples:

```
results/
├── example/                          # Simple example outputs
│   ├── workflow_metadata.json
│   ├── analysis_report_*.json
│   ├── analysis_report_*.md
│   ├── analysis_report_*.html
│   └── visualizations/
│       ├── bar_chart_*.png
│       ├── scatter_plot_*.png
│       └── dashboard_*.png
│
└── parallel_example/                 # Parallel example outputs
    ├── workflow_metadata.json
    └── ...
```

---

## 🐛 Troubleshooting

### Import Errors
```bash
# Make sure you're in project root
cd data_analysis_agentic_system

# Install dependencies
pip install -r requirements.txt
```

### Data Directory Missing
```bash
# Create data directory
mkdir -p data

# Or let the script create it automatically
python examples/simple_example.py
```

### Permission Errors
```bash
# Make sure output directories are writable
chmod -R 755 results/
```

---

## 💡 Tips

1. **Start Simple:** Run the default example first to see the full workflow
2. **Check Logs:** View `logs/example.log` for detailed execution logs
3. **Explore Results:** Open generated HTML reports in your browser
4. **Modify Gradually:** Start with small changes to understand impact
5. **Use Verbose Mode:** Add `--verbose` flag to CLI for detailed logging

---

## 🎓 Next Steps

After running examples:

1. **Explore the generated reports** in `results/`
2. **Check the visualizations** to understand patterns
3. **Read the logs** to see workflow execution
4. **Try the CLI** with your own data:
   ```bash
   python src/main.py --data-source your_data.csv --mode parallel
   ```
5. **Review the code** in `examples/simple_example.py` to understand implementation

---

## 📚 Related Documentation

- [Main README](../README.md) - Project overview
- [User Guide](../docs/user_guide.md) - Comprehensive usage guide
- [API Documentation](../docs/api.md) - API reference
- [Custom Tool Docs](../docs/custom_tools.md) - AI tool details

---

**Happy Analyzing!** 🎉
