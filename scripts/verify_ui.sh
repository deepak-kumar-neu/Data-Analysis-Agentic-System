#!/bin/bash

# UI Setup Verification Script
# Checks if all dependencies are installed for the Streamlit UI

set -e

echo "🔍 Verifying Streamlit UI Setup..."
echo ""

# Check Python version
echo "1️⃣  Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "   ✓ Python $python_version"

# Check virtual environment
echo ""
echo "2️⃣  Checking virtual environment..."
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "   ✓ Virtual environment active: $VIRTUAL_ENV"
else
    echo "   ⚠️  No virtual environment detected"
    echo "      Run: python -m venv venv && source venv/bin/activate"
fi

# Check required packages
echo ""
echo "3️⃣  Checking required packages..."

packages=(
    "streamlit"
    "pandas"
    "plotly"
    "numpy"
)

all_installed=true

for package in "${packages[@]}"; do
    if python -c "import $package" 2>/dev/null; then
        version=$(python -c "import $package; print($package.__version__)" 2>/dev/null || echo "unknown")
        echo "   ✓ $package ($version)"
    else
        echo "   ✗ $package (not installed)"
        all_installed=false
    fi
done

# Check Streamlit executable
echo ""
echo "4️⃣  Checking Streamlit executable..."
if command -v streamlit &> /dev/null; then
    streamlit_version=$(streamlit --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
    echo "   ✓ Streamlit CLI available ($streamlit_version)"
else
    echo "   ✗ Streamlit CLI not found"
    all_installed=false
fi

# Check required directories
echo ""
echo "5️⃣  Checking directory structure..."

directories=(
    "src/ui"
    "sample_data"
    ".streamlit"
    "temp"
    "logs"
)

for dir in "${directories[@]}"; do
    if [ -d "$dir" ]; then
        echo "   ✓ $dir/"
    else
        echo "   ⚠️  $dir/ (missing, will be created)"
        mkdir -p "$dir"
    fi
done

# Check required files
echo ""
echo "6️⃣  Checking required files..."

files=(
    "src/ui/streamlit_app.py"
    "src/ui/components.py"
    "src/ui/utils.py"
    ".streamlit/config.toml"
    "scripts/run_ui.sh"
)

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "   ✓ $file"
    else
        echo "   ✗ $file (missing)"
        all_installed=false
    fi
done

# Check sample data
echo ""
echo "7️⃣  Checking sample data..."
if [ -f "sample_data/employees.csv" ]; then
    lines=$(wc -l < "sample_data/employees.csv")
    echo "   ✓ sample_data/employees.csv ($lines rows)"
else
    echo "   ⚠️  No sample data found"
fi

# Summary
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ "$all_installed" = true ]; then
    echo "✅ All checks passed! UI is ready to launch."
    echo ""
    echo "Launch the UI with:"
    echo "   ./scripts/run_ui.sh"
    echo ""
    echo "Or:"
    echo "   streamlit run src/ui/streamlit_app.py"
else
    echo "⚠️  Some dependencies are missing."
    echo ""
    echo "Install missing packages:"
    echo "   pip install -r requirements.txt"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
