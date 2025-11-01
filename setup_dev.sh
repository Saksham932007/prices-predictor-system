#!/bin/bash

# Development Environment Setup Script for House Price Prediction System

set -e  # Exit on any error

echo "🏠 Setting up House Price Prediction System Development Environment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if Python is installed
check_python() {
    echo -e "${BLUE}Checking Python installation...${NC}"
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d " " -f 2)
        echo -e "${GREEN}✅ Python ${PYTHON_VERSION} found${NC}"
    else
        echo -e "${RED}❌ Python 3 is not installed. Please install Python 3.8 or higher.${NC}"
        exit 1
    fi
}

# Check if pip is installed
check_pip() {
    echo -e "${BLUE}Checking pip installation...${NC}"
    if command -v pip3 &> /dev/null; then
        PIP_VERSION=$(pip3 --version | cut -d " " -f 2)
        echo -e "${GREEN}✅ pip ${PIP_VERSION} found${NC}"
    else
        echo -e "${RED}❌ pip is not installed. Please install pip.${NC}"
        exit 1
    fi
}

# Create virtual environment
create_venv() {
    echo -e "${BLUE}Creating virtual environment...${NC}"
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        echo -e "${GREEN}✅ Virtual environment created${NC}"
    else
        echo -e "${YELLOW}⚠️ Virtual environment already exists${NC}"
    fi
}

# Activate virtual environment
activate_venv() {
    echo -e "${BLUE}Activating virtual environment...${NC}"
    source venv/bin/activate
    echo -e "${GREEN}✅ Virtual environment activated${NC}"
}

# Install dependencies
install_dependencies() {
    echo -e "${BLUE}Installing dependencies...${NC}"
    pip install --upgrade pip
    pip install -r requirements.txt
    echo -e "${GREEN}✅ Dependencies installed${NC}"
}

# Create necessary directories
create_directories() {
    echo -e "${BLUE}Creating necessary directories...${NC}"
    
    directories=("logs" "results" "models" "data/processed" "data/raw")
    
    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            echo -e "${GREEN}✅ Created directory: $dir${NC}"
        else
            echo -e "${YELLOW}⚠️ Directory already exists: $dir${NC}"
        fi
    done
}

# Initialize MLflow
init_mlflow() {
    echo -e "${BLUE}Initializing MLflow...${NC}"
    
    # Create MLflow tracking directory if it doesn't exist
    if [ ! -d "mlruns" ]; then
        mkdir -p mlruns
        echo -e "${GREEN}✅ MLflow tracking directory created${NC}"
    fi
    
    echo -e "${GREEN}✅ MLflow initialized${NC}"
}

# Setup Git hooks (if in a git repository)
setup_git_hooks() {
    if [ -d ".git" ]; then
        echo -e "${BLUE}Setting up Git hooks...${NC}"
        
        # Create pre-commit hook for running tests
        cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
echo "Running tests before commit..."
cd prices-predictor-system
python -m pytest tests/ -v
if [ $? -eq 0 ]; then
    echo "✅ All tests passed"
else
    echo "❌ Tests failed. Commit aborted."
    exit 1
fi
EOF
        
        chmod +x .git/hooks/pre-commit
        echo -e "${GREEN}✅ Git hooks configured${NC}"
    else
        echo -e "${YELLOW}⚠️ Not a git repository, skipping Git hooks setup${NC}"
    fi
}

# Run tests to verify setup
run_tests() {
    echo -e "${BLUE}Running tests to verify setup...${NC}"
    cd prices-predictor-system
    
    if python -m pytest tests/ -v --tb=short; then
        echo -e "${GREEN}✅ All tests passed${NC}"
    else
        echo -e "${YELLOW}⚠️ Some tests failed, but setup is complete${NC}"
    fi
    
    cd ..
}

# Display completion message
completion_message() {
    echo
    echo -e "${GREEN}🎉 Development environment setup complete!${NC}"
    echo
    echo -e "${BLUE}Next steps:${NC}"
    echo "1. Activate the virtual environment: source venv/bin/activate"
    echo "2. Navigate to the project: cd prices-predictor-system"
    echo "3. Run the pipeline: python run_pipeline.py"
    echo "4. View MLflow UI: mlflow ui"
    echo "5. Run tests: pytest tests/"
    echo
    echo -e "${BLUE}Useful commands:${NC}"
    echo "- Run all tests: pytest tests/ -v"
    echo "- Run specific test: pytest tests/test_model_evaluator.py -v"
    echo "- Check code coverage: pytest tests/ --cov=src"
    echo "- Format code: black src/ tests/"
    echo "- Lint code: flake8 src/ tests/"
    echo
    echo -e "${GREEN}Happy coding! 🚀${NC}"
}

# Main setup function
main() {
    echo "Starting development environment setup..."
    echo
    
    check_python
    check_pip
    create_venv
    
    # Note: In a script, we need to source venv in the same shell
    echo -e "${BLUE}Installing dependencies in virtual environment...${NC}"
    
    # Create a temporary script to install dependencies
    cat > setup_temp.sh << 'EOF'
#!/bin/bash
source venv/bin/activate
pip install --upgrade pip
pip install -r prices-predictor-system/requirements.txt
echo "✅ Dependencies installed"
EOF
    
    chmod +x setup_temp.sh
    ./setup_temp.sh
    rm setup_temp.sh
    
    create_directories
    init_mlflow
    setup_git_hooks
    run_tests
    completion_message
}

# Check if we're in the right directory
if [ ! -f "prices-predictor-system/requirements.txt" ]; then
    echo -e "${RED}❌ Error: Please run this script from the project root directory${NC}"
    echo -e "${YELLOW}Expected to find: prices-predictor-system/requirements.txt${NC}"
    exit 1
fi

# Run main setup
main