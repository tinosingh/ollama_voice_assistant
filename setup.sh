#!/bin/bash
# 
# Voice Assistant Setup Script
#
# This script ensures a Python virtual environment is created and
# dependencies are installed before running the main application.

# --- Configuration ---
VENV_DIR=".venv"
APP_FILE="app.py"

# --- Functions ---

# Function to check for Python executable
find_python() {
    if command -v python3 &> /dev/null; then
        echo "python3"
    elif command -v python &> /dev/null; then
        echo "python"
    else
        echo ""
    fi
}

# Function to run the main application
run_app() {
    # Activate the virtual environment
    if [ -d "$VENV_DIR" ]; then
        echo "-> Activating virtual environment..."
        # Use 'source' to activate the venv, checking both Linux/macOS and Windows paths
        source "$VENV_DIR/bin/activate" 2>/dev/null || source "$VENV_DIR/Scripts/activate" 2>/dev/null
    fi

    # Run the main Python application
    echo "-> Starting Voice Assistant (running initial setup if config.json not found)..."
    python "$APP_FILE"
}


# --- Main Logic ---

# 1. Check for Python
PYTHON_EXE=$(find_python)
if [ -z "$PYTHON_EXE" ]; then
    echo "Error: Python 3 is required but could not be found." >&2
    exit 1
fi

# 2. Check for Ollama
if ! command -v ollama &> /dev/null
then
    echo "Warning: Ollama is not installed or not in PATH."
    echo "Please ensure Ollama is installed and running before starting the assistant."
    echo "Proceeding with setup, but the application will likely fail without it."
    echo "Install instructions: https://ollama.com/download"
fi

# 3. Create or verify virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo "--- Creating virtual environment ($VENV_DIR) ---"
    $PYTHON_EXE -m venv "$VENV_DIR"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create virtual environment." >&2
        exit 1
    fi
    
    # Run the initial app setup (which installs dependencies and pulls models)
    run_app
else
    # Environment exists, just run the app
    run_app
fi

echo "Setup and run sequence completed."
