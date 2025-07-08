#!/usr/bin/env python3
"""
Run script for the RAG PDF Chat System with PyTorch/Streamlit compatibility fixes.
"""

import os
import sys
import warnings
import subprocess

def setup_environment():
    """Setup environment variables and warning filters."""
    # Environment variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    
    # Warning filters
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning, message=".*torch.*")
    warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.*")
    
    print("✅ Environment configured for PyTorch/Streamlit compatibility")

def run_streamlit():
    """Run the Streamlit application."""
    try:
        print("🚀 Starting RAG PDF Chat System...")
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "main.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--server.headless", "true"
        ])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error running application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    setup_environment()
    run_streamlit() 