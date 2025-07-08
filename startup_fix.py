#!/usr/bin/env python3
"""
Startup script to fix PyTorch/Streamlit compatibility issues.
Run this before starting the Streamlit app.
"""

import os
import sys
import warnings

def apply_torch_streamlit_fixes():
    """Apply comprehensive fixes for PyTorch/Streamlit compatibility."""
    
    # Set environment variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["STREAMLIT_SERVER_ENABLE_STATIC_SERVING"] = "false"
    
    # Comprehensive warning suppression
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning, message=".*torch.*")
    warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.*")
    warnings.filterwarnings("ignore", message=".*torch.classes.*")
    warnings.filterwarnings("ignore", message=".*__path__._path.*")
    warnings.filterwarnings("ignore", message=".*_get_custom_class_python_wrapper.*")
    warnings.filterwarnings("ignore", message=".*no running event loop.*")
    warnings.filterwarnings("ignore", message=".*Tried to instantiate class.*")
    
    # Monkey patch Streamlit's file watcher
    try:
        import streamlit.watcher.local_sources_watcher as watcher
        
        def safe_extract_paths(module):
            """Safe version that handles torch modules."""
            try:
                if hasattr(module, '__name__') and 'torch' in str(module.__name__):
                    return []
                if hasattr(module, '__path__'):
                    if hasattr(module.__path__, '_path'):
                        return list(module.__path__._path)
                    elif hasattr(module.__path__, '__iter__'):
                        return list(module.__path__)
                return []
            except (RuntimeError, AttributeError, TypeError):
                return []
        
        # Replace the problematic function
        if hasattr(watcher, 'extract_paths'):
            watcher.extract_paths = safe_extract_paths
        
        # Also patch get_module_paths if it exists
        def safe_get_module_paths(module):
            """Safe version of get_module_paths."""
            try:
                return safe_extract_paths(module)
            except Exception:
                return []
        
        if hasattr(watcher, 'get_module_paths'):
            watcher.get_module_paths = safe_get_module_paths
            
    except ImportError:
        pass
    
    print("✅ Applied PyTorch/Streamlit compatibility fixes")

if __name__ == "__main__":
    apply_torch_streamlit_fixes()
    
    # Import and run the main app
    try:
        import main
        print("🚀 Starting Streamlit app...")
    except Exception as e:
        print(f"❌ Error starting app: {e}")
        sys.exit(1) 